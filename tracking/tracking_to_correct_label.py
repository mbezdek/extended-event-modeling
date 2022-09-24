from typing import Dict, Tuple
import traceback

import cv2
import numpy as np
import torch
# torch.backends.cudnn.enabled=False
import sys

import os

sys.path.append(os.getcwd())
sys.path.append('../pysot')
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from scipy.optimize import linear_sum_assignment

import csv
import pandas as pd
from time import perf_counter
from copy import deepcopy
from utils import BoxWrapper, FrameWrapper, CV2VideoWriter, CV2VideoReader, \
    ColorBGR, Sample, logger, parse_config, ColorRef


class TrackerWrapper:
    """
    This class keep track of tracker_algorithm and boxes over time
    Each instance of this class tracks one object
    """

    def __init__(self, box_wrapper: BoxWrapper, frame: np.ndarray,
                 tracker_type: str = 'notrack', model_config: str = '', model_path: str = ''):
        tracker_type = tracker_type
        if tracker_type == 'siam':
            # init siamrpn tracker
            logger.debug(f'Building siamrpn')
            cfg.merge_from_file(model_config)
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            if cfg.CUDA:
                cuda_id = np.random.randint(torch.cuda.device_count())
                self.cuda_id = cuda_id
                self.device = torch.device(f'cuda:{self.cuda_id}')
                with torch.cuda.device(self.cuda_id):
                    model = ModelBuilder()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval().to(self.device)
                    siam_tracker = build_tracker(model)
                    siam_tracker.init(frame, box_wrapper.get_xywh())
            else:
                self.device = torch.device('cpu')
                model = ModelBuilder()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval().to(self.device)
                siam_tracker = build_tracker(model)
                siam_tracker.init(frame, box_wrapper.get_xywh())

            self.tracker = siam_tracker
        else:
            self.tracker = None
            logger.error(f'Unknown tracker type: {tracker_type}')
        self.boxes = [box_wrapper]
        self.active = True
        self.terminate = False
        self.first_frame = frame
        self.no_hit = 0
        self.no_hit_threshold = 30
        self.no_match_intervals = 0
        self.no_match_threshold = 1

    def __str__(self) -> str:
        return f'track_name={self.get_track_name()}, active={self.active}, ' \
               f'terminate={self.terminate}'

    def get_track_name(self):
        return self.boxes[0].object_name

    def predict_next_box(self, frame: np.ndarray) -> Dict:
        """
        This method receive a frame and return coordinates of its object on that frame
        :param frame:
        :return: outputs: object's coordinates and confidence score
        """
        if self.device != torch.device('cpu'):
            with torch.cuda.device(self.cuda_id):
                outputs = self.tracker.track(frame)
        else:
            outputs = self.tracker.track(frame)
        return outputs

    def append_box_wrapper(self, box_wrapper: BoxWrapper) -> None:
        """
        Update boxes
        :param box_wrapper:
        :return:
        """
        self.boxes.append(box_wrapper)

    def re_init(self, box_wrapper: BoxWrapper, frame: np.ndarray) -> None:
        self.active = True
        if self.device != torch.device('cpu'):
            with torch.cuda.device(self.cuda_id):
                self.tracker.init(frame, box_wrapper.get_xywh())
        else:
            self.tracker.init(frame, box_wrapper.get_xywh())

    def deactivate_track(self) -> None:
        self.active = False

    def release_tracker(self) -> None:
        logger.info(f'Releasing {self.get_track_name()}')
        self.terminate = True
        self.tracker.model = None
        self.tracker = None

    def change_name(self, object_name: str) -> None:
        for box in self.boxes:
            box.object_name = object_name

    def sort_boxes(self) -> None:
        self.boxes = sorted(self.boxes, key=lambda box_wrapper: box_wrapper.frame_id)


class Context:
    """
    This class keep track of active tracks and inactive tracks
    """

    def __init__(self, track_kwargs: dict, color_reference: ColorRef):
        self.tracks = dict()
        self.frame_results = dict()
        self.track_kwargs = track_kwargs
        self.color_reference = color_reference

    def matching(self, boxes: np.ndarray, object_type: str,
                 frame_wrapper: FrameWrapper) -> None:
        self.frame_results[frame_wrapper.frame_id][object_type] = dict()
        distance_matrix = np.array([])
        if object_type not in self.tracks.keys():
            self.tracks[object_type] = dict()
            row_ind = col_ind = []
        else:
            for object_id, tracks in sorted(self.tracks[object_type].items(),
                                            key=lambda kv: kv[0]):
                last_box = np.array(tracks.boxes[-1].get_xyxy())
                dists = np.linalg.norm(boxes[:, :4] - last_box[:4], axis=1)
                distance_matrix = np.hstack([distance_matrix, dists])
            distance_matrix = distance_matrix.reshape(len(self.tracks[object_type]),
                                                      len(boxes))
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matched_boxes = []
        # Updating old tracks
        for row, col in zip(row_ind, col_ind):
            if distance_matrix[row, col] < 200:
                logger.info(
                    f'FrameID {frame_wrapper.frame_id:4d}: match, re-initializing ')
                matched_boxes.append(col)
                object_name = object_type + str(row)
                box_wrapper = BoxWrapper(xmin=boxes[col][0], ymin=boxes[col][1],
                                         xmax=boxes[col][2], ymax=boxes[col][3],
                                         frame_id=frame_wrapper.frame_id,
                                         object_name=object_name,
                                         conf_score=1.0, state='match',
                                         color=self.color_reference.color_dict['match'])
                self.tracks[object_type][row].re_init(frame=frame_wrapper.frame,
                                                      box_wrapper=box_wrapper)
                self.tracks[object_type][row].boxes[-1] = box_wrapper
                self.frame_results[frame_wrapper.frame_id][object_type][
                    row] = box_wrapper.get_xyxy() + [box_wrapper.conf_score]
                frame_wrapper.put_bbox(bbox=box_wrapper,
                                       color=box_wrapper.color)
        # Creating new tracks
        for col, box in enumerate(boxes):
            if col not in col_ind or col not in matched_boxes:
                logger.info(
                    f'FrameID {frame_wrapper.frame_id:4d}: Init track '
                    f'{object_type + str(len(self.tracks[object_type]))} at {boxes[col]}')
                object_name = object_type + str(len(self.tracks[object_type]))
                box_wrapper = BoxWrapper(xmin=boxes[col][0], ymin=boxes[col][1],
                                         xmax=boxes[col][2], ymax=boxes[col][3],
                                         frame_id=frame_wrapper.frame_id,
                                         object_name=object_name,
                                         conf_score=1.0, state='init',
                                         color=self.color_reference.color_dict['init'])
                track_kwargs = self.track_kwargs
                track_kwargs['frame'] = frame_wrapper.frame
                track_kwargs['box_wrapper'] = box_wrapper
                self.frame_results[frame_wrapper.frame_id][object_type][
                    (len(self.tracks[object_type]))] = boxes[col]
                self.tracks[object_type][
                    (len(self.tracks[object_type]))] = TrackerWrapper(**track_kwargs)
                frame_wrapper.put_bbox(bbox=box_wrapper,
                                       color=box_wrapper.color)

    def tracking(self, object_type: str, frame_wrapper: FrameWrapper,
                 conf_threshold=0.4) -> None:
        if object_type not in self.tracks:
            logger.debug(f'Tracking: no instance of {object_type} initialized')
            return
        category_track = self.tracks[object_type]
        # Init
        self.frame_results[frame_wrapper.frame_id][object_type] = dict()
        for object_id, track_wrapper in category_track.items():
            if not track_wrapper.active:
                continue
            logger.debug(f'Tracking {object_type + str(object_id)}')
            outputs = track_wrapper.predict_next_box(frame_wrapper.frame)
            object_name = object_type + str(object_id)
            box_wrapper = BoxWrapper(xmin=outputs['bbox'][0], ymin=outputs['bbox'][1],
                                     xmax=outputs['bbox'][0] + outputs['bbox'][2],
                                     ymax=outputs['bbox'][1] + outputs['bbox'][3],
                                     frame_id=frame_wrapper.frame_id, object_name=object_name,
                                     conf_score=outputs['best_score'], state='track',
                                     color=self.color_reference.color_dict['track'])
            self.tracks[object_type][object_id].append_box_wrapper(box_wrapper)
            if box_wrapper.conf_score < conf_threshold:
                track_wrapper.no_hit += 1
                if track_wrapper.no_hit > track_wrapper.no_hit_threshold:
                    logger.info(f'FrameID {frame_wrapper.frame_id}: '
                                f'deactivate track {object_type + str(object_id)}')
                    track_wrapper.deactivate_track()
                    box_wrapper.state = 'deactivate'
                    box_wrapper.color = self.color_reference.color_dict['deactivate']
            # else:
            # reset count
            # track_wrapper.no_hit = 0
            self.frame_results[frame_wrapper.frame_id][object_type][
                object_id] = box_wrapper.get_xyxy() + [box_wrapper.conf_score]
            frame_wrapper.put_bbox(bbox=box_wrapper,
                                   color=box_wrapper.color)

    def merge_bw_track(self, object_type: str, backward_track: TrackerWrapper):
        backward_track.change_name(object_type + str(len(self.tracks[object_type])))
        backward_track.sort_boxes()
        backward_track.re_init(backward_track.boxes[-1], backward_track.first_frame)
        self.tracks[object_type][len(self.tracks[object_type])] = backward_track


def track_buffer(context: Context, buffer_frames: dict,
                 label_df, labeled_frames, is_backward=False, fps=30) -> Context:
    if not is_backward:
        # Delete label frame while tracking forward
        del buffer_frames[max(buffer_frames.keys())]
    # Create video writer and reader

    for frame_id, frame in sorted(buffer_frames.items(), key=lambda kv: kv[0],
                                  reverse=is_backward):
        frame_wrapper = FrameWrapper(frame=frame, frame_id=frame_id)
        context.frame_results[frame_wrapper.frame_id] = dict()
        if frame_id in labeled_frames:  # If this is a label frame
            df_current = label_df[
                (label_df['index'] == round(frame_id / fps))]
            object_types = set(df_current['class'].unique())
            for object_type in object_types:
                df_category = df_current[df_current['class'] == object_type]
                boxes = df_category[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
                # no need to run tracking, only backward go into this condition,
                # and matching in this case only does initialization
                # context.tracking(object_type, frame_wrapper)
                context.matching(boxes, object_type, frame_wrapper)
        else:  # If this is not a label frame
            for object_type in context.tracks.keys():
                context.tracking(object_type, frame_wrapper)

    return context


def matching_and_merging(context_forward: Context, context_backward: Context,
                         agreement_threshold=0.5) -> Context:
    forward_categories = set(context_forward.tracks.keys())
    backward_categories = set(context_backward.tracks.keys())
    shared_categories = forward_categories.intersection(backward_categories)
    new_categories = backward_categories.difference(forward_categories)
    old_categories = forward_categories.difference(backward_categories)

    # Process shared categories between fw and bw
    for object_type in shared_categories:
        n_fw_tracks = len(context_forward.tracks[object_type])
        n_bw_tracks = len(context_backward.tracks[object_type])
        distance_matrix = np.full(shape=(n_fw_tracks, n_bw_tracks), fill_value=10000,
                                  dtype=np.float)
        for fw_object_id, fw_track in context_forward.tracks[object_type].items():
            if fw_track.terminate:
                logger.info(f'FW {object_type + str(fw_object_id)} terminated last interval')
                continue
            for bw_object_id, bw_track in context_backward.tracks[object_type].items():
                logger.info(
                    f'Compare FW {object_type + str(fw_object_id)} with BW {object_type + str(bw_object_id)}')
                distance_matrix[fw_object_id][bw_object_id] = compare_tracks(fw_track,
                                                                             bw_track)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matched_bw_tracks = []
        matched_fw_tracks = []
        # Process matched pairs
        for row, col in zip(row_ind, col_ind):
            if 1 - distance_matrix[row][col] > agreement_threshold:
                logger.info(
                    f'Merge FW {object_type + str(row)} with BW {object_type + str(col)} '
                    f'Matching score: {1 - distance_matrix[row][col]}')
                bw_track = context_backward.tracks[object_type][col]
                fw_track = context_forward.tracks[object_type][row]
                fw_track.re_init(frame=bw_track.first_frame, box_wrapper=bw_track.boxes[0])
                fw_track.no_match_intervals = 0
                bw_track.change_name(fw_track.get_track_name())
                merge_boxes(context_forward.tracks[object_type][row],
                            context_backward.tracks[object_type][col])
                matched_bw_tracks.append(col)
                matched_fw_tracks.append(row)
                bw_track.terminate = True  # Mark to release after

        # Process unmatched fw tracks
        for fw_object_id, fw_track in context_forward.tracks[object_type].items():
            if fw_object_id not in row_ind or fw_object_id not in matched_fw_tracks:
                if not fw_track.active and fw_track.tracker is not None:
                    logger.info(f'Inactive fw track {fw_track.get_track_name()} does not match'
                                f' any bw track, Terminate tracker to release memory')
                    fw_track.release_tracker()
                elif fw_track.active:
                    fw_track.no_match_intervals += 1
                    if fw_track.no_match_intervals >= fw_track.no_match_threshold:
                        logger.info(
                            f'Active fw track {fw_track.get_track_name()} does not match'
                            f' any bw track for {fw_track.no_match_threshold} intervals,'
                            f'Deactivate and Terminate')
                        fw_track.deactivate_track()
                        fw_track.release_tracker()
                    else:
                        logger.info(
                            f'Active fw track {fw_track.get_track_name()} does not match'
                            f' any bw track, duplicate last box for indexing')
                        box_wrapper = deepcopy(fw_track.boxes[-1])
                        box_wrapper.frame_id = fw_track.boxes[-1].frame_id + 1
                        fw_track.append_box_wrapper(box_wrapper=box_wrapper)
        # Process unmatched bw tracks
        for bw_object_id, bw_track in context_backward.tracks[object_type].items():
            if bw_object_id not in col_ind or bw_object_id not in matched_bw_tracks:
                logger.info(
                    f'Merge new bw track {bw_track.get_track_name()} to fw as {object_type + str(len(context_forward.tracks[object_type]))}')
                context_forward.merge_bw_track(object_type, bw_track)

    # Process categories that bw doesn't have
    for object_type in old_categories:
        for fw_object_id, fw_track in context_forward.tracks[object_type].items():
            if fw_track.active:
                fw_track.no_match_intervals += 1
                if fw_track.no_match_intervals >= fw_track.no_match_threshold:
                    logger.info(
                        f'Active fw category {fw_track.get_track_name()} does not match'
                        f' any bw track for {fw_track.no_match_threshold} intervals,'
                        f'Deactivate and Terminate')
                    fw_track.deactivate_track()
                    fw_track.release_tracker()
                else:
                    logger.info(
                        f'Active fw category {fw_track.get_track_name()} does not match'
                        f' any bw category, duplicate last box for indexing')
                    box_wrapper = deepcopy(fw_track.boxes[-1])
                    box_wrapper.frame_id = fw_track.boxes[-1].frame_id + 1
                    fw_track.append_box_wrapper(box_wrapper=box_wrapper)
            elif fw_track.tracker is not None:
                logger.info(f'Inactive fw category {fw_track.get_track_name()} does not match'
                            f' any bw category, Terminate tracker to release memory')
                fw_track.release_tracker()

    # Process categories that fw doesn't have
    for object_type in new_categories:
        logger.info(f'Init new category FW {object_type}')
        context_forward.tracks[object_type] = dict()
        for bw_object_id, bw_track in context_backward.tracks[object_type].items():
            logger.info(f'Merge new BW {bw_track.get_track_name()} into fw')
            context_forward.merge_bw_track(object_type, bw_track)
    return context_forward


def draw_context_on_frames(context: Context, buffer_frames: Dict,
                           cv2_video_writer: CV2VideoWriter, labeled_frames=[]) -> None:
    """
    This functions draw all tracks from a context on a list of frames
    :param context:
    :param buffer_frames:
    :param cv2_video_writer:
    :param labeled_frames:
    :return:
    """
    # going over all tracks
    for object_type, category_track in context.tracks.items():
        for object_id, track_wrapper in category_track.items():
            for box_wrapper in track_wrapper.boxes:
                if box_wrapper.frame_id in buffer_frames:
                    frame = buffer_frames[box_wrapper.frame_id]
                    frame_wrapper = FrameWrapper(frame, frame_id=box_wrapper.frame_id)
                    frame_wrapper.put_bbox(box_wrapper,
                                           color=box_wrapper.color)

    for frame_id, frame in buffer_frames.items():
        frame_wrapper = FrameWrapper(frame, frame_id=frame_id)
        frame_wrapper.put_text(f'FrameID {frame_id}')
        frame_wrapper.put_text(f'Second {frame_id // 30}')
        if frame_id in labeled_frames:
            frame_wrapper.put_text(f'LABEL!!!', color=ColorBGR.cyan)
        cv2_video_writer.write_frame(frame)


def print_context(context: Context):
    log_str = ''
    num_tracks = 0
    num_active = 0
    num_terminate = 0
    for object_type, tracks in context.tracks.items():
        for object_id, track in tracks.items():
            num_tracks += 1
            num_active += int(track.active)
            num_terminate += int(track.terminate)
            log_str += str(track) + '\n'
    log_str += f'Stats: num_tracks: {num_tracks}, active {num_active}, terminate {num_terminate}'
    logger.info(log_str)
    return log_str


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_temporal_tracks(forward_track: TrackerWrapper,
                        backward_track: TrackerWrapper) -> Tuple:
    """
    This function gets temporal information from forward_track and backward_track
    :param forward_track:
    :param backward_track:
    :return:
    """
    fw_start = forward_track.boxes[0].frame_id
    fw_stop = forward_track.boxes[-1].frame_id
    bw_start = backward_track.boxes[-1].frame_id
    bw_stop = backward_track.boxes[0].frame_id
    intersection_start = max(bw_start, fw_start)
    intersection_stop = min(fw_stop, bw_stop)
    logger.info(
        f'fw start {fw_start} - fw stop {fw_stop} - bw start {bw_start} - bw stop {bw_stop}')
    logger.info(
        f'intersection start {intersection_start} - intersection stop {intersection_stop}')
    return fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop


def compare_tracks(forward_track: TrackerWrapper, backward_track: TrackerWrapper,
                   iou_threshold=0.2) -> float:
    """
    This functions measures agreement score between forward_track and backward_track
    :param forward_track:
    :param backward_track:
    :param iou_threshold:
    :return:
    """
    # find a temporal intersection
    fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop = \
        get_temporal_tracks(forward_track, backward_track)
    intersection = intersection_stop - intersection_start + 1
    if intersection_stop < intersection_start:
        logger.info(
            f'No temporal overlap between FW {forward_track.get_track_name()} and BW {backward_track.get_track_name()}')
        return 1
    # calculate agreement within temporal intersection
    agreement = 0
    for frame_id in range(intersection_start, intersection_stop + 1):
        fw_box = forward_track.boxes[frame_id - fw_start]
        bw_box = backward_track.boxes[bw_start - frame_id - 1]
        iou = bbox_iou(fw_box.get_xyxy(), bw_box.get_xyxy())
        if iou > iou_threshold:
            agreement += 1
    logger.info(f'Agreement {agreement} over intersection {intersection}: '
                f'{agreement / intersection}')
    return 1.0 - (agreement / intersection)


def merge_boxes(forward_track: TrackerWrapper, backward_track: TrackerWrapper) -> None:
    """
    This function merges boxes from backward_track to forward_track
    :param forward_track:
    :param backward_track:
    :return:
    """
    # re-init tracker
    # adding backward boxes to forward, using conf_score to judge
    fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop = \
        get_temporal_tracks(forward_track, backward_track)
    fw_votes = 0
    bw_votes = 0
    for frame_id in range(intersection_start, intersection_stop + 1):
        fw_box = forward_track.boxes[frame_id - fw_start]
        bw_box = backward_track.boxes[bw_start - frame_id - 1]
        if fw_box.conf_score >= bw_box.conf_score:
            fw_votes += 1
        else:
            bw_votes += 1
    logger.info(f'fw_votes {fw_votes} - bw_votes {bw_votes}')
    if fw_votes < bw_votes:
        logger.info(f'bw_votes win - '
                    f'Remove fw and replace by bw boxes from {intersection_start} to {intersection_stop}')
    else:
        logger.info(
            f'fw_votes win - Keep fw boxes from {intersection_start} to {intersection_stop}')
    logger.info(f'Append bw boxes from {fw_stop + 1} to {bw_stop} to fw')
    for frame_id in range(intersection_start, bw_stop + 1):
        if frame_id <= fw_stop:
            if fw_votes >= bw_votes:
                continue
            else:
                forward_track.boxes[frame_id - fw_start] = backward_track.boxes[
                    bw_start - frame_id - 1]
        else:
            forward_track.append_box_wrapper(backward_track.boxes[bw_start - frame_id - 1])


if __name__ == '__main__':
    try:
        # Parse config file
        args = parse_config()
        logger.info(f'Config {args}')
        # Create video writer and reader
        start = perf_counter()
        INPUT_VIDEO_PATH = os.path.join(args.input_video_dir, args.run + f'_trim.mp4')
        # INPUT_LABEL_PATH = os.path.join(args.input_label_dir, args.run + f'_labels_fixed.csv')
        INPUT_LABEL_PATH = os.path.join(args.input_label_dir, args.run + f'_labels.csv')
        if not os.path.exists(args.output_video_dir):
            os.makedirs(args.output_video_dir)
        if not os.path.exists(args.output_csv_dir):
            os.makedirs(args.output_csv_dir)
        OUTPUT_VIDEO_FW = os.path.join(args.output_video_dir, args.run + f'_{args.track_tag}_fw.avi')
        OUTPUT_VIDEO_BW = os.path.join(args.output_video_dir, args.run + f'_{args.track_tag}_bw.avi')
        OUTPUT_VIDEO_MERGED = os.path.join(args.output_video_dir,
                                           f"{args.run}_{args.track_tag}_merged.avi")
        OUTPUT_CSV_PATH = os.path.join(args.output_csv_dir, args.run + f'_r50.csv')

        cv2_video_reader = CV2VideoReader(INPUT_VIDEO_PATH)
        # cv2_video_writer_fw = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_FW,
        #                                      width=cv2_video_reader.width,
        #                                      height=cv2_video_reader.height, fps=120)
        # cv2_video_writer_bw = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_BW,
        #                                      width=cv2_video_reader.width,
        #                                      height=cv2_video_reader.height, fps=120)
        cv2_video_writer_merged = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_MERGED,
                                                 width=cv2_video_reader.width,
                                                 height=cv2_video_reader.height, fps=120)
        width = cv2_video_reader.width
        height = cv2_video_reader.height
        csv_headers = ['frame', 'name', 'x', 'y', 'w', 'h', 'confidence', 'ground_truth', 'width', 'height']
        with open(OUTPUT_CSV_PATH, 'w') as g:
            writer = csv.writer(g)
            writer.writerow(csv_headers)
        label_df = pd.read_csv(INPUT_LABEL_PATH)
        labeled_seconds = np.array(sorted(label_df['index'].unique()))
        labeled_frames = labeled_seconds * cv2_video_reader.fps
        labeled_frames = list(map(round, labeled_frames))
        logger.info(f'Label frames: {labeled_frames}')
        cv2_video_reader.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0 - 1
        buffer_frames = dict()
        track_kwargs = dict(model_config=args.model_config, model_path=args.model_path,
                            tracker_type='siam')
        color_reference = ColorRef(ColorRef.forward_set)
        context_forward = Context(track_kwargs=track_kwargs, color_reference=color_reference)
        while cv2_video_reader.capture.isOpened():
            frame_id += 1
            # if frame_id > 400:
            #     break
            ret, frame = cv2_video_reader.read_frame()
            if not ret:
                logger.info('End of video stream, ret is False!')
                break
            buffer_frames[frame_id] = frame
            if frame_id in labeled_frames:  # If this is a label frame
                # do forward tracking
                logger.info(
                    f'Forward tracking from {frame_id - len(buffer_frames) + 1} to {frame_id}')
                context_forward = track_buffer(context=context_forward,
                                               buffer_frames=deepcopy(buffer_frames),
                                               label_df=label_df,
                                               labeled_frames=labeled_frames,
                                               is_backward=False, fps=cv2_video_reader.fps)
                # draw_context_on_frames(context_forward, deepcopy(buffer_frames),
                #                        cv2_video_writer_fw, labeled_frames=labeled_frames)
                # do backward tracking
                logger.info(
                    f'Backward tracking from {frame_id} to {frame_id - len(buffer_frames) + 1}')
                # color_reference to assign color to boxes during bbox matching and tracking
                color_reference = ColorRef(ColorRef.backward_set)
                context_backward = Context(track_kwargs=track_kwargs,
                                           color_reference=color_reference)
                context_backward = track_buffer(context=context_backward,
                                                buffer_frames=deepcopy(buffer_frames),
                                                label_df=label_df,
                                                labeled_frames=labeled_frames,
                                                is_backward=True, fps=cv2_video_reader.fps)
                # draw_context_on_frames(context_backward, deepcopy(buffer_frames),
                #                        cv2_video_writer_bw, labeled_frames=labeled_frames)
                # merge backward and forward, some tracks of context_forward.tracks[...]
                # will be active after merging. context_merged is context_fw
                context_forward = matching_and_merging(context_forward, context_backward)
                logger.info(f"After merging")
                log_str = print_context(context_forward)
                draw_context_on_frames(context_forward, deepcopy(buffer_frames),
                                       cv2_video_writer_merged, labeled_frames=labeled_frames)
                # write csv tracking
                with open(OUTPUT_CSV_PATH, 'a') as g:
                    for object_type, category_track in context_forward.tracks.items():
                        for object_id, track_wrapper in category_track.items():
                            for box_wrapper in track_wrapper.boxes:
                                if box_wrapper.frame_id in buffer_frames:
                                    writer = csv.writer(g)
                                    writer.writerow(box_wrapper.get_csv_row() + [width, height])
                # reset buffer
                buffer_frames = dict()
                # release backward tracks
                logger.info('Release backward trackers')
                for object_type, category_track in context_backward.tracks.items():
                    for object_id, track_wrapper in category_track.items():
                        if track_wrapper.terminate:
                            track_wrapper.release_tracker()
                torch.cuda.empty_cache()

        # cv2_video_writer_bw.writer.release()
        # cv2_video_writer_fw.writer.release()
        cv2_video_writer_merged.writer.release()
        cv2_video_reader.capture.release()

        end = perf_counter()
        logger.info(f'Running time: {end - start}')

        with open('track_complete.txt', 'a') as f:
            index = log_str.find('Stats')
            if index != -1:
                f.write(args.run + '\n')
                f.write(log_str[index:] + '\n')

    except Exception as error:
        cv2_video_writer_merged.writer.release()
        cv2_video_reader.capture.release()
        error_str = repr(error)
        with open('track_error.txt', 'a') as f:
            f.write(args.run + '\n')
            f.write(error_str + '\n')
            f.write(traceback.format_exc() + '\n')
