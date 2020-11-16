import cv2
import numpy as np
import torch
# torch.backends.cudnn.enabled=False
import sys

sys.path.append('C:\\Users\\nguye\\Documents\\PBS\\Research\\pysot')
import os

sys.path.append(os.getcwd())
import csv
import pandas as pd
import argparse
import configparser
from typing import Tuple
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from utils import TrackerWrapper, BoxWrapper, FrameWrapper, CV2VideoWriter, CV2VideoReader, \
    Context, ColorBGR, Sample, logger, parse_config, ColorRef


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


def track_buffer(context: Context, args, buffer_frames: dict,
                 backward=False) -> Context:
    if backward == False:
        # Delete label frame while tracking forward
        del buffer_frames[max(buffer_frames.keys())]
    # Create video writer and reader
    label_df = pd.read_csv(args.input_label_path)
    label_times = np.array(sorted(label_df['index'].unique()))
    label_frames = label_times * cv2_video_reader.fps
    label_frames = list(label_frames.astype(np.int))

    for frame_id, frame in sorted(buffer_frames.items(), key=lambda kv: kv[0],
                                  reverse=backward):
        if frame_id % 150 == 0:
            logger.info(f'Processing frame {frame_id}')
        frame_wrapper = FrameWrapper(frame=frame, frame_id=frame_id)
        context.frame_results[frame_wrapper.frame_id] = dict()
        if frame_id in label_frames:  # If this is a label frame
            df_interested = label_df[
                (label_df['index'] == (frame_id // cv2_video_reader.fps)) & (
                        label_df['class'] == 'bagel')]
            boxes = df_interested[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            context.tracking('bagel', frame_wrapper)
            context.matching(boxes, 'bagel', frame_wrapper)
        else:  # If this is not a label frame
            context.tracking('bagel', frame_wrapper)

    return context


def merge_fw_bw_tracks(context_forward: Context, context_backward: Context,
                       last_frame: np.ndarray, agreement_threshold=0.8) -> Context:
    forward_categories = set(context_forward.tracks.keys())
    backward_categories = set(context_backward.tracks.keys())
    shared_categories = forward_categories.intersection(backward_categories)
    for object_type in shared_categories:
        n_fw_tracks = len(context_forward.tracks[object_type])
        n_bw_tracks = len(context_backward.tracks[object_type])
        distance_matrix = np.full(shape=(n_fw_tracks, n_bw_tracks), fill_value=10000,
                                  dtype=np.float)
        for fw_object_id, fw_track in context_forward.tracks[object_type].items():
            for bw_object_id, bw_track in context_backward.tracks[object_type].items():
                logger.info(
                    f'Compare fw {object_type + str(fw_object_id)} with bw {object_type + str(bw_object_id)}')
                distance_matrix[fw_object_id][bw_object_id] = compare_tracks(fw_track,
                                                                             bw_track)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matched_bw_tracks = []
        for row, col in zip(row_ind, col_ind):
            if 1 - distance_matrix[row][col] > agreement_threshold:
                logger.info(
                    f'Merge fw {object_type + str(row)} with bw {object_type + str(col)} '
                    f'Matching score: {1 - distance_matrix[row][col]}')
                context_forward.tracks[object_type][row].re_init(frame=frame, box_wrapper=context_backward.tracks[object_type][col].boxes[0])
                merge_tracks(context_forward.tracks[object_type][row],
                             context_backward.tracks[object_type][col])
                matched_bw_tracks.append(col)

        for bw_object_id, bw_track in context_backward.tracks[object_type].items():
            if bw_object_id not in col_ind or bw_object_id not in matched_bw_tracks:
                logger.info(
                    f'Merge new bw track {bw_track.object_name} to fw as {object_type + str(len(context_forward.tracks[object_type]))}')
                bw_track.change_name(
                    object_type + str(len(context_forward.tracks[object_type])))
                bw_track.sort_boxes()
                bw_track.re_init(bw_track.boxes[-1], frame=last_frame)
                context_forward.tracks[object_type][
                    len(context_forward.tracks[object_type])] = bw_track

    new_categories = backward_categories.difference(forward_categories)
    for object_type in new_categories:
        for bw_object_id, bw_track in context_backward.tracks[object_type].items():
            logger.info(f'Merge new bw {bw_track.object_name} into fw')
            logger.info(f'Init new category fw {object_type}')
            context_forward.tracks[object_type] = dict()
            bw_track.change_name(object_type + str(len(context_forward.tracks[object_type])))
            bw_track.sort_boxes()
            bw_track.re_init(bw_track.boxes[-1], frame=last_frame)
            context_forward.tracks[object_type][
                len(context_forward.tracks[object_type])] = bw_track
    return context_forward


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
    logger.info(f'intersection start {intersection_start} - intersection stop {intersection_stop}')
    return fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop


def compare_tracks(forward_track: TrackerWrapper, backward_track: TrackerWrapper, iou_threshold=0.2) -> float:
    """
    This functions measures agreement score between forward_track and backward_track
    :param forward_track:
    :param backward_track:
    :param iou_threshold:
    :return:
    """
    # find a temporal intersection
    fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop = get_temporal_tracks(
        forward_track,
        backward_track)
    intersection = intersection_stop - intersection_start + 1
    if intersection_stop < intersection_start:
        logger.info(f'No temporal overlap between fw {forward_track.object_name} and bw {backward_track.object_name}')
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
                f'{agreement/intersection}')
    return 1.0 - (agreement / intersection)


def merge_tracks(forward_track: TrackerWrapper, backward_track: TrackerWrapper) -> None:
    """
    This function merges boxes from backward_track to forward_track
    :param forward_track:
    :param backward_track:
    :return:
    """
    # re-init tracker
    # adding backward boxes to forward, using conf_score to judge
    fw_start, fw_stop, bw_start, bw_stop, intersection_start, intersection_stop = get_temporal_tracks(
        forward_track,
        backward_track)
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
        logger.info(f'fw_votes win - Keep fw boxes from {intersection_start} to {intersection_stop}')
    logger.info(f'Append bw boxes from {fw_stop + 1} to {bw_stop} to fw')
    for frame_id in range(intersection_start, bw_stop + 1):
        if frame_id <= fw_stop:
            if fw_votes >= bw_votes:
                # if False:
                continue
            else:
                forward_track.boxes[frame_id - fw_start] = backward_track.boxes[
                    bw_start - frame_id - 1]
        else:
            forward_track.boxes.append(backward_track.boxes[bw_start - frame_id - 1])


def draw_context_on_frames(context: Context, buffer_frames: dict,
                           cv2_video_writer: CV2VideoWriter) -> None:
    """
    This functions draw all tracks from a context on a list of frames
    :param context:
    :param buffer_frames:
    :param cv2_video_writer:
    :return:
    """
    color_reference = context.color_reference
    # going over all tracks
    for object_type, category_track in context.tracks.items():
        for object_id, track_wrapper in category_track.items():
            for box_wrapper in track_wrapper.boxes:
                if box_wrapper.frame_id in buffer_frames:
                    frame = buffer_frames[box_wrapper.frame_id]
                    frame_wrapper = FrameWrapper(frame, frame_id=box_wrapper.frame_id)
                    frame_wrapper.put_text(f'FrameID {box_wrapper.frame_id}')
                    frame_wrapper.put_text(f'Second {box_wrapper.frame_id // 30}')
                    frame_wrapper.put_bbox(box_wrapper,
                                           color=box_wrapper.color)
    for frame_id, frame in buffer_frames.items():
        cv2_video_writer.write_frame(frame)


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')
    # Create video writer and reader
    cv2_video_reader = CV2VideoReader(args.input_video_path)
    cv2_video_writer_fw = CV2VideoWriter(output_video_path=args.output_video_path_forward,
                                         width=cv2_video_reader.width,
                                         height=cv2_video_reader.height, fps=120)
    cv2_video_writer_bw = CV2VideoWriter(output_video_path=args.output_video_path_backward,
                                         width=cv2_video_reader.width,
                                         height=cv2_video_reader.height, fps=120)
    cv2_video_writer_merged = CV2VideoWriter(output_video_path=args.output_video_path_merged,
                                             width=cv2_video_reader.width,
                                             height=cv2_video_reader.height, fps=120)
    label_df = pd.read_csv(args.input_label_path)
    labeled_seconds = np.array(sorted(label_df['index'].unique()))
    labeled_frames = labeled_seconds * cv2_video_reader.fps
    labeled_frames = list(labeled_frames.astype(np.int))
    cv2_video_reader.capture.set(cv2.CAP_PROP_POS_FRAMES, 7010)
    frame_id = 7009
    buffer_frames = dict()
    track_kwargs = dict(model_config=args.model_config, model_path=args.model_path,
                        tracker_type='siam')
    color_reference = ColorRef(ColorRef.forward_set)
    context_forward = Context(track_kwargs=track_kwargs, color_reference=color_reference)
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        ret, frame = cv2_video_reader.read_frame()
        if not ret or frame_id > 7950:
            logger.info('End of video stream, ret is False!')
            break
        buffer_frames[frame_id] = frame
        last_frame = deepcopy(frame)
        if frame_id in labeled_frames:  # If this is a label frame
            # do forward tracking
            logger.info(f'Forward tracking from {frame_id - len(buffer_frames) + 1} to {frame_id}')
            context_forward = track_buffer(context_forward, args, deepcopy(buffer_frames),
                                           backward=False)
            draw_context_on_frames(context_forward, deepcopy(buffer_frames),
                                   cv2_video_writer_fw)
            # do backward tracking
            logger.info(f'Backward tracking from {frame_id - len(buffer_frames) + 1} to {frame_id}')
            color_reference = ColorRef(ColorRef.backward_set)
            context_backward = Context(track_kwargs=track_kwargs,
                                       color_reference=color_reference)
            context_backward = track_buffer(context_backward, args,
                                            deepcopy(buffer_frames),
                                            backward=True)
            draw_context_on_frames(context_backward, deepcopy(buffer_frames),
                                   cv2_video_writer_bw)
            # merge backward and forward, some tracks of context.tracks[...]
            # will be active after merging. context_merged is context_fw
            context_merged = merge_fw_bw_tracks(context_forward, context_backward,
                                                last_frame)
            context_merged.color_reference = ColorRef(ColorRef.merged_set)
            draw_context_on_frames(context_merged, deepcopy(buffer_frames),
                                   cv2_video_writer_merged)
            context_forward.color_reference = ColorRef(ColorRef.forward_set)
            # reset buffer
            buffer_frames = dict()
            context_forward = context_merged

    cv2_video_writer_bw.writer.release()
    cv2_video_writer_fw.writer.release()
    cv2_video_writer_merged.writer.release()
    cv2_video_reader.capture.release()
