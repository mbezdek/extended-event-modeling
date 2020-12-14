import numpy as np
import cv2
import os
import logging
import pandas as pd
import matplotlib.animation as animation
import math
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import argparse
import configparser
from scipy.optimize import linear_sum_assignment
import coloredlogs
from copy import deepcopy
from colorlog.colorlog import escape_codes  # necessary for ANSI escape

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(name)s - %(levelname)s - %(message)s'
# c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
c_handler.setFormatter(coloredlogs.ColoredFormatter(LOGFORMAT))
logger.addHandler(c_handler)


def parse_config():
    """
    This function receive a config file *.ini and parse those parameters
    :return: args
    """
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('-c', '--config_file')
    args, remaining_argv = arg_parser.parse_known_args()
    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    # Defaults arguments are taken from config file
    defaults = {}
    if args.config_file:
        config_parser = configparser.ConfigParser()
        config_parser.read(args.config_file)
        for section in config_parser.sections():
            defaults.update(dict(config_parser.items(section=section)))
    # Parse the rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[arg_parser]
    )
    parser.set_defaults(**defaults)
    # These arguments can be overridden by command line
    parser.add_argument("--run")
    parser.add_argument("--tag")
    args = parser.parse_args(remaining_argv)

    return args


def bin_times(array, max_seconds, bin_size=1.0):
    """ Helper function to learn the bin the subject data"""
    cumulative_binned = [np.sum(array <= t0 * 1000) for t0 in
                         np.arange(bin_size, max_seconds + bin_size, bin_size)]
    binned = np.array(cumulative_binned)[1:] - np.array(cumulative_binned)[:-1]
    binned = np.concatenate([[cumulative_binned[0]], binned])
    return binned


def load_comparison_data(data, bin_size=1.0):
    # Movie A is Saxaphone (185s long)
    # Movie B is making a bed (336s long)
    # Movie C is doing dishes (255s long)

    # here, we'll collapse over all of the groups (old, young; warned, unwarned) for now
    n_subjs = len(set(data.SubjNum))

    sax_times = np.sort(list(set(data.loc[data.Movie == 'A', 'MS']))).astype(np.float32)
    binned_sax = bin_times(sax_times, 185, bin_size) / np.float(n_subjs)

    bed_times = np.sort(list(set(data.loc[data.Movie == 'B', 'MS']))).astype(np.float32)
    binned_bed = bin_times(bed_times, 336, bin_size) / np.float(n_subjs)

    dishes_times = np.sort(list(set(data.loc[data.Movie == 'C', 'MS']))).astype(np.float32)
    binned_dishes = bin_times(dishes_times, 255, bin_size) / np.float(n_subjs)

    return binned_sax, binned_bed, binned_dishes


def get_frequency_ground_truth(second_boundaries, second_interval=1,
                               end_second=555) -> Tuple:
    frequency, bins = np.histogram(second_boundaries,
                                   bins=np.arange(0, end_second + second_interval,
                                                  second_interval))
    return frequency, bins


def get_binned_prediction(posterior, second_interval=1, sample_per_second=30) -> np.ndarray:
    e_hat = np.argmax(posterior, axis=1)
    frame_boundaries = np.concatenate([[0], e_hat[1:] != e_hat[:-1]])
    frame_interval = int(second_interval * sample_per_second)
    # Sum for each interval
    time_boundaries = np.add.reduceat(frame_boundaries,
                                      range(0, len(frame_boundaries), frame_interval))
    return np.array(time_boundaries, dtype=bool)


def get_point_biserial(boundaries_binned, binned_comp) -> float:
    M_1 = np.mean(binned_comp[boundaries_binned == 1])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned == 1)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))
    return r_pb


class Sample:
    """
    This class contains example python object to test syntax
    """
    sample_dict = {'1': 1, '10': 10, '2': 2}
    sample_list = [1, 5, 7]
    sample_list_of_lists = [[1, 2], [3, 4], [5, 6]]
    sample_tuple = (1, 3)
    sample_2d_array = np.array([[1, 2, 3], [4, 5, 6]])
    sample_list_of_array = [np.array([1, 2]), np.array([3, 4, 5])]


class SegmentationVideo:
    """
    This class receives a data_frame and a video_path to extract segmentation results
    for that video
    """

    def __init__(self, data_frame, video_path):
        self.data_frame = data_frame[
            data_frame['movie1'] == os.path.splitext(os.path.basename(video_path))[0]]
        self.n_participants = 0
        self.biserials = None
        self.seg_points = None
        self.gt_freqs = None
        self.gt_boundaries = None

    @staticmethod
    def string_to_segments(raw_string: str) -> np.ndarray:
        """
        This method turn a raw string to a list of timepoints
        :param raw_string: a string in csv file
        :return: list_of_segments: a list of timepoints
        """
        raw_string = raw_string.split('\n')
        list_of_segments = [float(x.split(' ')[1]) for x in raw_string if 'BreakPoint' in x]
        list_of_segments = np.array(list_of_segments)
        return list_of_segments

    def get_biserial_subjects(self, second_interval=1, end_second=555):
        self.biserials = []
        all_seg_points = np.hstack(self.seg_points)
        self.gt_boundaries, _ = get_frequency_ground_truth(all_seg_points,
                                                      second_interval=second_interval,
                                                      end_second=end_second)
        self.gt_freqs = self.gt_boundaries / self.n_participants
        for seg_point in self.seg_points:
            participant_seg, _ = get_frequency_ground_truth(seg_point,
                                                            second_interval=second_interval,
                                                            end_second=end_second)
            if sum(participant_seg) == 0:
                logger.info(f'Subject_segmentation={participant_seg} out of end_second={end_second}')
                continue
            point = get_point_biserial(participant_seg, self.gt_freqs)
            self.biserials.append(point)

    def preprocess_segments(self):
        average = np.mean([len(seg) for seg in self.seg_points if len(seg)])
        std = np.std([len(seg) for seg in self.seg_points if len(seg)])
        empty = 0
        out_lier = 0
        new_seg_points = []
        for seg in self.seg_points:
            if len(seg) > 0:
                if average + 2 * std > len(seg) > average - 2 * std:
                    new_seg_points.append(seg)
                else:
                    out_lier += 1
            else:
                empty += 1
        assert len(self.seg_points) == empty + out_lier + len(new_seg_points)
        logger.info(f'{empty} Empty participants and {out_lier} Outlier participants')
        self.seg_points = new_seg_points
        self.n_participants = len(self.seg_points)

    def get_segments(self, n_annotators=100, condition='coarse') -> List:
        """
        This method extract a list of segmentations, each according to an annotator
        :param n_annotators: number of annotators to return
        :param condition: coarse or fine grains
        :return:
        """
        seg = self.data_frame[self.data_frame['condition'] == condition]
        logger.info(f'Total of participants {len(seg)}')
        # parse annotations, from string to a list of breakpoints for each annotation
        seg_processed = seg['segment1'].apply(SegmentationVideo.string_to_segments)
        self.seg_points = seg_processed.values[:n_annotators]
        self.preprocess_segments()
        return self.seg_points


class CV2VideoReader:
    """
    This class is a wrapper of opencv video-capturing stream. It stores some commonly used
    variables and implements some commonly used method
    """

    def __init__(self, input_video_path):
        """
        Initialize some commonly used variables, can be extended
        :param input_video_path: path to input video to read
        """
        logger.debug('Creating an instance of CV2VideoReader')
        self.capture = cv2.VideoCapture(input_video_path)
        if self.capture.isOpened() is False:
            logger.error("Error opening video stream for reading")
        self.height, self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def __repr__(self) -> Dict:
        """
        For printing purpose, can be changed based on needs
        :return:
        """
        return {'reader': self.capture,
                'height': self.height,
                'width': self.width,
                'fps': self.fps}

    def __del__(self) -> None:
        """
        Release the video stream
        :return:
        """
        logger.debug('Destroying an instance of CV2VideoReader')
        self.capture.release()

    def read_frame(self) -> Tuple:
        """
        Get next frame
        :return: ret, frame
        """
        return self.capture.read()


class CV2VideoWriter:
    """
    This class is a wrapper of opencv video-writing stream. It stores some commonly used
    variables and implements some commonly used method
    """

    def __init__(self, output_video_path, fps=30, height=740, width=960):
        """

        :param output_video_path: path to output video
        :param fps: fps of the video
        :param height: height of output video
        :param width: width of output video
        """
        logger.debug('Creating an instance of CV2VideoWriter')
        if os.path.splitext(output_video_path)[1] == '.avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif os.path.splitext(output_video_path)[1] == '.mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        else:
            logger.error(f'Error opening video stream for writing \n'
                         f'Incorrect output format: {os.path.splitext(output_video_path)[1]}')
        self.writer = cv2.VideoWriter(output_video_path, fourcc=self.fourcc, fps=fps,
                                      frameSize=(width, height))
        self.fps = fps
        self.height = height
        self.width = width

    def __repr__(self) -> Dict:
        """
        For printing purpose
        :return:
        """
        return {'writer': self.writer,
                'fps': self.fps,
                'height': self.height,
                'width': self.width}

    def __del__(self) -> None:
        """
        Release writing stream
        :return:
        """
        logger.debug('Destroying an instance of CV2VideoWriter')
        self.writer.release()

    def write_frame(self, frame) -> None:
        """
        Write next frame
        :param frame:
        :return:
        """
        self.writer.write(frame)


class ColorBGR:
    """
    For explicit color
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    cyan = (255, 255, 0)


class ColorRef:
    forward_set = dict(init=ColorBGR.blue, track=ColorBGR.green, match=ColorBGR.blue,
                       deactivate=ColorBGR.red)
    backward_set = dict(init=ColorBGR.blue, track=ColorBGR.magenta, match=ColorBGR.blue,
                        deactivate=ColorBGR.red)
    merged_set = dict(init=ColorBGR.blue, track=ColorBGR.cyan, match=ColorBGR.blue,
                      deactivate=ColorBGR.red)

    def __init__(self, color_dict: Dict):
        self.color_dict = color_dict


class BoxWrapper:
    """
    This class keeps track of relevant variables for a bounding box and implements
    commonly used methods
    """

    def __init__(self, xmin, xmax, ymin, ymax, frame_id, state='init',
                 object_name='unknown', conf_score=-1.0, color=(255, 255, 255)):
        """
        Initialize bounding box's information
        :param xmin:
        :param xmax:
        :param ymin:
        :param ymax:
        :param frame_id:
        :param object_name:
        :param conf_score:
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.frame_id = frame_id
        self.conf_score = conf_score
        self.object_name = object_name
        self.state = state
        self.color = color

    def get_xywh(self) -> List:
        return [self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin]

    def get_xxyy(self) -> List:
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def get_xyxy(self) -> List:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def get_csv_row(self) -> List:
        """
        Get a csv row, can be changed to suit different formats
        :return:
        """
        return [self.frame_id, self.object_name, self.xmin, self.ymin,
                self.xmax - self.xmin,
                self.ymax - self.ymin, self.conf_score, 1]


class FrameWrapper:
    """
    This class keep tracks of relevant variables of a frame and implement commonly used methods
    """

    def __init__(self, frame: np.ndarray, frame_id=-1):
        """
        Initialize
        :param frame:
        :param frame_id:
        """
        self.frame = frame
        self.current_text_position = 0.0
        self.frame_id = frame_id
        self.boxes = dict()

    def put_text(self, text: str, color=ColorBGR.red, font_scale=0.7) -> None:
        """
        This method is designed to put annotations on the frame
        :param text:
        :param color:
        :param font_scale:
        :return:
        """
        # This variable keeps track of occupied positions on the frame
        self.current_text_position = self.current_text_position + 0.05
        cv2.putText(self.frame, text,
                    org=(50, int(self.current_text_position * self.get_height())),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=color)

    def put_bbox(self, bbox: BoxWrapper, color=ColorBGR.blue, font_scale=0.4) -> None:
        """
        This method is designed to draw a bounding box and relevant information on the frame,
        intensity of color reflects conf_score
        :param bbox:
        :param color:
        :param font_scale:
        :return:
        """
        # Be aware of type of color, cv2 is not good at throwing error messages
        color = tuple(map(int, np.array(color) * bbox.conf_score))
        cv2.rectangle(self.frame, pt1=(int(bbox.xmin), int(bbox.ymin)),
                      pt2=(int(bbox.xmax), int(bbox.ymax)),
                      color=color, thickness=1)
        cv2.putText(self.frame, text=bbox.object_name,
                    org=(int(bbox.xmin), int(bbox.ymax - 5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color)
        # cv2.putText(self.frame, text=str(f'{bbox.conf_score:.3f}'),
        #             org=(int(bbox.xmin), int(bbox.ymin + 10)),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=font_scale,
        #             color=color)

    def get_width(self) -> int:
        return self.frame.shape[1]

    def get_height(self) -> int:
        return self.frame.shape[0]


# This is a strategy function
class Canvas:
    """
    This is a template to use plt drawing functions and extract an image instead of visualizing
    """

    def __init__(self, rows: int = 3, columns: int = 1):
        """
        Initialize a figure and attached axes, then attach the figure to a canvas
        :param rows:
        :param columns:
        """
        # self.figure = Figure()
        self.figure, self.axes = plt.subplots(rows, columns)
        if rows * columns == 1:
            self.axes = [self.axes]
        self.canvas = FigureCanvasAgg(figure=self.figure)

    def get_current_canvas(self, width=960, height=200, left=0.145, right=0.88) -> np.ndarray:
        """
        This methods resize the current canvas and return an image
        :param width:
        :param height:
        :param left:
        :param right:
        :return:
        """
        self.canvas.draw()
        img = cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        canvas_resized = cv2.resize(
            img[:, int(img.shape[1] * left): int(img.shape[1] * right), :], (width, height))
        return canvas_resized

    def save_fig(self, img_name=''):
        self.figure.savefig(img_name)

    # This is a strategy function
    def draw_on_canvas(self, seg_points) -> None:
        """
        This method is designed to draw something on the canvas. We can define another methods
        to draw different things while using the structure of this class
        :param seg_points: a list of timepoints
        :return:
        """
        # Do some plotting.
        # sns.violinplot(data=seg_points, orient='h', ax=ax)
        sns.swarmplot(data=seg_points, orient='h', ax=self.axes[0], alpha=.1)
        sns.histplot(data=seg_points, bins=100, ax=self.axes[1])
        # sns.stripplot(data=seg_points, orient='h', ax=ax1, alpha=.1)
        self.axes[2].vlines(seg_points, ymin=0, ymax=1, alpha=0.05)


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


# Tracking utilities
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
                (label_df['index'] == (frame_id // fps))]
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
                    logger.info(f'Active fw track {fw_track.get_track_name()} does not match'
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
                logger.info(f'Active fw category {fw_track.get_track_name()} does not match'
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


# Skeleton utilities
def calc_joint_dist(df, joint):
    # df : skeleton tracking dataframe with 3D joint coordinates
    # joint : integer 0 to 24 corresponding to a Kinect skeleton joint
    # returns df with column of distance between the joint and spine mid:
    # some key joints are spine_mid:J1,left hand:J7,right hand: J11, foot left J15, foot right: J19
    j = str(joint)
    jx = 'J' + j + '_3D_X'
    jy = 'J' + j + '_3D_Y'
    jz = 'J' + j + '_3D_Z'
    jout = 'J' + j + '_dist_from_J1'
    df[jout] = np.sqrt(
        (df[jx] - df.J1_3D_X) ** 2 + (df[jy] - df.J1_3D_Y) ** 2 + (df[jz] - df.J1_3D_Z) ** 2)
    return df


def calc_joint_speed(df, joint):
    j = str(joint)
    jx = 'J' + j + '_3D_X'
    jy = 'J' + j + '_3D_Y'
    jz = 'J' + j + '_3D_Z'
    jout = 'J' + j + '_speed'
    df[jout] = np.sqrt((df[jx] - df[jx].shift(1)) ** 2 + (df[jy] - df[jy].shift(1)) ** 2 + (
            df[jz] - df[jz].shift(1)) ** 2) / (df.sync_time - df.sync_time.shift(1))
    return df


def calc_joint_acceleration(df, joint):
    j = str(joint)
    js = 'J' + j + '_speed'
    if js not in df.columns:
        df = calc_joint_speed(df, joint)
    jout = 'J' + j + '_acceleration'
    df[jout] = (df[js] - df[js].shift(1)) / (df.sync_time - df.sync_time.shift(1))
    return df


def calc_interhand_dist(df):
    df['interhand_dist'] = np.sqrt(
        (df.J11_3D_X - df.J7_3D_X) ** 2 + (df.J11_3D_Y - df.J7_3D_Y) ** 2 + (
                df.J11_3D_Z - df.J7_3D_Z) ** 2)
    return df


def calc_interhand_speed(df):
    # Values are positive when right hand (J11) is faster than left hand (J7)
    if 'J7_speed' not in df.columns:
        df = calc_joint_speed(df, 7)
    if 'J11_speed' not in df.columns:
        df = calc_joint_speed(df, 11)
    df['interhand_speed'] = df.J11_speed - df.J7_speed
    return df


def calc_interhand_acceleration(df):
    # Values are positive when right hand (J11) is faster than left hand (J7)
    if 'J7_acceleration' not in df.columns:
        df = calc_joint_acceleration(df, 7)
    if 'J11_acceleration' not in df.columns:
        df = calc_joint_acceleration(df, 11)
    df['interhand_acceleration'] = df.J11_acceleration - df.J7_acceleration
    return df


# Object-hand utilities
def resample_df(objhand_df, rate='40ms'):
    outdf = objhand_df.set_index(pd.to_datetime(objhand_df['sync_time'], unit='s'), drop=False,
                                 verify_integrity=True)
    resample_index = pd.date_range(start=outdf.index[0], end=outdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=outdf.columns)
    outdf = outdf.combine_first(dummy_frame).interpolate('time').resample(rate).first()
    return outdf


def calculateDistance(x1, y1, x2, y2):
    if (x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


def boxDistance(rx, ry, rw, rh, px, py):
    # find shortest distance between a point and a axis-aligned bounding box.
    # rx,ry: bottom left corner of rectangle
    # rw: rectangle width
    # rh: rectangle height
    # px,py: point coordinates
    dx = max([rx - px, 0, px - (rx + rw)])
    dy = max([ry - py, 0, py - (ry + rh)])
    return math.sqrt(dx * dx + dy * dy)


def calc_center(df):
    # Input: a dataframe with x,y,w, & h columns.
    # Output: dataframe with added columns for x and y center of each box.
    df['x_cent'] = df['x'] + (df['w'] / 2.0)
    df['y_cent'] = df['y'] + (df['h'] / 2.0)
    return df


def animate_video(resampledf, output_video_path):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=14, bitrate=1800)
    valcols = [i for i in resampledf.columns if i not in ['sync_time']]
    fdflong = pd.melt(resampledf, id_vars=['sync_time'], value_vars=valcols)
    fdflong = fdflong.sort_values('sync_time')

    times = fdflong['sync_time'].unique()
    NUM_COLORS = len(fdflong['variable'].unique())

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(0. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    fig, ax = plt.subplots(figsize=(14, 8))
    thresh = 299

    def draw_barchart(current_time):
        fdff = fdflong[fdflong['sync_time'].eq(current_time)].sort_values('variable')
        mask = fdff['value'] > thresh
        fdff.loc[mask, 'value'] = -1
        ax.clear()
        ax.barh(fdff['variable'], fdff['value'], color=colors)
        ax.text(0, 0.4, round(current_time, 2), transform=ax.transAxes, color='#777777',
                size=45, ha='right', weight=800)
        plt.xlim([-1, thresh])
        ax.set_axisbelow(True)
        ax.text(-1, 1.15, 'Object Movement',
                transform=ax.transAxes, size=23, weight=600, ha='left', va='top')
        # plt.box(False)

    draw_barchart(times[9])
    fig, ax = plt.subplots(figsize=(14, 8))
    animator = animation.FuncAnimation(fig, draw_barchart, frames=times)
    # HTML(animator.to_jshtml())
    animator.save(output_video_path, writer=writer)


def gen_feature_video(track_csv, skel_csv, output_csv, fps=30):
    # Read tracking result
    track_df = pd.read_csv(track_csv)
    track_df = calc_center(track_df)
    track_df['sync_time'] = track_df['frame'] / fps
    # Read skeleton result and set index by frame to merge tracking and skeleton
    skel_df = pd.read_csv(skel_csv)
    # sync_time, Right Hand: J11_2D_X, J11_2D_Y
    hand_df = skel_df.loc[:, ['sync_time', 'J11_2D_X', 'J11_2D_Y']]
    hand_df['frame'] = (hand_df.loc[:, 'sync_time'] * fps).apply(round).astype(np.int)
    hand_df.set_index('frame', drop=False, verify_integrity=True, inplace=True)
    # Process tracking result to create tracking dataframe
    final_frameid = max(max(hand_df['frame']), max(track_df['frame']))
    objs = track_df['name'].unique()
    objs_df = pd.DataFrame(index=range(final_frameid))
    objs_df.index.name = 'frame'
    for obj in objs:
        obj_df = track_df[track_df['name'] == obj].set_index('frame', drop=False,
                                                             verify_integrity=True)
        objs_df[obj + '_x_cent'] = obj_df['x_cent']
        objs_df[obj + '_y_cent'] = obj_df['y_cent']
        objs_df[obj + '_x'] = obj_df['x']
        objs_df[obj + '_y'] = obj_df['y']
        objs_df[obj + '_w'] = obj_df['w']
        objs_df[obj + '_h'] = obj_df['h']
        objs_df[obj + '_confidence'] = obj_df['confidence']
    logger.info('Combine hand dataframe and objects dataframe')
    # objhand_df = pd.concat([hand_df, objs_df], axis=1)
    objhand_df = hand_df.combine_first(objs_df)
    objhand_df = objhand_df.sort_index()
    # Process null entry by interpolation
    logger.info('Interpolate')
    for obj in objs:
        objhand_df[obj + '_x_cent'] = objhand_df[obj + '_x_cent'].interpolate(method='linear')
        objhand_df[obj + '_y_cent'] = objhand_df[obj + '_y_cent'].interpolate(method='linear')
        objhand_df[obj + '_x'] = objhand_df[obj + '_x'].interpolate(method='linear')
        objhand_df[obj + '_y'] = objhand_df[obj + '_y'].interpolate(method='linear')
        objhand_df[obj + '_w'] = objhand_df[obj + '_w'].interpolate(method='linear')
        objhand_df[obj + '_h'] = objhand_df[obj + '_h'].interpolate(method='linear')
    objhand_df['J11_2D_X'] = objhand_df['J11_2D_X'].interpolate(method='linear')
    objhand_df['J11_2D_Y'] = objhand_df['J11_2D_Y'].interpolate(method='linear')
    objhand_df['J11_2D_X'] = objhand_df['J11_2D_X'] / 2
    objhand_df['J11_2D_Y'] = objhand_df['J11_2D_Y'] / 2
    # Smooth movements
    logger.info('Gaussian filtering')
    for obj in objs:
        objhand_df[obj + '_x_cent'] = gaussian_filter1d(objhand_df[obj + '_x_cent'], 3)
        objhand_df[obj + '_y_cent'] = gaussian_filter1d(objhand_df[obj + '_y_cent'], 3)
        objhand_df[obj + '_x'] = gaussian_filter1d(objhand_df[obj + '_x'], 3)
        objhand_df[obj + '_y'] = gaussian_filter1d(objhand_df[obj + '_y'], 3)
        objhand_df[obj + '_w'] = gaussian_filter1d(objhand_df[obj + '_w'], 3)
        objhand_df[obj + '_h'] = gaussian_filter1d(objhand_df[obj + '_h'], 3)
    objhand_df['J11_2D_X'] = gaussian_filter1d(objhand_df['J11_2D_X'], 3)
    objhand_df['J11_2D_Y'] = gaussian_filter1d(objhand_df['J11_2D_Y'], 3)
    # Resample dataframe
    objhand_df.loc[:, 'sync_time'] = objhand_df.index / fps
    objhand_df.loc[:, 'frame'] = objhand_df.index
    resampledf = resample_df(objhand_df, rate='333ms')
    # Calculate distances between all objects and hand
    logger.info('Calculate object-hand distances')
    for obj in objs:
        resampledf[obj + '_dist'] = resampledf[
            [obj + '_x', obj + '_y', obj + '_w', obj + '_h', 'J11_2D_X',
             'J11_2D_Y']].apply(
            lambda x: boxDistance(x[0], x[1], x[2], x[3], x[4], x[5]) if (
                np.all(pd.notnull(x))) else np.nan, axis=1)
    resampledf.to_csv(output_csv, index=False)
    # animate_video(resampledf, output_video_path='output/objhand/1.2.5_C1_objhand.mp4')
