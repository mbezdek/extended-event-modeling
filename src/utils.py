import numpy as np
import cv2
import os
import logging

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse
import configparser
import coloredlogs
from scipy.ndimage import gaussian_filter1d

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(funcName)s:%(lineno)d - %(name)s - %(levelname)s: %(message)s'
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
# c_handler.setFormatter(coloredlogs.ColoredFormatter(LOGFORMAT))
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
    parser.add_argument("--sem_tag")
    parser.add_argument("--track_tag")
    parser.add_argument("--feature_tag")
    parser.add_argument("--train")
    parser.add_argument("--valid")
    parser.add_argument("--alfa")
    parser.add_argument("--lmda")
    parser.add_argument("--lr")
    parser.add_argument("--trigger")
    parser.add_argument("--threshold")
    parser.add_argument("--equal_sigma")
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


def get_binned_prediction(boundaries, second_interval=1, sample_per_second=3) -> np.ndarray:
    frame_boundaries = boundaries.astype(bool).astype(int)
    frame_interval = round(second_interval * sample_per_second)
    # Sum for each interval
    time_boundaries = np.add.reduceat(frame_boundaries,
                                      range(0, len(frame_boundaries), frame_interval))
    return np.array(time_boundaries, dtype=bool)


def get_point_biserial(boundaries_binned, binned_comp, scale=True) -> float:
    M_1 = np.mean(binned_comp[boundaries_binned != 0])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned != 0)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))
    if scale:
        num_boundaries = boundaries_binned.astype(bool).sum()
        fake_upper = np.zeros(np.shape(binned_comp), dtype=bool)
        fake_upper[np.argsort(binned_comp)[-num_boundaries:]] = True
        M_1 = np.mean(binned_comp[fake_upper != 0])
        M_0 = np.mean(binned_comp[fake_upper == 0])
        r_upper = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))

        fake_lower = np.zeros(np.shape(binned_comp), dtype=bool)
        fake_lower[np.argsort(binned_comp)[:num_boundaries]] = True
        M_1 = np.mean(binned_comp[fake_lower != 0])
        M_0 = np.mean(binned_comp[fake_lower == 0])
        r_lower = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))
        return (r_pb - r_lower) / (r_upper - r_lower)
    else:
        return r_pb


class ReadoutDataframes:
    pass


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
        video_path = video_path.replace('kinect', 'C1').replace('C2', 'C1')
        if 'trim' not in video_path:
            video_path = video_path + '_trim'
        if 'C1' not in video_path:
            video_path = video_path + '_C1'
        self.data_frame = data_frame[
            data_frame['Movie'] == os.path.splitext(os.path.basename(video_path))[0]]
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

    def get_gt_freqs(self, second_interval=1, end_second=555):
        all_seg_points = np.hstack(self.seg_points)
        self.gt_boundaries, _ = get_frequency_ground_truth(all_seg_points,
                                                           second_interval=second_interval,
                                                           end_second=end_second)
        self.gt_freqs = self.gt_boundaries / self.n_participants
        self.gt_freqs = gaussian_filter1d(self.gt_freqs, 2)

        return self.gt_freqs

    def get_biserial_subjects(self, second_interval=1, end_second=555, number_boundary_range=(1, 555)):
        self.biserials = []
        if self.gt_freqs is None:
            self.get_gt_freqs(second_interval=second_interval, end_second=end_second)

        for seg_point in self.seg_points:
            participant_seg, _ = get_frequency_ground_truth(seg_point,
                                                            second_interval=second_interval,
                                                            end_second=end_second)
            # if sum(participant_seg) == 0:
            #     logger.info(f'Subject has no segments within end_second={end_second}')
            #     continue
            if sum(participant_seg) < number_boundary_range[0] or sum(participant_seg) > number_boundary_range[1]:
                logger.info(f'Subject has {sum(participant_seg)} boundaries, outside of {number_boundary_range}')
                continue

            point = get_point_biserial(participant_seg.astype(bool), self.gt_freqs)
            if not np.isnan(point):  # some participants yield null bicorr
                self.biserials.append(point)
        return self.biserials

    def preprocess_segments(self, second_interval=1):
        # Preprocess individual segmentation to avoid undue influence
        new_seg_points = []
        for seg in self.seg_points:
            if len(seg) > 0:
                new_seg = np.array([])
                # For an interval, each participant have at most one vote
                for i in range(0, round(min(max(seg), 1500)) + 1, second_interval):
                    # seg is a list of timepoints, averaging all timepoints within a second
                    if seg[(seg > i) & (seg < i + 1)].shape[0]:
                        new_seg = np.hstack([new_seg, seg[(seg > i) & (seg < i + 1)].mean()])
            else:
                new_seg = seg
            new_seg_points.append(new_seg)
        self.seg_points = new_seg_points

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
        # logger.info(f'{empty} Empty participants and {out_lier} Outlier participants')
        self.seg_points = new_seg_points
        self.n_participants = len(self.seg_points)

    def get_human_segments(self, n_annotators=100, condition='coarse', second_interval=1) -> List:
        """
        This method extract a list of segmentations, each according to an annotator
        :param n_annotators: number of annotators to return
        :param condition: coarse or fine grains
        :param second_interval: interval to bin segmentations
        :return:
        """
        seg = self.data_frame[self.data_frame['Condition'] == condition]
        # parse annotations, from string to a list of breakpoints for each annotation
        annotators = list(set(seg.workerId))
        # logger.info(f'Total of participants {len(annotators)}')
        seg_points = []
        for a in annotators:
            tmp = seg[seg.workerId == a].Sec.to_numpy()
            seg_points.append(tmp)
        # seg_processed = seg['segment1'].apply(SegmentationVideo.string_to_segments)
        self.seg_points = seg_points[:n_annotators]
        self.preprocess_segments(second_interval=second_interval)
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
        logger.info(f'CV2 Reader fps {input_video_path}={self.fps}')
        logger.info(f'CV2 Reader # frames {input_video_path}={self.total_frames}')

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

    def __init__(self, output_video_path, fps=30, height=740, width=960, is_color=True):
        """

        :param output_video_path: path to output video
        :param fps: fps of the video
        :param height: height of output video
        :param width: width of output video
        :param is_color: if writing a color or grayscale video
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
                                      frameSize=(width, height), isColor=int(is_color))
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
    mild_cyan = (100, 100, 0)


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


class Sampler:
    def __init__(self, df_select, validation_runs):
        self.df_select = df_select
        self.validation_runs = validation_runs
        self.chapter_to_list = dict()
        self.max_epoch = 0
        self.counter = 0
        self.train_list = []

    def _get_one_run(self, chapter):
        return self.chapter_to_list[chapter].pop(-1)

    def get_one_run(self):
        run = self.train_list[self.counter % len(self.train_list)]
        self.counter += 1
        return run

    def prepare_list(self, min_boundary=5, max_boundary=25, metric='percentile'):
        chapters = self.df_select.groupby('chapter').mean().sort_values(metric, ascending=False).index.to_numpy()
        for c in chapters:
            df_chapter = self.df_select[self.df_select['chapter'] == c]
            df_chapter = df_chapter[
                (df_chapter['number_boundaries'] >= min_boundary) & (df_chapter['number_boundaries'] <= max_boundary)]
            df_chapter = df_chapter[~df_chapter['run'].isin(self.validation_runs)]
            ascend_percentile = list(df_chapter.sort_values('percentile')['run'])
            self.chapter_to_list[c] = ascend_percentile
        self.max_epoch = min(map(len, self.chapter_to_list.values()))
        for i in range(self.max_epoch):
            for c in [4, 2, 3, 1]:
                self.train_list.append(self._get_one_run(c))
        # added on july_15 to test noisy video hypothesis
        # self.train_list = self.train_list[::-1]
        # print(f'Maximum number of epoch is {self.max_epoch}')
        logger.info(f"Sampler's Training List: {self.train_list} with {len(self.train_list)} runs")


def contain_substr(column: str, keeps):
    for k in keeps:
        if k in column:
            return 1
    return 0


def get_overlap(reference: Tuple, hypothesis: Tuple, length=None):
    t2_overlap = min(reference[1], hypothesis[1])
    t1_overlap = max(reference[0], hypothesis[0])
    overlap = max(t2_overlap - t1_overlap, 0)
    if length is None:
        length = (reference[1] - reference[0])
    return overlap / length


def get_coverage(annotated_event: pd.Series, event_to_intervals: Dict):
    annotated_time = (annotated_event['startsec'], annotated_event['endsec'])
    max_coverage = -1
    max_coverage_event = -1
    for sem_event, sem_intervals in event_to_intervals.items():
        c = max([get_overlap(annotated_time, interval) for interval in sem_intervals])
        if c >= max_coverage:
            max_coverage = c
            max_coverage_event = sem_event
    # print(f"Max coverage for {annotated_event['evname']} is SEM's event {max_coverage_event} with {max_coverage}")
    return annotated_event['evname'], max_coverage_event, max_coverage


def get_purity(sem_event: int, sem_intervals: List, run_df: pd.DataFrame):
    max_purity = -1
    max_purity_ann_event = ''
    total_length = sum([interval[1] - interval[0] for interval in sem_intervals])
    for i, annotated_event in run_df.iterrows():
        annotated_time = (annotated_event['startsec'], annotated_event['endsec'])
        c = max([get_overlap(interval, annotated_time, length=total_length) for interval in sem_intervals])
        if c >= max_purity:
            max_purity = c
            max_purity_ann_event = annotated_event['evname']
    # print(f"Max purity for SEM's event {sem_event} is annotated event {max_purity_ann_event} with {max_purity}")
    return sem_event, max_purity_ann_event, max_purity


def event_label_to_interval(event_label: np.ndarray, start_second):
    events = set(event_label)
    event_to_intervals = {e: [] for e in events}
    for e in event_to_intervals.keys():
        time = np.where(event_label == e)[0]
        prev = time[0]
        start = time[0]
        # add 1 frame to length in cases the event only exists in one frame, making length=0
        for cur in time[1:]:
            if cur > prev + 1:
                if start == prev:
                    prev += 1
                event_to_intervals[e].append((start / 3 + start_second, prev / 3 + start_second))
                start = cur
                prev = cur
            else:
                prev = prev + 1
        if start == prev:
            prev += 1
        event_to_intervals[e].append((start / 3 + start_second, prev / 3 + start_second))
    return event_to_intervals


class DictObj:
    def __init__(self, in_dict: dict):
        """
        This is a helper class to translate a nested dictionary to an object, easy reference/access
        :param in_dict:
        """
        self.appear_post = None
        self.optical_post = None
        self.skel_post = None
        self.objhand_post = None
        self.scene_post = None
        self.categories_z = None
        self.combined_resampled_df = None
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


def merge_feature_lists(txt_out="intersect_features.txt"):
    with open('appear_complete.txt', 'r') as f:
        appears = f.readlines()

    with open('vid_complete.txt', 'r') as f:
        vids = f.readlines()

    with open('skel_complete.txt', 'r') as f:
        skels = f.readlines()

    with open('objhand_complete.txt', 'r') as f:
        objhands = f.readlines()

    sem_runs = set(appears).intersection(set(skels)).intersection(set(vids)).intersection(
        set(objhands))
    with open(txt_out, 'w') as f:
        f.writelines(sem_runs)


def remove_flurries(boundaries, k=3) -> np.ndarray:
    # a function to group boundaries into a single boundary if they are within k steps of each other
    # boundaries: a list of 0s and 1s
    # k: the number of updates to group together
    # returns: a list of 0s and 1s
    boundaries = list(boundaries)
    boundaries_no_flurry = [0] * len(boundaries)
    flag = 0
    last = -1
    for i, b in enumerate(boundaries):
        if b == 1 and i > last:
            def find_last_boundary(boundaries, first, k):
                last = first
                offset = 0
                for j in range(first, len(boundaries)):
                    if boundaries[j] == 1:
                        last = j
                        offset = 0
                    else:
                        offset += 1
                    if offset >= k:
                        break
                return last

            first = i
            last = find_last_boundary(boundaries, first, k)
            # print(last, first, len(boundaries))
            boundaries_no_flurry[(last + first) // 2] = 1

    # logger.info(f"Number of boundaries before-after removing flurries: {sum(boundaries)}-{sum(boundaries_no_flurry)}")
    # print(f"boundaries before-after: {boundaries}-{boundaries_no_flurry}")
    return np.array(boundaries_no_flurry)


def remove_random_boundaries(boundaries, n) -> np.ndarray:
    # a function to remove n random boundaries
    # boundaries: a list of 0s and 1s
    # n: the number of boundaries to remove
    # returns: a list of 0s and 1s
    boundaries = list(boundaries)
    # boundaries indices
    b_indices = [i for i, b in enumerate(boundaries) if b == 1]
    # remove n random boundaries
    b_indices = np.random.choice(b_indices, size=max(0, len(b_indices) - n), replace=False)
    boundaries_removed_n = [0] * len(boundaries)
    for i in b_indices:
        boundaries_removed_n[i] = 1
    logger.info(f"Number of boundaries before-after removing random boundaries: {sum(boundaries)}-{sum(boundaries_removed_n)}")
    return np.array(boundaries_removed_n)


def adjust_n_boundaries(boundaries, k=0):
    """
    This function count the number of distinct boundary chunks in an array of 0s and 1s. A boundary chunk is defined as an interval whcih starts with a boundary and ends with a boundary and the distances between consecutive boundaries in this interval is smaller than or equal to k.
    e.g.: if k=1 and the input list is [0, 1, 1, 1, 0, 1, 1, 1, 1, 0], the function would return 2.
    """
    flag = 0
    count = 0
    for b in boundaries:
        if b == 1:
            flag = 1  # flag the first boundary
            offset = -1
        if flag:
            offset += 1
            if offset >= k:
                count += 1  # if there has been k non-boundary timesteps, count this as a boundary chunk
                offset = -1
                flag = 0
    return count

class PermutationBiserial:
    """
    This class is used to calculate and cache permuted biserial correlations for all combinations of run and #boundaries
    """
    def __init__(self, n_permutations=1000):
        self.n_permutations = n_permutations
        self.run_nb_to_biserial = {}  # {('1.2.3_kinect', 15): [0.001, -0.001, ...], ...}

    def get_null_bicorrs(self, run, nb, gt_freqs, scale=True) -> List:
        """
        This function returns a list of permuted biserial correlations for a given combo of run and #boundaries
        :param run: run name, e.g. '1.2.3_kinect'
        :param nb: number of boundaries, e.g. 15
        :param gt_freqs: human normative boundary frequencies, should be smoothed already
        :return: a list of permuted biserial correlations
        """
        if (run, nb) not in self.run_nb_to_biserial:
            self.run_nb_to_biserial[(run, nb)] = []
            # randomly select n_b boundaries, then calculate all random point biserial correlations
            for _ in range(self.n_permutations):
                boundary_array = np.zeros(len(gt_freqs))
                # randomly select n_b boundaries
                boundary_indices = np.random.choice(np.arange(len(gt_freqs)), nb, replace=False)
                boundary_array[boundary_indices] = 1
                self.run_nb_to_biserial[(run, nb)].append(get_point_biserial(boundary_array, gt_freqs, scale=scale))
            return self.run_nb_to_biserial[(run, nb)]
        else:
            return self.run_nb_to_biserial[(run, nb)]

