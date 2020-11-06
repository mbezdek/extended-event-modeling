import numpy as np
import cv2
import os
import logging
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import argparse
import configparser

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
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
    # parser.add_argument("--input_video_path")
    args = parser.parse_args(remaining_argv)

    return args


def get_frequency_ground_truth(time_boundaries, time_interval=1) -> np.ndarray:
    frequency, _ = np.histogram(time_boundaries,
                                bins=np.arange(0, max(time_boundaries) + time_interval,
                                               time_interval))
    return frequency


def get_binned_prediction(posterior, time_interval=1, fps=30) -> np.ndarray:
    e_hat = np.argmax(posterior, axis=1)
    frame_boundaries = np.concatenate([[0], e_hat[1:] != e_hat[:-1]])
    frame_interval = int(time_interval * fps)
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

    def get_segments(self, n_annotators=1, condition='coarse') -> List:
        """
        This method extract a list of segmentations, each according to an annotator
        :param n_annotators: number of annotators to return
        :param condition: coarse or fine grains
        :return:
        """
        seg = self.data_frame[self.data_frame['condition'] == condition]
        # parse annotations, from string to a list of breakpoints for each annotation
        seg_processed = seg['segment1'].apply(SegmentationVideo.string_to_segments)
        seg_points = seg_processed.values[:n_annotators]
        return seg_points


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


class BoxWrapper:
    """
    This class keeps track of relevant variables for a bounding box and implements
    commonly used methods
    """

    def __init__(self, xmin, xmax, ymin, ymax, frame_id, category='unknown', conf_score=-1):
        """
        Initialize bounding box's information
        :param xmin:
        :param xmax:
        :param ymin:
        :param ymax:
        :param frame_id:
        :param category:
        :param conf_score:
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.frame_id = frame_id
        self.conf_score = conf_score
        self.category = category

    def get_xywh(self) -> List:
        return [self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin]

    def get_xxyy(self) -> List:
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def get_csv_row(self) -> List:
        """
        Get a csv row, can be changed to suit different formats
        :return:
        """
        return [self.frame_id, self.category, self.xmin, self.ymin,
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

    def put_text(self, text: str, color=ColorBGR.red, font_scale=1.0) -> None:
        """
        This method is designed to put annotations on the frame
        :param text:
        :param color:
        :param font_scale:
        :return:
        """
        # This variable keeps track of occupied positions on the frame
        self.current_text_position = self.current_text_position + 0.1
        cv2.putText(self.frame, text,
                    org=(50, int(self.current_text_position * self.get_height())),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=color)

    def put_bbox(self, bbox: BoxWrapper, color=ColorBGR.blue, font_scale=0.3) -> None:
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
        cv2.putText(self.frame, text=bbox.category, org=(int(bbox.xmin), int(bbox.ymin - 5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color)
        cv2.putText(self.frame, text=str(f'{bbox.conf_score:.3f}'),
                    org=(int(bbox.xmin), int(bbox.ymin + 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color)

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

    def __init__(self, **kwargs):
        tracker_type = kwargs['tracker_type']
        if tracker_type == 'siam':
            # init siamrpn tracker
            logger.debug(f'Building siamrpn')
            cfg.merge_from_file(kwargs['model_config'])
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            device = torch.device('cuda' if cfg.CUDA else 'cpu')
            model = ModelBuilder()
            model.load_state_dict(
                torch.load(kwargs['model_path'],
                           map_location=lambda storage, loc: storage.cpu()))
            model.eval().to(device)
            siam_tracker = build_tracker(model)
            siam_tracker.init(kwargs['frame'], kwargs['box_wrapper'].get_xywh())
            self.tracker = siam_tracker
        else:
            self.tracker = None
            logger.error(f'Unknown tracker type: {tracker_type}')
        self.current_frame = kwargs['frame']
        self.previous_frame = None
        self.object_name = kwargs['box_wrapper'].category
        self.boxes = [kwargs['box_wrapper']]

    def get_next_box(self, frame) -> Dict:
        """
        This method receive a frame and return coordinates of its object on that frame
        :param frame:
        :return: outputs: object's coordinates and confidence score
        """
        self.previous_frame = self.current_frame
        self.current_frame = frame
        outputs = self.tracker.track(frame)
        return outputs

    def update(self, box_wrapper: BoxWrapper) -> None:
        """
        Update boxes
        :param box_wrapper:
        :return:
        """
        self.boxes.append(box_wrapper)


class Context:
    """
    This class keep track of active tracks and inactive tracks
    """

    def __init__(self):
        self.tracks = dict()

    pass
