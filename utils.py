import numpy as np
import cv2
import os
import logging
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.tracker_builder import build_tracker

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)


class Sample:
    sample_dict = {'1': 1, '10': 10, '2': 2}
    sample_list = [1, 5, 7]
    sample_list_of_lists = [[1, 2], [3, 4], [5, 6]]
    sample_tuple = (1, 3)
    sample_2d_array = np.array([[1, 2, 3], [4, 5, 6]])
    sample_list_of_array = [np.array([1, 2]), np.array([3, 4, 5])]


class SegmentationVideo:
    def __init__(self, data_frame, video_path):
        self.data_frame = data_frame[
            data_frame['movie1'] == os.path.splitext(os.path.basename(video_path))[0]]

    @staticmethod
    def string_to_segments(raw_string: str) -> np.ndarray:
        raw_string = raw_string.split('\n')
        list_of_segments = [float(x.split(' ')[1]) for x in raw_string if 'BreakPoint' in x]
        list_of_segments = np.array(list_of_segments)
        return list_of_segments

    def get_segments(self, n_annotators=1, condition='coarse') -> List:
        seg = self.data_frame[self.data_frame['condition'] == condition]
        # parse annotations, from string to a list of breakpoints for each annotation
        seg_processed = seg['segment1'].apply(SegmentationVideo.string_to_segments)
        seg_points = seg_processed.values[:n_annotators]
        return seg_points


class CV2VideoReader:
    def __init__(self, input_video_path):
        logger.debug('Creating an instance of CV2VideoReader')
        self.capture = cv2.VideoCapture(input_video_path)
        if self.capture.isOpened() is False:
            logger.error("Error opening video stream for reading")
        self.height, self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def __repr__(self) -> Dict:
        return {'reader': self.capture,
                'height': self.height,
                'width': self.width,
                'fps': self.fps}

    def __del__(self):
        logger.debug('Destroying an instance of CV2VideoReader')
        self.capture.release()

    def read_frame(self):
        return self.capture.read()


class CV2VideoWriter:
    def __init__(self, output_video_path, fps=30, height=740, width=960):
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
        return {'writer': self.writer,
                'fps': self.fps,
                'height': self.height,
                'width': self.width}

    def __del__(self):
        logger.debug('Destroying an instance of CV2VideoWriter')
        self.writer.release()

    def write_frame(self, frame):
        self.writer.write(frame)


class ColorBGR:
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)


class BoxWrapper:
    """
    Objects of this class keep track of bounding box, frame_id, conf_score
    """

    def __init__(self, xmin, xmax, ymin, ymax, frame_id, category='unknown', conf_score=-1):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.frame_id = frame_id
        self.conf_score = conf_score
        self.category = category

    def get_xywh(self):
        pass

    def get_xxyy(self):
        return self.xmin, self.xmax, self.ymin, self.ymax


class FrameWrapper:
    def __init__(self, frame: np.ndarray, frame_id=-1):
        self.frame = frame
        self.current_text_position = 0.0
        self.frame_id = frame_id

    def put_text(self, text: str, color=ColorBGR.red, font_scale=1.0):
        self.current_text_position = self.current_text_position + 0.1
        cv2.putText(self.frame, text,
                    org=(50, int(self.current_text_position * self.get_height())),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=color)

    def put_bbox(self, bbox: BoxWrapper, color=ColorBGR.blue, font_scale=0.3):
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
                    org=(int(bbox.xmin), int(bbox.ymin+20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color)

    def get_width(self):
        return self.frame.shape[1]

    def get_height(self):
        return self.frame.shape[0]


# This is a strategy function
class Canvas:
    def __init__(self, rows: int = 3, columns: int = 1):
        # self.figure = Figure()
        self.figure, self.axes = plt.subplots(rows, columns)
        if rows * columns == 1:
            self.axes = [self.axes]
        self.canvas = FigureCanvasAgg(figure=self.figure)

    def get_current_canvas(self, width=960, height=200, left=0.145, right=0.88):
        self.canvas.draw()
        img = cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        canvas_resized = cv2.resize(
            img[:, int(img.shape[1] * left): int(img.shape[1] * right), :], (width, height))
        return canvas_resized

    # This is a strategy function
    def draw_on_canvas(self, seg_points) -> None:
        # Do some plotting.
        # sns.violinplot(data=seg_points, orient='h', ax=ax)
        sns.swarmplot(data=seg_points, orient='h', ax=self.axes[0], alpha=.1)
        sns.histplot(data=seg_points, bins=100, ax=self.axes[1])
        # sns.stripplot(data=seg_points, orient='h', ax=ax1, alpha=.1)
        self.axes[2].vlines(seg_points, ymin=0, ymax=1, alpha=0.05)


class TrackerWrapper:
    """
    Objects of this class keep track of tracker_algo, current and previous frame,
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
            siam_tracker.init(kwargs['frame'], kwargs['init_bbox'])

            self.tracker = siam_tracker
            self.current_frame = kwargs['frame']
            self.previous_frame = None
            self.object_name = kwargs['object_name']
        else:
            logger.error(f'Unknown tracker type: {tracker_type}')
        pass

    def get_next_box(self, frame):
        self.previous_frame = self.current_frame
        self.current_frame = frame
        outputs = self.tracker.track(frame)
        return outputs

    def predict(self, frame):
        pass


class Context:
    """
    Objects of this class keep track of active tracks and inactive tracks
    """

    def __init__(self):
        self.tracks = dict()

    pass
