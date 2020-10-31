import cv2
import numpy as np
import torch
# torch.backends.cudnn.enabled=False
import math

import sys
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.tracker_builder import build_tracker
import csv
import pandas as pd
import argparse
import configparser
from utils import TrackerWrapper, BoxWrapper, FrameWrapper, CV2VideoWriter, CV2VideoReader, \
    Context, ColorBGR, Sample, logger

if __name__ == '__main__':
    # Parse config file
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('-c', '--config_file')
    args, remaining_argv = arg_parser.parse_known_args()
    defaults = {}
    if args.config_file:
        config_parser = configparser.ConfigParser()
        config_parser.read(args.config_file)
        for section in config_parser.sections():
            defaults.update(dict(config_parser.items(section=section)))
    parser = argparse.ArgumentParser(parents=[arg_parser])
    parser.set_defaults(**defaults)
    # Can insert arguments that we want to override by command line
    # parser.add_argument("--input_video_path")
    args = parser.parse_args(remaining_argv)

    # Create video writer and reader
    cv2_video_reader = CV2VideoReader(input_video_path=args.input_video_path)
    cv2_video_writer = CV2VideoWriter(output_video_path=args.output_video_path,
                                      width=cv2_video_reader.width,
                                      height=cv2_video_reader.height)
    # Parse label file
    label_df = pd.read_csv(args.input_label_path)
    label_times = np.array(sorted(label_df['index'].unique()))
    label_frames = label_times * cv2_video_reader.fps
    label_frames = list(label_frames.astype(np.int))
    logger.info(f'label_frames {label_frames} \n'
                f'fps {cv2_video_reader.fps}')
    # Tracker information
    track_kwargs = {'model_config': args.model_config,
                    'model_path': args.model_path,
                    'tracker_type': 'siam'}
    # Create a context to manage all tracks
    context = Context()
    frame_id = 0
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        ret, frame = cv2_video_reader.read_frame()
        if not ret:
            logger.info('End of video stream, ret is False!')
            break
        logger.debug(f'Processing frame {frame_id}-th')
        if frame_id == 1000 or frame_id == cv2_video_reader.total_frames:
            logger.info(f'Processing frame {frame_id}')
        my_frame = FrameWrapper(frame=frame, frame_id=frame_id)
        if frame_id in label_frames:  # If this is a label frame
            box_df = label_df[label_df['index'] == (frame_id // cv2_video_reader.fps)]
            for i in range(len(box_df)):
                # Extract box coordinates and category
                x, y = box_df.iloc[i]['xmin'], box_df.iloc[i]['ymin']
                w, h = box_df.iloc[i]['xmax'] - x, box_df.iloc[i]['ymax'] - y
                init_bbox = [x, y, w, h]
                object_name = box_df.iloc[i]['class']
                # Create a wrapper and draw
                box_wrapper = BoxWrapper(xmin=x, ymin=y,
                                         xmax=x + w,
                                         ymax=y + h,
                                         frame_id=frame_id, category=object_name,
                                         conf_score=1.0)
                my_frame.put_bbox(bbox=box_wrapper, color=ColorBGR.green)
                if object_name in context.tracks.keys():
                    # TODO: Prolong tracks?
                    # Currently, if there are two objects of the same category at a time,
                    # we choose the second one. By watching video output, we can know if
                    # objects of the same category switch.
                    track_kwargs['frame'] = my_frame.frame
                    track_kwargs['init_bbox'] = init_bbox
                    track_kwargs['object_name'] = object_name
                    context.tracks[object_name] = TrackerWrapper(**track_kwargs)
                else:
                    # Initialize a new track
                    track_kwargs['frame'] = my_frame.frame
                    track_kwargs['init_bbox'] = init_bbox
                    track_kwargs['object_name'] = object_name
                    context.tracks[object_name] = TrackerWrapper(**track_kwargs)
        else:  # If this is not a label frame
            for object_name, track_wrapper in context.tracks.items():
                # Track this object and draw a box
                outputs = track_wrapper.get_next_box(my_frame.frame)
                box_wrapper = BoxWrapper(xmin=outputs['bbox'][0], ymin=outputs['bbox'][1],
                                         xmax=outputs['bbox'][0] + outputs['bbox'][2],
                                         ymax=outputs['bbox'][1] + outputs['bbox'][3],
                                         frame_id=frame_id, category=object_name,
                                         conf_score=outputs['best_score'])
                my_frame.put_bbox(bbox=box_wrapper)

        cv2_video_writer.write_frame(my_frame.frame)
    cv2_video_reader.capture.release()
    cv2_video_writer.writer.release()
