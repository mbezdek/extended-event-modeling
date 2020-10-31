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

    context = Context()
    # csvin = '1.1.4_C1_labels.csv'
    # labeldf = pd.read_csv(csvin)
    # label_frames = sorted(labeldf['index'].unique())
    cv2_video_reader = CV2VideoReader(input_video_path=args.input_video_path)
    cv2_video_writer = CV2VideoWriter(output_video_path=args.output_video_path,
                                      width=cv2_video_reader.width,
                                      height=cv2_video_reader.height)
    # cfg.merge_from_file(args.model_config)
    # cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    # device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # # device = 'cpu'
    # model = ModelBuilder()
    # model.load_state_dict(
    #     torch.load(args.model_path, map_location=lambda storage, loc: storage.cpu()))
    # model.to(device)
    # model.eval()
    # siam_tracker = build_tracker(model)
    # siam_tracker.init(frame, init_bbox)
    init_bbox = [100, 100, 50, 100]
    ret, frame = cv2_video_reader.read_frame()
    track_kwargs = {'model_config': args.model_config,
                    'model_path': args.model_path,
                    'tracker_type': 'siam',
                    'frame': frame,
                    'init_bbox': init_bbox}
    track_wrapper = TrackerWrapper(**track_kwargs)
    counter = 0
    while ret:
        counter += 1
        logger.debug(f'Processing frame {counter}')
        outputs = track_wrapper.get_next_box(frame)
        bbox = list(map(int, outputs['bbox']))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 0, 255), 1)
        cv2_video_writer.write_frame(frame)
        ret, frame = cv2_video_reader.read_frame()
