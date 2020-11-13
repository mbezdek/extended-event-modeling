import cv2
import numpy as np
import pandas as pd
import csv
import sys
import os
sys.path.append(os.getcwd())
from scipy.stats.stats import pearsonr
from collections import namedtuple
import argparse
import configparser
from utils import CV2VideoReader, logger, parse_config

if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')

    csv_headers = ['frame', 'optical_flow_avg','pixel_correlation']
    with open(args.output_csv_path, 'w') as g:
        writer = csv.writer(g)
        writer.writerow(csv_headers)
    cv2_video_reader = CV2VideoReader(input_video_path=args.input_video_path)
    frame_id = 0
    ret, frame = cv2_video_reader.read_frame()
    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        ret, frame = cv2_video_reader.read_frame()
        if not ret:
            logger.info('End of video stream, ret is False!')
            break
        logger.debug(f'Processing frame {frame_id}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # compute dense optical flow:
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, float(args.pyr_scale),
                                            int(args.levels), int(args.winsize), int(args.iterations),
                                            int(args.poly_n), float(args.poly_sigma), int(args.flags))
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        cor,p=pearsonr(gray.flatten(),prevgray.flatten())
        with open(args.output_csv_path, 'a') as g:
            writer = csv.writer(g)
            writer.writerow([str(frame_id),str(np.mean(magnitude)),str(1.0-cor)])
        prevgray = gray
    cv2_video_reader.capture.release()


