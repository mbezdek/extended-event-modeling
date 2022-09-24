import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info('vidfeatures.py')
import sys
import traceback

sys.path.append('.')

import cv2
import numpy as np
import csv
import os
import json
from joblib import Parallel, delayed
from scipy.stats.stats import pearsonr
from utils import CV2VideoReader, parse_config, contain_substr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def gen_vid_features(args, run, tag):
    try:
        logger.info(f'Config {args}')
        import cv2
        csv_headers = ['frame', 'optical_flow_avg', 'pixel_correlation']
        input_video_path = os.path.join(args.input_video_path, run + '_trim.mp4')
        output_csv_path = os.path.join(args.output_csv_path, f'{run}_{tag}_video_features.csv')
        with open(output_csv_path, 'w') as g:
            writer = csv.writer(g)
            writer.writerow(csv_headers)
        # cv2_video_reader = CV2VideoReader(input_video_path=input_video_path)
        from vidgear.gears import VideoGear
        vg_video_reader = VideoGear(source=input_video_path, stabilize=True).start()
        frame_id = 0
        # ret, frame = cv2_video_reader.read_frame()
        prev_gray = None
        # while cv2_video_reader.capture.isOpened():
        # for frame in mvp_video_reader.iter_frames():
        while True:
            frame_id += 1
            frame = vg_video_reader.read()
            if frame is None:
                logger.info('End of video stream, frame is None!')
                break
            # ret, frame = cv2_video_reader.read_frame()
            # if not ret:
            #     logger.info('End of video stream, ret is False!')
            #     break
            if frame_id % int(args.skip_frame):
                continue

            logger.debug(f'Processing frame {frame_id}')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # compute dense optical flow:
            if prev_gray is None:
                prev_gray = gray
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, float(args.pyr_scale),
                                                int(args.levels), int(args.winsize),
                                                int(args.iterations),
                                                int(args.poly_n), float(args.poly_sigma),
                                                int(args.flags))
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            cor, p = pearsonr(gray.flatten(), prev_gray.flatten())
            with open(output_csv_path, 'a') as g:
                writer = csv.writer(g)
                writer.writerow([str(frame_id), str(np.mean(magnitude)), str(1.0 - cor)])
            prev_gray = gray
        # cv2_video_reader.capture.release()
        vg_video_reader.stop()
        logger.info(f'Done Vid {run}')
        with open(f'optical_complete_{tag}.txt', 'a') as f:
            f.write(run + '\n')
        return input_video_path, output_csv_path
    except Exception as e:
        with open(f'optical_error_{tag}.txt', 'a') as f:
            f.write(run + '\n')
            f.write(repr(e) + '\n')
            f.write(traceback.format_exc() + '\n')
        return None, None


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    if '.txt' in args.run:
        choose = ['kinect']
        # choose = ['C1']
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if contain_substr(run, choose)]
    else:
        runs = [args.run]

    # runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    if os.path.exists(f'optical_complete_{args.feature_tag}.txt'):
        os.remove(f'optical_complete_{args.feature_tag}.txt')
    if os.path.exists(f'optical_error_{args.feature_tag}.txt'):
        os.remove(f'optical_error_{args.feature_tag}.txt')
    if not os.path.exists(args.output_csv_path):
        os.makedirs(args.output_csv_path)
    res = Parallel(n_jobs=8, prefer="threads")(delayed(
        gen_vid_features)(args, run, args.feature_tag) for run in runs)
