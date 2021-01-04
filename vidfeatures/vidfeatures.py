import sys
import traceback

sys.path.append('.')
sys.path.append('../pysot')

import cv2
import numpy as np
import csv
import os
import json
from joblib import Parallel, delayed
from scipy.stats.stats import pearsonr
from utils import CV2VideoReader, logger, parse_config, contain_substr

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def gen_vid_features(args, run, tag):
    try:
        args.run = run
        args.tag = tag
        logger.info(f'Config {args}')

        csv_headers = ['frame', 'optical_flow_avg', 'pixel_correlation']
        input_video_path = os.path.join(args.input_video_path, run + '_trim.mp4')
        output_csv_path = os.path.join(args.output_csv_path, run + '_video_features.csv')
        with open(output_csv_path, 'w') as g:
            writer = csv.writer(g)
            writer.writerow(csv_headers)
        cv2_video_reader = CV2VideoReader(input_video_path=input_video_path)
        frame_id = 0
        ret, frame = cv2_video_reader.read_frame()
        prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while cv2_video_reader.capture.isOpened():
            frame_id += 1
            ret, frame = cv2_video_reader.read_frame()
            if not ret:
                logger.info('End of video stream, ret is False!')
                break
            if frame_id % int(args.skip_frame):
                continue

            logger.debug(f'Processing frame {frame_id}')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # compute dense optical flow:
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, float(args.pyr_scale),
                                                int(args.levels), int(args.winsize),
                                                int(args.iterations),
                                                int(args.poly_n), float(args.poly_sigma),
                                                int(args.flags))
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            cor, p = pearsonr(gray.flatten(), prevgray.flatten())
            with open(output_csv_path, 'a') as g:
                writer = csv.writer(g)
                writer.writerow([str(frame_id), str(np.mean(magnitude)), str(1.0 - cor)])
            prevgray = gray
        cv2_video_reader.capture.release()

        logger.info(f'Done Vid {run}')
        with open('vid_complete.txt', 'a') as f:
            f.write(run + '\n')
        return input_video_path, output_csv_path
    except Exception as e:
        with open('vid_error.txt', 'a') as f:
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
    tag = '_dec_28'
    res = Parallel(n_jobs=16)(delayed(
        gen_vid_features)(args, run, tag) for run in runs)
    input_video_paths, output_csv_paths = zip(*res)
    results = dict()
    for i, run in enumerate(runs):
        results[run] = dict(inpput_video_path=input_video_paths[i],
                            output_csv_path=output_csv_paths[i])
    with open('results_vid_features.json', 'w') as f:
        json.dump(results, f)
