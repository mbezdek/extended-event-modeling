import numpy as np
import pandas as pd
import cv2
import os
import logging
from shutil import rmtree
from typing import List
# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLVL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)


def string_to_segments(raw_string: str) -> List:
    raw_string = raw_string.split('\n')
    list_segments = [float(x.split(' ')[1]) for x in raw_string if 'BreakPoint' in x]
    return list_segments


def load_and_draw(input_video_path, input_segmentation, output_dir, output_video_name, region_length):
    # create directory for visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load segmentation data
    seg = pd.read_csv(input_segmentation)
    # select relevant annotations for the video
    # one video can have multiple annotations
    seg = seg[seg['movie1'] == os.path.splitext(os.path.basename(input_video_path))[0]]
    # parse annotations, from string to a list of breakpoints for each annotation
    seg['segment_processed'] = seg['segment1']
    seg['segment_processed'] = seg['segment_processed'].apply(string_to_segments)
    # initialize video_read and video_write streams
    cap = cv2.VideoCapture(input_video_path)
    if cap.isOpened() is False:
        logger.error("Error opening video stream or file")
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    output_video_name = os.path.join(output_dir, output_video_name)
    out = cv2.VideoWriter(output_video_name, fourcc=fourcc, fps=fps, frameSize=(width, height))
    # calculate segmentation points according to fps
    # consider only one annotation
    condition = seg['condition'].iloc[0]
    seg_points = seg['segment_processed'].values[0]
    seg_points = np.array(seg_points)
    seg_points = (seg_points * fps).astype(np.int)
    # extracting frames while adding annotations to frames
    frame_id = 0
    buffer = dict()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frame_id += 1
            # Add frame_id and condition (coarse or fine)
            cv2.putText(frame, f'Frame: {frame_id}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0))
            cv2.putText(frame, f'Condition: {condition}', org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0))
            if frame_id in seg_points:
                # Add tag for the segmented frame
                cv2.putText(frame, f'SEGMENTED', org=(50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 255, 0))
                # cv2.imwrite(f'{frame_id}.jpg', frame)
            out.write(frame)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            buffer[frame_id] = frame
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # drawing segmented regions from buffer
    for seg_point in seg_points:
        logger.debug(f'Segmented frame_id: {seg_point}')
        if os.path.exists(f'{output_dir}/{seg_point}'):
            rmtree(f'{output_dir}/{seg_point}')
        os.makedirs(f'{output_dir}/{seg_point}')
        for i in range(seg_point - region_length, seg_point + region_length, 3):
            cv2.imwrite(f'{output_dir}/{seg_point}/{i}.jpg', buffer[i])


if __name__ == "__main__":
    logger.info('Input video: data/small_videos/6.2.5_C1_trim.mp4')
    logger.info('Input segmentation: database.200731.1.csv')
    logger.info('Output dir: output/')
    load_and_draw(input_video_path='data/small_videos/6.2.5_C1_trim.mp4',
                  input_segmentation='database.200731.1.csv', output_dir='output',
                  output_video_name='output_video.mp4', region_length=60)
