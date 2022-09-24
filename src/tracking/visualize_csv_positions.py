import numpy as np
# torch.backends.cudnn.enabled=False
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
from src.utils import BoxWrapper, FrameWrapper, CV2VideoWriter, CV2VideoReader, \
    ColorBGR, logger, parse_config

if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')
    # Create video writer and reader
    cv2_video_reader = CV2VideoReader(input_video_path=args.input_video_path)
    cv2_video_writer = CV2VideoWriter(output_video_path=args.output_video_path,
                                      width=cv2_video_reader.width,
                                      height=cv2_video_reader.height)
    logger.info(f'Video width={cv2_video_reader.width}')
    logger.info(f'Video height={cv2_video_reader.height}')
    # Parse label file
    label_df = pd.read_csv(args.input_label_path)
    if 'index' in label_df.columns:
        label_times = np.array(sorted(label_df['index'].unique()))
        label_frames = label_times * cv2_video_reader.fps
        label_frames = list(label_frames.astype(np.int))
    else:
        label_frames = np.array(sorted(label_df['frame'].unique()))
    logger.info(f'label_frames {label_frames} \n'
                f'fps {cv2_video_reader.fps}')
    frame_id = 0
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        ret, frame = cv2_video_reader.read_frame()
        if not ret:
            logger.info('End of video stream, ret is False!')
            break
        if frame_id % 100 == 0 or frame_id == cv2_video_reader.total_frames:
            logger.info(f'Processing frame {frame_id}')
        if frame_id in label_frames:  # If this is a label frame
            my_frame = FrameWrapper(frame=frame, frame_id=frame_id)
            my_frame.put_text(f'FrameID: {frame_id}')
            if 'index' in label_df.columns:
                box_df = label_df[label_df['index'] == round(frame_id / cv2_video_reader.fps)]
            else:
                label_df['xmax'] = label_df['x'] + label_df['w']
                label_df['xmin'] = label_df['x']
                label_df['ymax'] = label_df['y'] + label_df['h']
                label_df['ymin'] = label_df['y']
                label_df['class'] = label_df['name']
                box_df = label_df[label_df['frame'] == frame_id]
            for i in range(len(box_df)):
                # Extract box coordinates and category
                x, y = box_df.iloc[i]['xmin'], box_df.iloc[i]['ymin']
                w, h = box_df.iloc[i]['xmax'] - x, box_df.iloc[i]['ymax'] - y
                object_name = box_df.iloc[i]['class']
                x = x * cv2_video_reader.width / box_df.iloc[i]['width']
                w = w * cv2_video_reader.width / box_df.iloc[i]['width']
                y = y * cv2_video_reader.height / box_df.iloc[i]['height']
                h = h * cv2_video_reader.height / box_df.iloc[i]['height']
                # Create a wrapper, write csv, and draw
                box_wrapper = BoxWrapper(xmin=x, ymin=y, xmax=x + w, ymax=y + h,
                                         frame_id=frame_id, object_name=object_name,
                                         conf_score=1.0, state='init')
                my_frame.put_bbox(bbox=box_wrapper, color=ColorBGR.green)
            cv2_video_writer.write_frame(my_frame.frame)
    cv2_video_reader.capture.release()
    cv2_video_writer.writer.release()
