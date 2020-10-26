import numpy as np
import pandas as pd
import cv2
import os
import logging
from shutil import rmtree
from time import perf_counter
import skvideo

ffmpeg_path = 'C:/Users/nguye/ffmpeg-4.3.1-2020-10-01-full_build/bin'
skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io
from utils import CV2VideoReader, CV2VideoWriter, SegmentationVideo, Canvas, MyFrame, ColorBGR

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLVL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)


def draw_segmentations(input_video_path, input_segmentation, output_dir, output_video_name):
    # create directory for visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load segmentation data
    data_frame = pd.read_csv(input_segmentation)
    seg_video = SegmentationVideo(data_frame=data_frame, video_path=input_video_path)
    # initialize video_read and video_write streams
    cv2_video_reader = CV2VideoReader(input_video_path=input_video_path)
    output_video_path = os.path.join(output_dir, output_video_name)
    # sk_video_writer = skvideo.io.FFmpegWriter(output_video_path)
    # logger.debug(f'Video writer: {type(sk_video_writer)}')
    cv2_video_writer = CV2VideoWriter(output_video_path=output_video_path)
    # calculate segmentation points according to fps
    # consider only one annotation
    condition = 'coarse'
    seg_points = seg_video.get_segments(n_annotators=100, condition=condition)
    seg_points = np.hstack(seg_points)
    # get canvas drawn with segmentation points
    canvas_agg = Canvas(rows=3, columns=1)
    canvas_agg.draw_on_canvas(seg_points)
    canvas_img = canvas_agg.get_current_canvas()
    # extracting frames while adding annotations to frames
    logger.info('Processing video...')
    # Transfer seg_points to frame_id rather than seconds
    seg_points = (seg_points * cv2_video_reader.fps).astype(np.int)
    frame_id = 0
    ret, frame = cv2_video_reader.read_frame()
    while ret is True:
        frame_id += 1
        my_frame = MyFrame(frame)
        # Add frame_id and condition (coarse or fine)
        my_frame.put_text(f'Frame: {frame_id}')
        my_frame.put_text(f'Condition: {condition}')
        if frame_id in seg_points:
            # Add tag for the segmented frame
            my_frame.put_text(f'SEGMENTED', color=ColorBGR.green)
        my_frame.frame = np.concatenate((my_frame.frame, canvas_img), axis=0)
        cv2_video_writer.write_frame(my_frame.frame)
        # sk_video_writer.writeFrame(my_frame.frame[:, :, ::-1])
        # Real-time showing video
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        ret, frame = cv2_video_reader.read_frame()
    if ret is False:
        logger.info('End of reading stream, ret is False!')

    # sk_video_writer.close()
    logger.info('Done!')


def draw_images():
    pass
    # drawing segmented regions from buffer
    # for seg_point in seg_points:
    #     logger.debug(f'Segmented frame_id: {seg_point}')
    #     if os.path.exists(f'{output_dir}/{seg_point}'):
    #         rmtree(f'{output_dir}/{seg_point}')
    #     os.makedirs(f'{output_dir}/{seg_point}')
    #     for i in range(seg_point - region_length, seg_point + region_length, 3):
    #         cv2.imwrite(f'{output_dir}/{seg_point}/{i}.jpg', buffer[i])


if __name__ == "__main__":
    logger.info('Input video: data/small_videos/6.2.5_C1_trim.mp4')
    logger.info('Input segmentation: database.200731.1.csv')
    logger.info('Output dir: output/')
    start = perf_counter()
    draw_segmentations(input_video_path='data/small_videos/6.2.5_C1_trim.mp4',
                       input_segmentation='database.200731.1.csv', output_dir='output',
                       output_video_name='output_video_test.avi')
    end = perf_counter()
    logger.info(f'Running time: {end - start}')
