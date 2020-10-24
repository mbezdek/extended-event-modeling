import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import seaborn as sns
import cv2
import os
import logging
from shutil import rmtree
import skvideo

ffmpeg_path = 'C:/Users/nguye/ffmpeg-4.3.1-2020-10-01-full_build/bin'
skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io
from utils import CV2VideoReader, CV2VideoWriter, SegmentationVideo, Sample

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLVL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)


def draw_on_canvas(seg_points) -> np.ndarray:
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    # Do some plotting.
    ax = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    # sns.violinplot(data=seg_points, orient='h', ax=ax)
    sns.swarmplot(data=seg_points, orient='h', ax=ax, alpha=.1)
    sns.histplot(data=seg_points, bins=100)
    # sns.stripplot(data=seg_points, orient='h', ax=ax1, alpha=.1)
    ax2.vlines(seg_points, ymin=0, ymax=1, alpha=0.05)
    canvas.draw()
    img = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    canvas_resized = cv2.resize(
        img[:, int(img.shape[1] * 0.145): int(img.shape[1] * 0.88), :], (960, 200))
    return canvas_resized


def load_and_draw(input_video_path, input_segmentation, output_dir, output_video_name):
    # create directory for visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load segmentation data
    data_frame = pd.read_csv(input_segmentation)
    seg_video = SegmentationVideo(data_frame=data_frame, video_path=input_video_path)
    # initialize video_read and video_write streams
    cv2_video_reader = CV2VideoReader(input_video_path=input_video_path)
    output_video_path = os.path.join(output_dir, output_video_name)
    sk_video_writer = skvideo.io.FFmpegWriter(output_video_path)
    # cv2_video_writer = CV2VideoWriter(output_video_path=output_video_path)
    # calculate segmentation points according to fps
    # consider only one annotation
    condition = 'coarse'
    seg_points = seg_video.get_segments(n_annotators=100, condition=condition)
    seg_points = np.hstack(seg_points)
    seg_points = (seg_points * cv2_video_reader.fps).astype(np.int)
    # get canvas drawn with segmentation points
    canvas_img = draw_on_canvas(seg_points)
    # extracting frames while adding annotations to frames
    logger.info('Processing video...')
    frame_id = 0
    while cv2_video_reader.capture.isOpened():
        ret, frame = cv2_video_reader.capture.read()
        if ret is True:
            frame_id += 1
            # Add frame_id and condition (coarse or fine)
            cv2.putText(frame, f'Frame: {frame_id}', org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0))
            cv2.putText(frame, f'Condition: {condition}', org=(50, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0))
            if frame_id in seg_points:
                # Add tag for the segmented frame
                cv2.putText(frame, f'SEGMENTED', org=(50, 150),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 255, 0))
            frame = np.concatenate((frame, canvas_img), axis=0)
            # cv2_video_writer.writer.write(frame)
            sk_video_writer.writeFrame(frame[:, :, ::-1])
            # Real-time showing video
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else:
            break

    logger.info('Done load_and_draw!')


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
    load_and_draw(input_video_path='data/small_videos/6.2.5_C1_trim.mp4',
                  input_segmentation='database.200731.1.csv', output_dir='output',
                  output_video_name='output_video.mp4')
