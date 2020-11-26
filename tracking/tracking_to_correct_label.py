import cv2
import numpy as np
import torch
# torch.backends.cudnn.enabled=False
import sys

import os

sys.path.append(os.getcwd())
import csv
import pandas as pd
from time import perf_counter
from copy import deepcopy
from utils import TrackerWrapper, BoxWrapper, FrameWrapper, CV2VideoWriter, CV2VideoReader, \
    Context, ColorBGR, Sample, logger, parse_config, ColorRef
from utils import track_buffer, matching_and_merging, draw_context_on_frames, print_context

if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')
    # Create video writer and reader
    start = perf_counter()
    INPUT_VIDEO_PATH = os.path.join(args.input_video_dir, args.run + f'_trim.mp4')
    INPUT_LABEL_PATH = os.path.join(args.input_label_dir, args.run + f'_labels.csv')
    OUTPUT_VIDEO_FW = os.path.join(args.output_video_dir, args.run + f'_{args.tag}_fw.avi')
    OUTPUT_VIDEO_BW = os.path.join(args.output_video_dir, args.run + f'_{args.tag}_bw.avi')
    OUTPUT_VIDEO_MERGED = os.path.join(args.output_video_dir, args.run + f'_{args.tag}_merged.avi')
    OUTPUT_CSV_PATH = os.path.join(args.output_csv_dir, args.run + f'_{args.tag}.csv')

    cv2_video_reader = CV2VideoReader(INPUT_VIDEO_PATH)
    cv2_video_writer_fw = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_FW,
                                         width=cv2_video_reader.width,
                                         height=cv2_video_reader.height, fps=120)
    cv2_video_writer_bw = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_BW,
                                         width=cv2_video_reader.width,
                                         height=cv2_video_reader.height, fps=120)
    cv2_video_writer_merged = CV2VideoWriter(output_video_path=OUTPUT_VIDEO_MERGED,
                                             width=cv2_video_reader.width,
                                             height=cv2_video_reader.height, fps=120)
    csv_headers = ['frame', 'name', 'x', 'y', 'w', 'h', 'confidence', 'ground_truth']
    with open(OUTPUT_CSV_PATH, 'w') as g:
        writer = csv.writer(g)
        writer.writerow(csv_headers)
    label_df = pd.read_csv(INPUT_LABEL_PATH)
    labeled_seconds = np.array(sorted(label_df['index'].unique()))
    labeled_frames = labeled_seconds * cv2_video_reader.fps
    labeled_frames = list(labeled_frames.astype(np.int))
    cv2_video_reader.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0 - 1
    buffer_frames = dict()
    track_kwargs = dict(model_config=args.model_config, model_path=args.model_path,
                        tracker_type='siam')
    color_reference = ColorRef(ColorRef.forward_set)
    context_forward = Context(track_kwargs=track_kwargs, color_reference=color_reference)
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        ret, frame = cv2_video_reader.read_frame()
        if not ret:
            logger.info('End of video stream, ret is False!')
            break
        buffer_frames[frame_id] = frame
        if frame_id in labeled_frames:  # If this is a label frame
            # do forward tracking
            logger.info(
                f'Forward tracking from {frame_id - len(buffer_frames) + 1} to {frame_id}')
            context_forward = track_buffer(context=context_forward,
                                           buffer_frames=deepcopy(buffer_frames),
                                           label_df=label_df, labeled_frames=labeled_frames,
                                           is_backward=False, fps=cv2_video_reader.fps)
            draw_context_on_frames(context_forward, deepcopy(buffer_frames),
                                   cv2_video_writer_fw, labeled_frames=labeled_frames)
            # do backward tracking
            logger.info(
                f'Backward tracking from {frame_id} to {frame_id - len(buffer_frames) + 1}')
            # color_reference to assign color to boxes during bbox matching and tracking
            color_reference = ColorRef(ColorRef.backward_set)
            context_backward = Context(track_kwargs=track_kwargs,
                                       color_reference=color_reference)
            context_backward = track_buffer(context=context_backward,
                                            buffer_frames=deepcopy(buffer_frames),
                                            label_df=label_df, labeled_frames=labeled_frames,
                                            is_backward=True, fps=cv2_video_reader.fps)
            draw_context_on_frames(context_backward, deepcopy(buffer_frames),
                                   cv2_video_writer_bw, labeled_frames=labeled_frames)
            # merge backward and forward, some tracks of context_forward.tracks[...]
            # will be active after merging. context_merged is context_fw
            context_forward = matching_and_merging(context_forward, context_backward)
            logger.info(f"After merging")
            print_context(context_forward)
            draw_context_on_frames(context_forward, deepcopy(buffer_frames),
                                   cv2_video_writer_merged, labeled_frames=labeled_frames)
            # write csv tracking
            with open(OUTPUT_CSV_PATH, 'a') as g:
                for object_type, category_track in context_forward.tracks.items():
                    for object_id, track_wrapper in category_track.items():
                        for box_wrapper in track_wrapper.boxes:
                            if box_wrapper.frame_id in buffer_frames:
                                writer = csv.writer(g)
                                writer.writerow(box_wrapper.get_csv_row())
            # reset buffer
            buffer_frames = dict()
            # release backward tracks
            logger.info('Release backward trackers')
            for object_type, category_track in context_backward.tracks.items():
                for object_id, track_wrapper in category_track.items():
                    if track_wrapper.terminate:
                        track_wrapper.release_tracker()
            torch.cuda.empty_cache()

    cv2_video_writer_bw.writer.release()
    cv2_video_writer_fw.writer.release()
    cv2_video_writer_merged.writer.release()
    cv2_video_reader.capture.release()

    end = perf_counter()
    logger.info(f'Running time: {end - start}')
