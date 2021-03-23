import math
import sys

from matplotlib import animation as animation, pyplot as plt
import traceback

sys.path.append('.')
sys.path.append('../pysot')
from utils import parse_config, contain_substr
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import json
from joblib import Parallel, delayed
import glob
import joblib
import cv2
import logging

# Set-up logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(name)s - %(levelname)s - %(message)s'
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
logger.addHandler(c_handler)

def get_depth_region_sparse(pixelwise_matrix, mask_matrix, xmin, xmax, ymin, ymax):
    # Coordinates range from 0-1079
    xmin = round(xmin)
    xmax = round(xmax)
    ymin = round(ymin)
    ymax = round(ymax)
    sum_depth = np.sum(pixelwise_matrix[ymin: ymax + 1, xmin: xmax + 1])
    n_pixels = np.sum(mask_matrix[ymin: ymax + 1, xmin: xmax + 1])
    if n_pixels == 0:
        xmin_enlarged = xmin
        ymin_enlarged = ymin
        xmax_enlarged = xmax
        ymax_enlarged = ymax
        while n_pixels == 0:
            xmin_enlarged = max(xmin_enlarged - 10, 0)
            ymin_enlarged = max(ymin_enlarged - 10, 0)
            xmax_enlarged = min(xmax_enlarged + 10, mask_matrix.shape[1] - 1)
            ymax_enlarged = min(ymax_enlarged + 10, mask_matrix.shape[0] - 1)
            n_pixels = np.sum(mask_matrix[ymin_enlarged: ymax_enlarged + 1, xmin_enlarged:xmax_enlarged + 1])
        logger.debug(f'Enlarge box from [xmin, ymin, xmax, ymax]={[xmin, ymin, xmax, ymax]} to '
                     f'{[xmin_enlarged, ymin_enlarged, xmax_enlarged, ymax_enlarged]}')
        sum_depth = np.sum(pixelwise_matrix[ymin_enlarged: ymax_enlarged + 1, xmin_enlarged:xmax_enlarged + 1])
        return sum_depth / n_pixels
    else:
        return sum_depth / n_pixels


def sample_joints(df, ptimes):
    '''
    # Get skeleton 3D (objectPoints) and 2D (cameraPoints) coordinates to estimate camera parameters:
    Input
    df : (pandas dataframe) skeleton dataframe with 3D and 2D joint coordinates
    ptimes : (list or 1D array of integers) rows to sample from the skeleton df to estimate camera parameters
    Output
    imagePoints : (length of ptimes * 5 joints) x 2 numpy array of sampled 2D coodinates
    objectPoints :(length of ptimes * 5 joints) x 3 numpy array of sampled 3D coordinates
    '''
    # key joints: Head, HandLeft, HandRight, FootLeft, FootRight
    js = ['3', '7', '11', '15', '19']
    imagePoints = np.full((len(js), len(ptimes), 2), np.nan)
    objectPoints = np.full((len(js), len(ptimes), 3), np.nan)
    for pi, p in enumerate(ptimes):
        for ji, j in enumerate(js):
            imagePoints[ji, pi, 0] = df['J' + j + '_2D_X'].iloc[p]
            imagePoints[ji, pi, 1] = df['J' + j + '_2D_Y'].iloc[p]
            objectPoints[ji, pi, 0] = df['J' + j + '_3D_X'].iloc[p]
            objectPoints[ji, pi, 1] = df['J' + j + '_3D_Y'].iloc[p]
            objectPoints[ji, pi, 2] = df['J' + j + '_3D_Z'].iloc[p]
    imagePoints = imagePoints.reshape(len(js) * len(ptimes), 2)
    objectPoints = objectPoints.reshape(len(js) * len(ptimes), 3)
    imagePoints = imagePoints.astype('float32')
    objectPoints = objectPoints.astype('float32')
    return imagePoints, objectPoints


def find_3D_point(cameraPoint, depthPoint, mtx, rvecs, tvecs):
    '''
    # Compute 3D X and 3D Y for object from 2D X, 2D Y, and 3D Z:
    Input:
        cameraPoint : (tuple) (x,y) coordinates in camera pixel space, (0,0) is top left of image
        depthPoint : (float) Z distance corresponding to camera point
        mtx : 3 x 3 numpy array of camera matrix
        rvecs : (list of a numpy array) camera rotational vector
        tvecs : (list of a numpy array) camera translation vector
    Output:
        spacePoint : 1 x 3 numpy array of estimated X, Y, and Z object coordinates
    '''
    point = np.array([cameraPoint[0], cameraPoint[1], 1.0])
    # Convert rotational vector to rotational matrix:
    rmat, _ = cv2.Rodrigues(rvecs[0])
    leftSideMat = np.linalg.inv(rmat).dot(np.linalg.inv(mtx)).dot(point)
    rightSideMat = np.linalg.inv(rmat).dot(tvecs[0])
    s = (depthPoint + rightSideMat[2]) / leftSideMat[2]
    spacePoint = np.linalg.inv(rmat).dot(np.transpose((s * np.linalg.inv(mtx)).dot(np.transpose(point)) - np.transpose(tvecs[0])))
    spacePoint = np.transpose(spacePoint)
    return spacePoint


def screen_distance(x1, y1, x2, y2):
    """
    Calculate distance xy plane
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    return np.linalg.norm(point1 - point2)


def depth_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate distance in a 3D space
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :return:
    """
    point1 = np.array([x1, y1, z1])
    point2 = np.array([x2, y2, z2])
    return np.linalg.norm(point1 - point2)


def gen_objhand_feature(args, run, tag):
    try:
        # FPS is used to concat track_df and skel_df, should be inferred from run
        if 'kinect' in run:
            fps = 25
            skel_csv = os.path.join(args.input_skel_csv,
                                    run.replace('_kinect', '') + '_skel_clean.csv')
        else:
            fps = 30
            skel_csv = os.path.join(args.input_skel_csv, run + '_skel.csv')
        args.run = run
        args.tag = tag
        logger.info(f'Config {args}')
        track_csv = os.path.join(args.input_track_csv, run + '_r50.csv')
        output_csv = os.path.join(args.output_objhand_csv, run + '_objhand.csv')
        # Read tracking result
        track_df = pd.read_csv(track_csv)
        # Scale tracking results because tracking is in 960x540 and skeleton is in 1920x1080
        track_df.loc[:, 'x'] = track_df.loc[:, 'x'] * 2
        track_df.loc[:, 'y'] = track_df.loc[:, 'y'] * 2
        track_df.loc[:, 'w'] = track_df.loc[:, 'w'] * 2
        track_df.loc[:, 'h'] = track_df.loc[:, 'h'] * 2
        track_df = calc_center(track_df)

        # ------Initialize camera model for the run------
        skel_df = pd.read_csv(skel_csv)
        # Choose rows of skeleton df to sample for 3D and 2D coordinates:
        ptimes = np.arange(skel_df.index[0], skel_df.index[-1], 25)
        # extract object and image points:
        imagePoints, objectPoints = sample_joints(skel_df, ptimes)
        # If there are null values in imagePoints or objectPoints, camera calibration is null as well, making all xyz_3d null
        null_indices = np.isnan(np.hstack([imagePoints, objectPoints])).any(axis=1)
        imagePoints = imagePoints[~null_indices]
        objectPoints = objectPoints[~null_indices]
        # Estimate the camera matrix
        idim = (1920, 1080)
        mtx = cv2.initCameraMatrix2D([objectPoints], [imagePoints], idim)
        # generate camera matrix, distortion, rotation, and translation parameters:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objectPoints], [imagePoints], idim, mtx, None,
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        # ------Iterate through key frames and calculate 3D coordinates for objects------
        all_arrays = glob.glob(f"output/depth/{run.replace('_kinect', '')}/*.joblib")
        key_frame_ids = list(set([int(os.path.basename(arr).split('_')[0]) for arr in all_arrays]))
        depth_df = pd.DataFrame(index=track_df.index, columns=['3D_x', '3D_y', 'z'], dtype=np.float)
        for frame_id in sorted(key_frame_ids):
            pixel_array = joblib.load(f"output/depth/{run.replace('_kinect', '')}/{frame_id}_pixel_array.joblib")
            mask_array = joblib.load(f"output/depth/{run.replace('_kinect', '')}/{frame_id}_mask_array.joblib")
            # For each key frame, select existing objects and calculate depth
            for line, row in track_df[track_df['frame'] == frame_id].iterrows():
                xmin, ymin, xmax, ymax = row['x'], row['y'], row['x'] + row['w'], row['y'] + row['h']
                # Already tested, using accumulated matrix or pixelwise matrix yield the exactly same result. Summing pixelwise
                # matrix is less time-consuming, relative to calculating accumulated matrix. #objects in a frame is small.
                # region_depth = get_depth_region(accu_array, mask_array, xmin, xmax, ymin, ymax)
                region_depth = get_depth_region_sparse(pixel_array, mask_array, xmin, xmax, ymin, ymax)
                # skeleton depth values is in meter while kinect extracted values (to calculate objects) is in millimeter
                region_depth /= 1000
                x_cent = (xmin + xmax) / 2
                y_cent = (ymin + ymax) / 2
                # returned values can be null, making dist_z (later) null. One reason is null is camera calibration
                xyz_3d = find_3D_point(cameraPoint=(x_cent, y_cent), depthPoint=region_depth, mtx=mtx, rvecs=rvecs, tvecs=tvecs)
                if np.any(pd.notnull(depth_df.iloc[line, :])):
                    logger.warning('the same object at a particular frame is assigned twice!!')
                depth_df.iloc[line, :] = xyz_3d[0]  # xyz_3d is a 2d-array
        track_df = pd.concat([track_df, depth_df], axis=1)

        # -----Read skeleton result and set index by frame to merge tracking and skeleton-----
        skel_df = pd.read_csv(skel_csv)
        # sync_time, Right Hand: J11_2D_X, J11_2D_Y
        hand_df = skel_df.loc[:, ['sync_time', 'J11_2D_X', 'J11_2D_Y', 'J11_3D_X', 'J11_3D_Y', 'J11_3D_Z']]
        hand_df['frame'] = (hand_df.loc[:, 'sync_time'] * fps).apply(round).astype(np.int)
        hand_df.set_index('frame', drop=False, verify_integrity=False, inplace=True)
        # Converting from second to frame can introduce duplicated frames -> remove
        hand_df = hand_df[~hand_df.index.duplicated(keep='first')]
        # Process tracking result to create object dataframe
        final_frameid = max(max(hand_df['frame']), max(track_df['frame']))
        objs = track_df['name'].unique()
        objs_df = pd.DataFrame(index=range(final_frameid))
        objs_df.index.name = 'frame'
        # For each object instance, create a set of columns and add values to objs_df. There will be a lot of null values
        for obj in objs:
            obj_df = track_df[track_df['name'] == obj].set_index('frame', drop=False,
                                                                 verify_integrity=True)
            objs_df[obj + '_x_cent'] = obj_df['x_cent']
            objs_df[obj + '_y_cent'] = obj_df['y_cent']
            objs_df[obj + '_x'] = obj_df['x']
            objs_df[obj + '_y'] = obj_df['y']
            objs_df[obj + '_z'] = obj_df['z']
            objs_df[obj + '_3D_x'] = obj_df['3D_x']
            objs_df[obj + '_3D_y'] = obj_df['3D_y']
            objs_df[obj + '_w'] = obj_df['w']
            objs_df[obj + '_h'] = obj_df['h']
            objs_df[obj + '_confidence'] = obj_df['confidence']
        logger.info('Combine hand dataframe and objects dataframe')
        # Concat hand_df and objs_df into a single DataFrame, either pd.concat or combine_first is ok,
        # maybe duplicated 'frame' column, so combine_first is safer
        # Note: pandas dataframes operation rely on index
        # objhand_df = pd.concat([hand_df, objs_df], axis=1)
        objhand_df = hand_df.combine_first(objs_df)
        objhand_df = objhand_df.sort_index()
        # Process null entry by interpolation, only for joints because for objects xy coordinates,
        # they are tracked frame-by-frame already.
        # However, because depth values is inferred from key frames only, need to interpolate. Also, for resampling consistency,
        # depth values should be extrapolate until the first frame it appears, this first frame should be inferred from xy
        logger.info('Interpolate Joints and Depth value')
        logger.info('For Joints, limit_area is inside (interpolate inside only), for Depth, limit_direction is both '
                    '(interpolate and extrapolate in both directions, but within a bound inferred by x existing)')
        for obj in objs:
            # frames where this object is existing
            bound_index = objhand_df[obj + '_x'].dropna().index
            # Interestingly, changing supposed_a_view does NOT change objhand_df[obj + '_z'] -> supposed_a_view is a copy
            # supposed_a_view = objhand_df[obj + '_z'].iloc[bound_index]
            # supposed_a_view.interpolate(limit_direction='both', inplace=True)
            # Interpolate and extrapolate both direction within the bound
            objhand_df[obj + '_z'].iloc[bound_index] = objhand_df[obj + '_z'].iloc[bound_index].interpolate(
                limit_direction='both')
            objhand_df[obj + '_3D_x'].iloc[bound_index] = objhand_df[obj + '_3D_x'].iloc[bound_index].interpolate(
                limit_direction='both')
            objhand_df[obj + '_3D_y'].iloc[bound_index] = objhand_df[obj + '_3D_y'].iloc[bound_index].interpolate(
                limit_direction='both')
        objhand_df['J11_2D_X'] = objhand_df['J11_2D_X'].interpolate(limit_area='inside')
        objhand_df['J11_2D_Y'] = objhand_df['J11_2D_Y'].interpolate(limit_area='inside')
        objhand_df['J11_3D_X'] = objhand_df['J11_3D_X'].interpolate(limit_area='inside')
        objhand_df['J11_3D_Y'] = objhand_df['J11_3D_Y'].interpolate(limit_area='inside')
        objhand_df['J11_3D_Z'] = objhand_df['J11_3D_Z'].interpolate(limit_area='inside')

        # Smooth movements
        logger.info('Gaussian filtering')
        for obj in objs:
            objhand_df[obj + '_x_cent'] = gaussian_filter1d(objhand_df[obj + '_x_cent'], 3)
            objhand_df[obj + '_y_cent'] = gaussian_filter1d(objhand_df[obj + '_y_cent'], 3)
            objhand_df[obj + '_x'] = gaussian_filter1d(objhand_df[obj + '_x'], 3)
            objhand_df[obj + '_y'] = gaussian_filter1d(objhand_df[obj + '_y'], 3)
            objhand_df[obj + '_z'] = gaussian_filter1d(objhand_df[obj + '_z'], 3)
            objhand_df[obj + '_3D_x'] = gaussian_filter1d(objhand_df[obj + '_3D_x'], 3)
            objhand_df[obj + '_3D_y'] = gaussian_filter1d(objhand_df[obj + '_3D_y'], 3)
            objhand_df[obj + '_w'] = gaussian_filter1d(objhand_df[obj + '_w'], 3)
            objhand_df[obj + '_h'] = gaussian_filter1d(objhand_df[obj + '_h'], 3)
        objhand_df['J11_2D_X'] = gaussian_filter1d(objhand_df['J11_2D_X'], 3)
        objhand_df['J11_2D_Y'] = gaussian_filter1d(objhand_df['J11_2D_Y'], 3)
        objhand_df['J11_3D_X'] = gaussian_filter1d(objhand_df['J11_3D_X'], 3)
        objhand_df['J11_3D_Y'] = gaussian_filter1d(objhand_df['J11_3D_Y'], 3)
        objhand_df['J11_3D_Z'] = gaussian_filter1d(objhand_df['J11_3D_Z'], 3)
        # Let do resampling when combining with other features while running SEM, not here
        # objhand_df.loc[:, 'sync_time'] = objhand_df.index / fps
        # objhand_df.loc[:, 'frame'] = objhand_df.index
        # resampledf = resample_df(objhand_df, rate='333ms')
        # resampledf['frame'] = resampledf['frame'].apply(round)
        resampledf = objhand_df
        resampledf['frame'] = resampledf.index  # To concatenate with other features
        # Calculate distances between all objects and hand
        logger.info('Calculate object-hand distances')
        # condition np.all(pd.notnull(x))) to ensure a valid row, e.g. J11_3D_Z have more values than J11_2D_X
        for obj in objs:
            resampledf[obj + '_dist'] = resampledf[
                [obj + '_x_cent', obj + '_y_cent', 'J11_2D_X', 'J11_2D_Y']].apply(
                lambda x: screen_distance(x[0], x[1], x[2], x[3])
                if (np.all(pd.notnull(x))) else np.nan, axis=1)
            resampledf[obj + '_dist_z'] = resampledf[
                [obj + '_3D_x', obj + '_3D_y', obj + '_z', 'J11_3D_X', 'J11_3D_Y', 'J11_3D_Z']].apply(
                lambda x: depth_distance(x[0], x[1], x[2], x[3], x[4], x[5])
                if (np.all(pd.notnull(x))) else np.nan, axis=1)
        resampledf.to_csv(output_csv, index=False)
        logger.info(f'Done Objhand {run}')
        with open('objhand_complete.txt', 'a') as f:
            f.write(run + '\n')
        return track_csv, skel_csv, output_csv
    except Exception as e:
        with open('objhand_error.txt', 'a') as f:
            f.write(run + '\n')
            f.write(repr(e) + '\n')
            f.write(traceback.format_exc() + '\n')
        return None, None, None


def calculateDistance(x1, y1, x2, y2):
    if (x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


def boxDistance(rx, ry, rw, rh, px, py):
    # find shortest distance between a point and a axis-aligned bounding box.
    # rx,ry: bottom left corner of rectangle
    # rw: rectangle width
    # rh: rectangle height
    # px,py: point coordinates
    dx = max([rx - px, 0, px - (rx + rw)])
    dy = max([ry - py, 0, py - (ry + rh)])
    return math.sqrt(dx * dx + dy * dy)


def calc_center(df):
    # Input: a dataframe with x,y,w, & h columns.
    # Output: dataframe with added columns for x and y center of each box.
    df['x_cent'] = df['x'] + (df['w'] / 2.0)
    df['y_cent'] = df['y'] + (df['h'] / 2.0)
    return df


def animate_video(resampledf, output_video_path):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=14, bitrate=1800)
    valcols = [i for i in resampledf.columns if i not in ['sync_time']]
    fdflong = pd.melt(resampledf, id_vars=['sync_time'], value_vars=valcols)
    fdflong = fdflong.sort_values('sync_time')

    times = fdflong['sync_time'].unique()
    NUM_COLORS = len(fdflong['variable'].unique())

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(0. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    fig, ax = plt.subplots(figsize=(14, 8))
    thresh = 299

    def draw_barchart(current_time):
        fdff = fdflong[fdflong['sync_time'].eq(current_time)].sort_values('variable')
        mask = fdff['value'] > thresh
        fdff.loc[mask, 'value'] = -1
        ax.clear()
        ax.barh(fdff['variable'], fdff['value'], color=colors)
        ax.text(0, 0.4, round(current_time, 2), transform=ax.transAxes, color='#777777',
                size=45, ha='right', weight=800)
        plt.xlim([-1, thresh])
        ax.set_axisbelow(True)
        ax.text(-1, 1.15, 'Object Movement',
                transform=ax.transAxes, size=23, weight=600, ha='left', va='top')
        # plt.box(False)

    draw_barchart(times[9])
    fig, ax = plt.subplots(figsize=(14, 8))
    animator = animation.FuncAnimation(fig, draw_barchart, frames=times)
    # HTML(animator.to_jshtml())
    animator.save(output_video_path, writer=writer)


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

    # gen_feature_video(track_csv=args.input_track_csv, skel_csv=args.input_skel_csv,
    #                   output_csv=args.output_objhand_csv)

    # runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    # runs = ['1.1.5_C1', '4.4.5_C1']
    tag = 'feb_12'
    if '.txt' not in args.run:
        gen_objhand_feature(args, runs[0], tag)
    else:
        res = Parallel(n_jobs=16)(delayed(
            gen_objhand_feature)(args, run, tag) for run in runs)
        track_csvs, skel_csvs, output_csvs = zip(*res)
        results = dict()
        for i, run in enumerate(runs):
            results[run] = dict(track_csv=track_csvs[i], skel_csv=skel_csvs[i],
                                output_csv=output_csvs[i])
        with open('results_objhand.json', 'w') as f:
            json.dump(results, f)
