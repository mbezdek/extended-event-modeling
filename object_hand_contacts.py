#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate Object/Hand Contacts using multicamera views
Created on Tue Sep 22 11:54:44 2020

@author: bezdek
"""

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
#import seaborn as sns
import os
#from scipy.interpolate import interp1d
#from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.animation as animation

import math  
def calculateDistance(x1,y1,x2,y2):
    if (x1,y1,x2,y2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return distance
    
def boxDistance(rx,ry,rw,rh, px,py):
    # find shortest distance between a point and a axis-aligned bounding box.
    # rx,ry: bottom left corner of rectangle
    # rw: rectangle width
    # rh: rectangle height
    # px,py: point coordinates
    dx = max([rx - px, 0, px - (rx+rw)])
    dy = max([ry - py, 0, py - (ry+rh)])
    return math.sqrt(dx*dx + dy*dy)

#boxDistance(2,2,2,2,4,4)

def calc_center(df):
    # Input: a dataframe with x,y,w, & h columns.
    # Output: dataframe with added columns for x and y center of each box.
    df['x_cent']=df['x']+(df['w']/2.0)
    df['y_cent']=df['y']+(df['h']/2.0)
    return df
#Resampling function:
def objresample(testdf,rate='40ms'):
    #tracklist=[i for i in testdf.columns if i[-4:]=='cent' or i[-1]=='X' or i[-1]=='Y']
    #tracklist=[i for i in testdf.columns]
    testdf = testdf.set_index(pd.to_datetime(testdf['sync_time'], unit='s'), drop=False)
    resample_index = pd.date_range(start=testdf.index[0], end=testdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
    outdf=testdf.combine_first(dummy_frame).interpolate('time')
    outdf=testdf.combine_first(dummy_frame).interpolate('time').resample(rate).last()
    
    #for col in tracklist:
    #    outdf[col]=testdf[col].interpolate('time').resample(rate).last().fillna(0)
    return outdf

def gen_feature_video(run):
    #kinecttrackingfiles=glob('/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/*kinect_merged.csv')
    kinecttrackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/'+run+'_kinect_merged.csv'
    #outdir='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/object_movement'
    skelfile='/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_data/'+run+'_skel_clean.csv'
    skelC1file='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/camera_calibration/projected_skeletons/'+run+'_C1_skel.csv'
    skelC2file='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/camera_calibration/projected_skeletons/'+run+'_C2_skel.csv'
    
    featuresout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_features.csv'
    videoout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_hand_object_test.mp4'
    
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    
    print(run)
    C1trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C1_merged.csv'
    C2trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C2_merged.csv'      
    #C1trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C1_merged.csv'
    #C2trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C2_merged.csv'
    #kinecttrackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_kinect_merged.csv'
    #run=os.path.splitext(os.path.basename(C1trackingfile))[0][0:-10]
    tdfC1=pd.read_csv(C1trackingfile)
    tdfC2=pd.read_csv(C2trackingfile)
    tdfkinect=pd.read_csv(kinecttrackingfile)
    
    tdfkinect['x']=960-tdfkinect['x']-tdfkinect['w']
    #tdfkinect['w']=(-1)*tdfkinect['w']
    # Check plots of frames and index:
    #plt.plot(tdfC1.index)
    #plt.plot(tdfC2.index)
    #plt.plot(tdfkinect.index)
    #plt.plot(tdfC1['frame'])
    #print(tdf['frame'].max())
    tdfkinect['name']=tdfkinect['name']+'_'+tdfkinect.groupby(['frame','name']).cumcount().add(1).astype(str)
    tdfC1['name']=tdfC1['name']+'_'+tdfC1.groupby(['frame','name']).cumcount().add(1).astype(str)
    tdfC2['name']=tdfC2['name']+'_'+tdfC2.groupby(['frame','name']).cumcount().add(1).astype(str)
    
    tdfC1 = calc_center(tdfC1)
    tdfC2 = calc_center(tdfC2)
    tdfkinect = calc_center(tdfkinect)
    tdfC1['sync_time'] = tdfC1['frame']/30.0
    tdfC2['sync_time'] = tdfC2['frame']/30.0
    tdfkinect['sync_time'] = tdfkinect['frame']/25.0
    objdfs=[tdfkinect,tdfC1,tdfC2]
    
    skeldf=pd.read_csv(skelfile)
    skelC1df=pd.read_csv(skelC1file)
    skelC2df=pd.read_csv(skelC2file)
    skeldfs=[skeldf,skelC1df,skelC2df]
    
    distdfs=[]
    n=0
    while n<3:
        # sync_time, Right Hand: J11_2D_X, J11_2D_Y
        handdf=skeldfs[n][['sync_time', 'J11_2D_X', 'J11_2D_Y']]
        '''
        # **move merging to after long to wide calculation below
        objKdf=pd.concat([handdf,tdfkinect.sort_values(by=['name'])],ignore_index=True,sort=False)
        
        objKdf=objKdf.sort_values(by=['sync_time'])
        '''
        objs=objdfs[n]['name'].unique()
        '''
        for obj in objs:    
            objKdf['x_cent'][objKdf['name']==obj]=objKdf['x_cent'][objKdf['name']==obj].interpolate(method='linear')
            objKdf['y_cent'][objKdf['name']==obj]=objKdf['y_cent'][objKdf['name']==obj].interpolate(method='linear')
            objKdf['J11_2D_X'][objKdf['name']==obj]=objKdf['J11_2D_X'][objKdf['name']==obj].interpolate(method='linear')
            objKdf['J11_2D_Y'][objKdf['name']==obj]=objKdf['J11_2D_Y'][objKdf['name']==obj].interpolate(method='linear')
        #testdf=testdf.set_index('sync_time')
        testdf['J11_2D_X']=testdf['J11_2D_X']/2
        testdf['J11_2D_Y']=testdf['J11_2D_Y']/2
        '''
        objKdf=pd.DataFrame()
        for obj in objs:
            print(obj)
            testdf=objdfs[n][objdfs[n]['name']==obj]
            #testdf['x_cent1']=testdf['x_cent'].shift(1)
            #testdf['y_cent1']=testdf['y_cent'].shift(1)
            #testdf[obj+'_move']=testdf[['x_cent1','y_cent1','x_cent','y_cent']].apply(lambda x: calculateDistance(x[0],x[1],x[2],x[3]) if(np.all(pd.notnull(x))) else np.nan, axis = 1)
            testdf=testdf.set_index('sync_time',drop=False)
            objKdf[obj+'_x_cent']=testdf['x_cent']
            objKdf[obj+'_y_cent']=testdf['y_cent']
            objKdf[obj+'_x']=testdf['x']
            objKdf[obj+'_y']=testdf['y']
            objKdf[obj+'_w']=testdf['w']
            objKdf[obj+'_h']=testdf['h']
            objKdf[obj+'_confidence']=testdf['confidence']
            #movedf[obj+'_move']=testdf[obj+'_move']
        
        objKdf['sync_time']=objKdf.index
        
        objKdf=pd.concat([handdf,objKdf],ignore_index=True,sort=False)
        
        objKdf=objKdf.sort_values(by=['sync_time'])
        
        for obj in objs:    
            objKdf[obj+'_x_cent']=objKdf[obj+'_x_cent'].interpolate(method='linear')
            objKdf[obj+'_y_cent']=objKdf[obj+'_y_cent'].interpolate(method='linear')
            objKdf[obj+'_x']=objKdf[obj+'_x'].interpolate(method='linear')
            objKdf[obj+'_y']=objKdf[obj+'_y'].interpolate(method='linear')
            objKdf[obj+'_w']=objKdf[obj+'_w'].interpolate(method='linear')
            objKdf[obj+'_h']=objKdf[obj+'_h'].interpolate(method='linear')
        objKdf['J11_2D_X']=objKdf['J11_2D_X'].interpolate(method='linear')
        objKdf['J11_2D_Y']=objKdf['J11_2D_Y'].interpolate(method='linear')
        #testdf=testdf.set_index('sync_time')
        objKdf['J11_2D_X']=objKdf['J11_2D_X']/2
        objKdf['J11_2D_Y']=objKdf['J11_2D_Y']/2
        
        # Apply Gaussian filter to object movement:
        # sigma=3
        # # compute FWHM for reference:
        # FWHM=sigma * np.sqrt(2 * np.log(2)) * 2 / np.sqrt(2)
        
        for obj in objs:
            objKdf[obj+'_x_cent']= gaussian_filter1d(objKdf[obj+'_x_cent'], 3)
            objKdf[obj+'_y_cent']= gaussian_filter1d(objKdf[obj+'_y_cent'], 3)
            objKdf[obj+'_x']= gaussian_filter1d(objKdf[obj+'_x'], 3)
            objKdf[obj+'_y']= gaussian_filter1d(objKdf[obj+'_y'], 3)
            objKdf[obj+'_w']= gaussian_filter1d(objKdf[obj+'_w'], 3)
            objKdf[obj+'_h']= gaussian_filter1d(objKdf[obj+'_h'], 3)
        objKdf['J11_2D_X']= gaussian_filter1d(objKdf['J11_2D_X'], 3)
        objKdf['J11_2D_Y']= gaussian_filter1d(objKdf['J11_2D_Y'], 3)
            #movedf=movedf.drop([obj+'_move'], axis=1)
                
        
        
        resampledf=objresample(objKdf,rate='333ms')
        for obj in objs:
            resampledf[obj+'_dist']=resampledf[[obj+'_x',obj+'_y',obj+'_w',obj+'_h','J11_2D_X','J11_2D_Y']].apply(lambda x: boxDistance(x[0],x[1],x[2],x[3],x[4],x[5]) if(np.all(pd.notnull(x))) else np.nan, axis = 1)    
        distdfs.append(resampledf)
        n+=1
    
    
    a=[i for i in distdfs[0].columns if i[-4:]=='dist']
    b=[i for i in distdfs[1].columns if i[-4:]=='dist']
    c=[i for i in distdfs[2].columns if i[-4:]=='dist']
    # common items in all camera angles:
    d = list(set(a) & set(b) & set(c))
    
    fig, axs = plt.subplots(3)
    # 2.2.1, 'dumbbell_1' works well:
    idx=5
    fig=plt.figure()
    for idx in range(0,14):
        plt.subplot(4, 4, idx+1)
        a=distdfs[0][d[idx]][:-5].fillna(distdfs[0][d[idx]][:-5].mean())
        b=distdfs[1][d[idx]][:-5].fillna(distdfs[1][d[idx]][:-5].mean())
        c=distdfs[2][d[idx]][:-5].fillna(distdfs[2][d[idx]][:-5].mean())
        plt.plot(distdfs[0]['sync_time'][:-5],(a+b+c))
        plt.ylim([0,500])
        plt.title(d[idx])
    
    plt.tight_layout()
    plt.show()
    
    combinedf=pd.DataFrame()
    combinedf['sync_time']=distdfs[0]['sync_time'][:-5]
    for idx in range(0,14):
        a=distdfs[0][d[idx]][:-5].fillna(distdfs[0][d[idx]][:-5].mean())
        b=distdfs[1][d[idx]][:-5].fillna(distdfs[1][d[idx]][:-5].mean())
        c=distdfs[2][d[idx]][:-5].fillna(distdfs[2][d[idx]][:-5].mean())
        combinedf[d[idx]]=(a+b+c)
    
    combinedf.to_csv(featuresout,index=False)
    '''
    fig=plt.figure()
    plt.scatter(tdfC1['x_cent'][tdfC1['name']==d[7][:-5]],tdfC1['y_cent'][tdfC1['name']==d[7][:-5]],label='C1')
    plt.scatter(tdfC2['x_cent'][tdfC2['name']==d[7][:-5]],tdfC2['y_cent'][tdfC2['name']==d[7][:-5]],label='C2')
    plt.scatter(tdfkinect['x_cent'][tdfkinect['name']==d[7][:-5]],tdfkinect['y_cent'][tdfkinect['name']==d[7][:-5]],label='kinect')
    plt.legend()
    plt.show()
    '''
    
    # feature video:
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    valcols=[i for i in combinedf.columns if i not in ['sync_time']]
    fdflong=pd.melt(combinedf,id_vars=['sync_time'],value_vars=valcols)
    fdflong=fdflong.sort_values('sync_time')
    
    times=fdflong['sync_time'].unique()
    NUM_COLORS = len(fdflong['variable'].unique())
    
    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    thresh=300
    def draw_barchart(current_time):
        
        fdff = fdflong[fdflong['sync_time'].eq(current_time)].sort_values('variable')
        fdff['value'][fdff['value'] > thresh]=0
        #fdff['value'][fdff['value'] <= 200]=200-fdff['value'][fdff['value'] <= 200]
        ax.clear()
        ax.barh(fdff['variable'], fdff['value'], color=colors)
        #dx = fdff['value'].max() / 200
        
        #for i, (value, name) in enumerate(zip(fdff['value'], fdff['variable'])):
        #    ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        #    ax.text(value-dx, i-.25, name, size=10, color='#444444', ha='right', va='baseline')
        #    ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
        # Print current sync time in gray:
        ax.text(1, 0.4, round(current_time,2), transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
        #ax.text(0, 1.06, 'Optional Gray title', transform=ax.transAxes, size=12, color='#777777')
        #plt.xlim([min(fdflong['value']),max(fdflong['value'])])
        plt.xlim([0,thresh])
        #ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        #ax.tick_params(axis='x', colors='#777777', labelsize=12)
        #ax.set_yticks([])
        #ax.margins(0, 0.01)
        #ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        ax.text(0, 1.15, 'Object Movement',
                transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
        #plt.box(False)
        
    draw_barchart(times[10])
    fig, ax = plt.subplots(figsize=(15, 8))
    animator = animation.FuncAnimation(fig, draw_barchart, frames=times)
    #HTML(animator.to_jshtml())
    animator.save(videoout, writer=writer)
runs=glob('/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/camera_calibration/projected_skeletons/*_C1_skel.csv')        
runs=[i[100:-12] for i in runs]
#run='2.2.1'
error_runs=[]
for run in runs:    
    try:
        gen_feature_video(run)
    except:
        error_runs.append(run)