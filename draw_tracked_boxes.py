#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:19:04 2020
Object Tracking Videos with Confidence Thresholding
@author: bezdek

"""
import csv
import cv2
import math
import pandas as pd

# Superimpose both forward and backward tracked boxes:

def draw_tracked_boxes(run,cam,inpath='../data',outpath='../data',back=True,framestep=2):  
  '''
  run: str, Run name, e.g., '2.2.1'
  cam: str, camera angle, 'C1','C2', or 'kinect'
  inpath: path to input files
  outpath: path to output location
  back: boolean, whether to include boxes from reverse tracking in addition to forward tracking
  framestep: step size of video frames to write. 1 is all frames, 2 is every 2 frames, etc.
  '''
  vidin=inpath+'/small_videos/'+run+'_'+cam+'_trim.mp4'
  forwarddf=pd.read_csv(inpath+'/'+run+'_'+cam+'_tracked_f.csv')
  if back:
      # Convert backward frames to forward frames
      fmax=forwarddf['frame'][forwarddf['ground_truth']==1].max()
      backwarddf=pd.read_csv(inpath+'/'+run+'_'+cam+'_tracked_r.csv')
      bmin=backwarddf['frame'][backwarddf['ground_truth']==1].min()
      backwarddf['frame']=(fmax+bmin)-backwarddf['frame']
      vidout=outpath+'/'+run+'_'+cam+'_f_and_r.avi'
  else:
      vidout=outpath+'/'+run+'_'+cam+'_f.avi'
  f = 0
  video_capture = cv2.VideoCapture()
  if video_capture.open(vidin):
    width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)/2.0
    #fps=29.97
    totalFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)/2.0)
    totalFramesEnd=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    totalDur = totalFrames/fps
    #!rm -f output.mp4 output.avi
    # can't write out mp4, so try to write into an AVI file
    video_writer = cv2.VideoWriter(vidout, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    thresh=.9 # minimum confidence for color scaling
    boxlen=0
    #step=labeldf['index'].max()
    counter=0
    forindex=forwarddf['frame'].unique()
    if back:
        backindex=backwarddf['frame'].unique()
    while video_capture.isOpened():
      print(f,'of',totalFramesEnd)
      sec=math.floor(f/fps)
      #print(sec)
      ret, frame = video_capture.read()
      if not ret:
        break
      if counter % framestep == 0:
        if f in forindex:
          boxdf=forwarddf[forwarddf['frame']==f]
          boxlen=len(boxdf)
          for i in range(boxlen):
            x,y=boxdf.iloc[i]['x'],boxdf.iloc[i]['y']
            w,h=boxdf.iloc[i]['w'], boxdf.iloc[i]['h']  
            name = boxdf.iloc[i]['name']
            conf= str(boxdf.iloc[i]['confidence'])
            if boxdf.iloc[i]['ground_truth']==1:
              cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 255, 0), 1)
              cv2.putText(frame, name, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            else:
              frac=max(0.0,(255*(float(conf))-(thresh*255))*10)
              cv2.rectangle(frame, (x, y),(x+w, y+h),(0,frac, 255-frac), 1)
              cv2.putText(frame, name+' '+conf, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  
        if back:
            if f in backindex:
              boxdf=backwarddf[backwarddf['frame']==f]
              boxlen=len(boxdf)
              for i in range(boxlen):
                x,y=boxdf.iloc[i]['x'],boxdf.iloc[i]['y']
                w,h=boxdf.iloc[i]['w'], boxdf.iloc[i]['h']  
                name = boxdf.iloc[i]['name']
                conf= str(boxdf.iloc[i]['confidence'])
                if boxdf.iloc[i]['ground_truth']==1:
                  cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 255, 0), 1)
                  cv2.putText(frame, name, (x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                else:
                  frac=max(0.0,(255*(float(conf))-(thresh*255))*10)
                  cv2.rectangle(frame, (x, y),(x+w, y+h),(0,frac, 255-frac), 1)
                  cv2.putText(frame, name+' '+conf, (x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        video_writer.write(frame)
      counter+=1  
      f += 1
      #only on first 100 frames for debugging
      #if f > 1000:
      #  break
      if f > totalFramesEnd:
        break
        
    video_capture.release()
    video_writer.release()
    
if __name__ == "__main__":   
    run='6.2.5'
    cam='C1'
    inpath='../data'
    outpath='../data'
    back=False
    draw_tracked_boxes(run,cam,inpath,outpath,back,framestep=2) 
    