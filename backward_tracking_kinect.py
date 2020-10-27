#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:04:11 2020

@author: bezdek
"""

'''
Backward Tracking Kinect
'''

import sys
#sys.path.append("/Users/bezdek/Documents/tracking/pysot")
sys.path.append("/scratch/mbezdek/pysot")
run=sys.argv[1]


# tracking function


def backward_video_tracking(vidpath,csvin,snapshot,config,csvout,framestep=2):
  import cv2
  import numpy as np
  import torch
  import math
  from pysot.core.config import cfg
  from pysot.models.model_builder import ModelBuilder
  from pysot.tracker.siamrpn_tracker import SiamRPNTracker
  import csv
  import pandas as pd
  #from tqdm import trange
  labeldf=pd.read_csv(csvin)
  label_frames=sorted(labeldf['index'].unique())
  #if small:
  #  labeldf[['width','height','xmin','ymin','xmax','ymax']]=labeldf[['width','height','xmin','ymin','xmax','ymax']].div(2).round().astype(int)
  cfg.merge_from_file(config)
  cfg.CUDA = torch.cuda.is_available()
  device = torch.device('cuda' if cfg.CUDA else 'cpu')
  csvheaders=['frame','name','x','y','w','h','confidence','ground_truth']
  with open(csvout, 'w') as g:
      writer = csv.writer(g)
      writer.writerow(csvheaders)
  #framestep=2
  f = 0
  video_capture = cv2.VideoCapture()
  if video_capture.open(vidpath):
    width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)/2.0
    #fps=29.97
    totalFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)/2.0)
    totalFramesEnd=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    totalDur = totalFrames/fps
    #!rm -f output.mp4 output.avi
    # can't write out mp4, so try to write into an AVI file
    #video_writer = cv2.VideoWriter(vidout, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    boxlen=0
    state=[]
    names=[]
    #step=labeldf['index'].max()
    counter=0
    idx=-1
    step = label_frames[idx]
    while video_capture.isOpened():
      #for f in trange(totalFramesEnd):
      #print(f,'of',totalFramesEnd)
      sec=math.floor(f/fps)
      #print(sec)
      ret, frame = video_capture.read()
      if not ret:
        break
      if counter % framestep == 0:
        roundsec=int(sec)
        if totalDur - (roundsec/2) <= step:
          #tindex=int(round(sec,0))
          boxdf=labeldf[labeldf['index']==step]
          boxlen=len(boxdf)
          state=[]
          names=[]
          for i in range(boxlen):
            x,y=boxdf.iloc[i]['xmin'],boxdf.iloc[i]['ymin']
            w,h=boxdf.iloc[i]['xmax'] - x, boxdf.iloc[i]['ymax'] - y  
            names.insert(i,boxdf.iloc[i]['class'])
            init_rect=[x,y,w,h]
            model = ModelBuilder()
            model.load_state_dict(torch.load(snapshot,
            map_location=lambda storage, loc: storage.cpu()))
            model.eval().to(device)
            state.insert(i,SiamRPNTracker(model))
            state[i].init(frame, init_rect)
            #cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 255, 0), 1)
            #cv2.putText(frame, names[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            rowout=[f,names[i],x,y,w,h,1.0,1]
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(rowout)
          #video_writer.write(frame)
          idx -= 1
          try:
              step=label_frames[idx]
          except:
              step=0
          
        else:
          # track
          #for i in state:
          for i in range(boxlen):
            outputs = state[i].track(frame)
            bbox = list(map(int, outputs['bbox']))
            #cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(0, 0, 255), 1)
            #cv2.putText(frame, names[i]+' '+str(outputs['best_score']), (bbox[0], bbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            rowout=[f,names[i],bbox[0], bbox[1],bbox[2],bbox[3],outputs['best_score'],0]
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(rowout)      
          #video_writer.write(frame)
        counter+=1
      else:
        counter+=1
    
      f += 1
      # only on first 100 frames
      if f > totalFramesEnd:
        break
        
    video_capture.release()
    #video_writer.release()
    
    # convert AVI to MP4
    #!ffmpeg -y -loglevel info -i $vidout $mp4out
  else:
    print("can't open the given input video file!")
  print('** tracking complete **')
    
# CHPC paths:
vidpath="/scratch/mbezdek/rev_videos/"+run+"_kinect_trim_rev.mp4"
csvin="/scratch/mbezdek/ground_truth_labels/"+run+"_kinect_labels.csv"
csvout="/scratch/mbezdek/rev/"+run+"_kinect_tracked_r.csv"
snapshot="/scratch/mbezdek/model/siamrpn_r50_1234_dwxcorr/model.pth"
config="/scratch/mbezdek/model/siamrpn_r50_1234_dwxcorr/config.yaml"


backward_video_tracking(vidpath,csvin,snapshot,config,csvout,framestep=2)