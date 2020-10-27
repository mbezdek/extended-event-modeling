#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:30:27 2020

@author: bezdek
"""

'''
INSTALL PYSOT
'''
#import os
#from os.path import exists, join, basename, splitext
#if not exists("/content/pysot"):
#  !git clone https://github.com/STVIR/pysot.git
#import os
#os.environ['PYTHONPATH'] += ':/Users/bezdek/Documents/tracking/pysot/'
#!pip install yacs colorama tensorboardX
#!pip install torch
#%cd '/content/pysot/'
#!python setup.py build_ext --inplace

'''
FORWARD TRACKING
'''



import sys
#sys.path.append("/Users/bezdek/Documents/tracking/pysot")
sys.path.append("/scratch/mbezdek/pysot")
run=sys.argv[1]



def forward_video_tracking(vidpath,csvin,snapshot,config,csvout,framestep=2):
  import math
  import csv
  import cv2
  #import torchvision
  #import cv2
  import numpy as np
  import pandas as pd
  import torch
  from pysot.core.config import cfg
  from pysot.models.model_builder import ModelBuilder
  from pysot.tracker.siamrpn_tracker import SiamRPNTracker
  #from tqdm import trange
  cfg.merge_from_file(config)
  cfg.CUDA = torch.cuda.is_available()
  device = torch.device('cuda' if cfg.CUDA else 'cpu')

  labeldf=pd.read_csv(csvin)
  label_frames=sorted(labeldf['index'].unique())
  #if small:
  #  labeldf[['width','height','xmin','ymin','xmax','ymax']]=labeldf[['width','height','xmin','ymin','xmax','ymax']].div(2).round().astype(int)
  csvheaders=['frame','name','x','y','w','h','confidence','ground_truth']
  with open(csvout, 'w') as g:
      writer = csv.writer(g)
      writer.writerow(csvheaders)

  framestep=2
  f = 0
  idx=0
  video_capture = cv2.VideoCapture()
  if video_capture.open(vidpath):
    width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)/2.0
    #fps=29.97
    totalFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)/2.0)
    totalFramesEnd = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
    #!rm -f output.mp4 output.avi
    # can't write out mp4, so try to write into an AVI file
    #video_writer = cv2.VideoWriter(vidout, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    boxlen=0
    state=[]
    names=[]
    step=label_frames[idx]
    counter=0
    while video_capture.isOpened():
      #for f in trange(totalFramesEnd):
      # print current frame:
      #print(f,'of',totalFramesEnd)
      #sec=round(f/fps,1)
      sec=math.floor(f/fps)
      #print(sec)
      ret, frame = video_capture.read()
      if not ret:
        break
      if counter % framestep == 0:

        roundsec=int(sec)
        if roundsec/2 >= step:
          #tindex=int(round(sec,0))
          boxdf=labeldf[labeldf['index']==step]
          boxlen=len(boxdf)
          state=[]
          names=[]
          for i in range(boxlen):
            x,y=boxdf.iloc[i]['xmin'],boxdf.iloc[i]['ymin']
            w,h=boxdf.iloc[i]['xmax'] - x, boxdf.iloc[i]['ymax'] - y  
            names.insert(i,boxdf.iloc[i]['class'])
            model = ModelBuilder()
            model.load_state_dict(torch.load(snapshot,
            map_location=lambda storage, loc: storage.cpu()))
            model.eval().to(device)
            init_rect=[x,y,w,h]
            state.insert(i,SiamRPNTracker(model))
            state[i].init(frame, init_rect)
            #cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 255, 0), 1)
            #cv2.putText(frame, names[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            rowout=[f,names[i],x,y,w,h,1.0,1]
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(rowout)
          #video_writer.write(frame)
          idx+=1
          try:
              step=label_frames[idx]
          except:
              step=999999999
          
        else:
          # track
          for i in range(boxlen):
            outputs = state[i].track(frame)
            bbox = list(map(int, outputs['bbox']))
            #cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(255, 0, 0), 1)
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
      # only on first 900 frames
      #if f > 305:
      #  break
      if f > totalFramesEnd:
        break
        
    video_capture.release()
    #video_writer.release()
    
    # convert AVI to MP4
    #!ffmpeg -y -loglevel info -i $vidout $mp4out
  else:
    print("can't open the given input video file!")
  print('*** tracking complete ***')
# CHPC paths:
vidpath="/scratch/mbezdek/small_videos/"+run+"_C2_trim.mp4"
csvin="/scratch/mbezdek/ground_truth_labels/"+run+"_C2_labels.csv"
csvout="/scratch/mbezdek/fwd/"+run+"_C2_tracked_f.csv"
snapshot="/scratch/mbezdek/model/siamrpn_r50_1234_dwxcorr/model.pth"
config="/scratch/mbezdek/model/siamrpn_r50_1234_dwxcorr/config.yaml"

'''
# local paths:
vidpath="/Users/bezdek/Documents/tracking/vid/"+run+"_C1_trim.mp4"
csvin="/Users/bezdek/Documents/tracking/"+run+"_C1_labels.csv"
csvout="/Users/bezdek/Documents/tracking/fwd/"+run+"_C1_tracked_f.csv"
snapshot="/Users/bezdek/Documents/tracking/model/siamrpn_r50_1234_dwxcorr/model.pth"
config="/Users/bezdek/Documents/tracking/model/siamrpn_r50_1234_dwxcorr/config.yaml"   
'''
forward_video_tracking(vidpath,csvin,snapshot,config,csvout,framestep=2)

