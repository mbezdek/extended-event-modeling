#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:38:02 2020

@author: bezdek
"""
import sys
#sys.path.append("/Users/bezdek/Documents/tracking/pysot")
sys.path.append("/scratch/mbezdek/pysot")
run=sys.argv[1]



def overlap(row1,row2):
  # determine the coordinates of the intersection rectangle
  x_left = max(row1['x'], row2['x'])
  y_top = max(row1['y'], row2['y'])
  x_right = min(row1['x']+row1['w'], row2['x']+row2['w'])
  y_bottom = min(row1['y']+row1['h'], row2['y']+row2['h'])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (row1['w']) * (row1['h'])
  bb2_area = (row2['w']) * (row2['h'])

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou
'''
def euclid_dist(row1,row2):
  # determine the coordinates of the center
  x1_cent = row1['x']+(row1['w']/2.0)
  x2_cent = row2['x']+(row2['w']/2.0)
  y1_cent = row1['y']+(row1['h']/2.0)
  y2_cent = row2['y']+(row2['h']/2.0)
  return euclidean((x1_cent,y1_cent),(x2_cent,y2_cent))
'''

def merge_forward_and_backward_csvs(forwardcsv,backwardcsv,csvout):

  import csv
  from tqdm import trange
  import pandas as pd
  forwarddf=pd.read_csv(forwardcsv)
  backwarddf=pd.read_csv(backwardcsv)
  fmax=forwarddf['frame'][forwarddf['ground_truth']==1].max()
  bmin=backwarddf['frame'][backwarddf['ground_truth']==1].min()
  backwarddf['frame']=(fmax+bmin)-backwarddf['frame']

  maxf=max(backwarddf['frame'].max(),forwarddf['frame'].max())

  csvheaders=['frame','name','x','y','w','h','confidence','ground_truth']
  with open(csvout, 'w') as g:
      writer = csv.writer(g)
      writer.writerow(csvheaders)

  for f in trange(maxf+1):
    try:
      forboxdf=forwarddf[forwarddf['frame']==f]
    except:
      forboxdf=pd.DataFrame()
    try:
      backboxdf=backwarddf[backwarddf['frame']==f]
    except:
      backboxdf=pd.DataFrame()  
    forboxlen=len(forboxdf)
    backboxlen=len(backboxdf)
    for i in range(forboxlen):
      # find out how many matching labels are in the forward and backward dfs:
      matchname=forboxdf['name'].iloc[i]
      matchfordf=forboxdf[forboxdf['name']==matchname]
      matchbackdf=backboxdf[backboxdf['name']==matchname]
      forboxlen=len(forboxdf)
      backboxlen=len(backboxdf)
      #print(len(matchfordf),len(matchbackdf))
      if len(matchbackdf)==0:
        # If no matches print to output df
        with open(csvout, 'a') as g:
          writer = csv.writer(g)
          writer.writerow(forboxdf.iloc[i])   
      elif len(matchbackdf)==1 and len(matchfordf)==1:
        # If 1 match in backboxdf, delete box with lower confidence and print the other to the output df. There can be only one!
        if forboxdf['confidence'].iloc[i] >= matchbackdf['confidence'].iloc[0]:
          #print('forward bigger')
          backboxdf=backboxdf[backboxdf['name'] != matchname]
          with open(csvout, 'a') as g:
            writer = csv.writer(g)
            writer.writerow(forboxdf.iloc[i])   
        elif forboxdf['confidence'].iloc[i] < matchbackdf['confidence'].iloc[0]:
          with open(csvout, 'a') as g:
            writer = csv.writer(g)
            writer.writerow(matchbackdf.iloc[0])   
          backboxdf=backboxdf[backboxdf['name'] != matchname]
      elif len(matchbackdf)>1 or len(matchfordf)>1:
        # If more than 1 label matches, try to find the one with shortest distance:
        k=0
        overlap_dict={}
        for k in range(backboxlen):
          if backboxdf['name'].iloc[k]==matchname:
            #print(overlap(forboxdf.iloc[i],backboxdf.iloc[k]))
            overlap_dict[k]=overlap(forboxdf.iloc[i],backboxdf.iloc[k])
          k+=1
        max_index=max(overlap_dict, key=overlap_dict.get)
        #print(max_index)
        if overlap_dict[max_index] > 0:
          if forboxdf['confidence'].iloc[i] >= backboxdf['confidence'].iloc[max_index]:
            #print('forward bigger')
            bad_df=backboxdf.index.isin([backboxdf.iloc[max_index].name])
            backboxdf=backboxdf[~bad_df]
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(forboxdf.iloc[i])   
          elif forboxdf['confidence'].iloc[i] < matchbackdf['confidence'].iloc[0]:
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(backboxdf.iloc[max_index])   
            bad_df=backboxdf.index.isin([backboxdf.iloc[max_index].name])
            backboxdf=backboxdf[~bad_df]
        else:
            with open(csvout, 'a') as g:
              writer = csv.writer(g)
              writer.writerow(forboxdf.iloc[i])   
        #print('     ')
      else:
        print('anomaly - len matchfordf',str(len(matchfordf)),'len matchbackdf',str(len(matchbackdf)))
    backboxlen=len(backboxdf)
    for i in range(backboxlen):
      with open(csvout, 'a') as g:
        writer = csv.writer(g)
        writer.writerow(backboxdf.iloc[i])   
        
# Creating one csv from the forward and backward labels:
forwardcsv="/scratch/mbezdek/fwd/"+run+"_tracked_f.csv"
backwardcsv="/scratch/mbezdek/rev/"+run+"_tracked_r.csv"
#vidpath="/content/drive/My Drive/tracking/vid/2.2.1_C1_trim.mp4"
#vidout="/content/drive/My Drive/tracking/2.2.1_C1_clip_f_and_r.avi"
csvout="/scratch/mbezdek/merged/"+run+"_merged.csv"

merge_forward_and_backward_csvs(forwardcsv,backwardcsv,csvout)
