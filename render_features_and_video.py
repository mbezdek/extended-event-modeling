#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:58:58 2020
Render feature graphs and 5x speed corpus videos
@author: bezdek
"""

#from glob import glob
import subprocess
import os
import sys
#from shutil import copyfile

comb_error_runs=[]
#for run in ['1.1.1','1.2.1','1.3.1','2.2.2','2.3.1','2.4.1','3.1.1','3.3.1','3.4.1','4.1.1','4.3.1','4.4.1','6.1.1','6.2.1','6.3.1']:
for run in ['4.4.4']:
    try:
        #featuresout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_features.csv'
        graphvideoin='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_hand_object_test.mp4'
        colorvideoin='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/small_videos/'+run+'_kinect_trim.mp4'
        fcolorvideoout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_fast.mp4'
        graphscaleout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_feature_test_scale.mp4'
        combinedout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/'+run+'_color_and_features.mp4'
        logout='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/modeling_features/log.txt'
        if not os.path.exists(fcolorvideoout):
            proc = subprocess.Popen(["ffmpeg",'-i',colorvideoin,'-an','-filter:v',"setpts=0.20*PTS",fcolorvideoout],stdout = subprocess.PIPE, stderr = subprocess.STDOUT,universal_newlines=True)
            with open(logout, 'a') as logfile:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logfile.write(line)
            proc.wait()
        if not os.path.exists(graphscaleout):
            proc = subprocess.Popen(["ffmpeg",'-i',graphvideoin,'-vf','scale=960:540',graphscaleout],stdout = subprocess.PIPE, stderr = subprocess.STDOUT,universal_newlines=True)
            with open(logout, 'a') as logfile:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logfile.write(line)
            proc.wait()
        if not os.path.exists(combinedout):
            proc = subprocess.Popen(["ffmpeg",'-i',fcolorvideoout,'-i',graphscaleout,'-filter_complex','hstack',combinedout],stdout = subprocess.PIPE, stderr = subprocess.STDOUT,universal_newlines=True)
            with open(logout, 'a') as logfile:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logfile.write(line)
            proc.wait()
    except:
        comb_error_runs.append(run)