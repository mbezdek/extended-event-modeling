#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:43:42 2019

@author: bezdek

update 191114:
    skip joints with NaN values

update 191111:
    draw tracking confidence (tracked/inferred) on joints
    use mov files for runs with shrunken mp4 files
    resample to correct frame rate for mov files
    skip files if output already exists
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import trange

#user='Eichenbaum'

'''
list of 12 colors:
Red,Lime	,Blue,Yellow	,Cyan,Magenta,Maroon,Olive,Green	,Purple,Teal	,Navy
'''
colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),
        (0,255,255),(255,0,255),(128,0,0),(128,128,0),
        (0,128,0),(128,0,128),(0,128,128),(0,0,128)]

# Location of skeleton csv files: 
#csv_dir="/Users/"+user+"/Box/DCL_ARCHIVE/Project-SEM-Corpus/extracted-3dcoords-2dvideocoords-dRGB-5aug19/"
#timing_file="/Users/"+user+"/Box/DCL_ARCHIVE/Project-SEM-Corpus/timing_start_stop_clap.xlsx"

#LOCAL:
skel_dir="/Users/bezdek/Documents/clean_skeleton_data/"
videofolder='/Users/bezdek/Documents/trimmed_videos/'
outdir="/Users/bezdek/Documents/clean_skeleton_videos/"

#skel_corr_file="/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/movie_clicker/skeleton_correction.xlsx"

small=True
'''
#BOX:
skel_dir="/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_data/"
videofolder='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/small_videos/'
outdir="/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_videos/"
'''
#timingdf=pd.read_excel(timing_file,sheet_name='Keepers')
# column 3 is Camera
# Column 25 is kinect filename
#timingdf.rename(columns={ timingdf.columns[4]: "claptime" }, inplace = True)
#timingdf.rename(columns={ timingdf.columns[9]: "starttime" }, inplace = True)
#timingdf.rename(columns={ timingdf.columns[14]: "stoptime" }, inplace = True)

#skel_corr_df=pd.read_excel(skel_corr_file)

#skelfiles=glob.glob(skel_dir+'*')
#skelfiles=[skelfiles[0]]
#'/Users/bezdek/Documents/clean_skeleton_data/1.3.9_skel_clean.csv'
#skelfiles=['/Users/bezdek/Documents/clean_skeleton_data/3.3.5_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/3.3.6_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/1.3.1_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/1.3.3_skel_clean.csv',         
#           '/Users/bezdek/Documents/clean_skeleton_data/3.3.2_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/3.3.3_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/3.3.4_skel_clean.csv',
#           '/Users/bezdek/Documents/clean_skeleton_data/3.3.8_skel_clean.csv',]
#skelfiles=['/Users/bezdek/Documents/clean_skeleton_data/2.2.6_skel_clean.csv']
#skelfiles=['/Users/bezdek/Documents/clean_skeleton_data/3.1.1_skel_clean.csv']
#skelfiles=['/Users/bezdek/Documents/clean_skeleton_data/1.3.1_skel_clean.csv']
#skelfiles=["/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_data/1.2.5_skel_clean.csv"]
skelfiles=['/Users/bezdek/Documents/clean_skeleton_data/1.2.5_skel_clean.csv']
'''
#print(sorted([i[99:-7] for i in skelfiles]))
runs=[i[99:-7] for i in skelfiles]
# [f(x) if condition else g(x) for x in sequence]
runs=[i[:4]+'0'+i[4:]if len(i)==5 else i for i in runs]
runs=sorted(runs)

#Check number of runs per chapter:
stem=[i[99:102] for i in skelfiles]
for i in sorted(set(stem)):
        print(i,':',stem.count(i))
'''

'''
Functions to summarize skeleton data:
'''
def bodystats(df):
    i=0
    bodies=df['body'].unique()
    bodystat=pd.DataFrame(columns=['id','min','max'])
    while i < len(bodies):
        temp=df[df['body']==bodies[i]]
        tempd={'id':bodies[i],'min':np.min(temp['sync_time']),'max':np.max(temp['sync_time'])}
        tempdf = pd.DataFrame([tempd])
        bodystat=bodystat.append(tempdf,sort=False)
        i+=1
    return bodystat 

def bodysplit(df):
    bodies=df[1].unique()
    i=0
    bodydata=[]
    while i < len(bodies):
        temp=df[df[1]==bodies[i]]
        bodydata.append(temp)
        i+=1
    return bodydata 

#testdf[[col]].astype('category').interpolate('time').resample(rate).nearest()
#for i in skeldf.columns:

#Resampling function:
def skelresample(testdf,rate='40ms'):
    #tracklist=[2,3,7,8,12,13,17,18,22,23,27,28,32,33,37,38,42,43,47,48,52,53,57,58,62,63,67,68,72,73,77,78,82,83,87,88,92,93,98,103,108,113,118,123]
    #tracklist=[i for i in testdf.columns if testdf[i].dtypes=='O']
    tracklist=[k for k in testdf.columns if 'Tracked' in k]
    #testdf=df.apply(pd.to_numeric, errors='coerce')
    testdf = testdf.set_index(pd.to_datetime(testdf['sync_time'], unit='s'), drop=False)
    resample_index = pd.date_range(start=testdf.index[0], end=testdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
    #df.combine_first(dummy_frame).interpolate('time').iloc[:6]
    outdf=testdf.combine_first(dummy_frame).interpolate('time').resample(rate).mean()
    #for col in tracklist:
    tracklist2D=[k for k in testdf.columns if '2D' in k]
    for col in tracklist:
        outdf[[col]]=testdf[[col]].astype('category').resample(rate).nearest()
    #for col in [128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]:
    #for col in range(128,178):
    for col in tracklist2D:
        #outdf[col]=outdf[col].dropna().astype(np.int64)
        if small:
            outdf[[col]]=outdf[[col]].divide(2).fillna(value=-1).astype(int)
        else:
            outdf[[col]]=outdf[[col]].fillna(value=-1).astype(int)
    #outdf[3]=testdf[3].interpolate('time').resample(rate).nearest()
    #outdf = outdf.reindex(columns=sorted(outdf.columns))
    return outdf

def drawskel(r,frame,i=0,color=(255,0,0),thickness=4):
    # r = row of a skeleton data frame
    # frame = image frame to draw on
    # i = body index
    # color = RGB tuple for color
    # thinkness = bone thickness
    #r=r.dropna().astype(int)
    
    #for col in [128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]:
    #for col in range(128,178):
    #    if ~np.isnan(r[col]):
    #        r[col]=int(r[col])
        #r[col]=r[col].to_numeric(errors='coerce',downcast='integer')
    #r=r.to_numeric(errors='coerce',downcast='integer')
    #spine
    if (all(x > 0 for x in [r[78],r[79],r[80],r[81]])):
        cv2.line(frame,(r[78],r[79]),(r[80],r[81]),color,thickness)
    if (all(x > 0 for x in [r[80],r[81],r[118],r[119]])):
        cv2.line(frame,(r[80],r[81]),(r[118],r[119]),color,thickness)
    if (all(x > 0 for x in [r[118],r[119],r[82],r[83]])):
        cv2.line(frame,(r[118],r[119]),(r[82],r[83]),color,thickness)
    if (all(x > 0 for x in [r[82],r[83],r[84],r[85]])):
        cv2.line(frame,(r[82],r[83]),(r[84],r[85]),color,thickness)
    #shoulder
    if (all(x > 0 for x in [r[118],r[119],r[94],r[95]])):
        cv2.line(frame,(r[118],r[119]),(r[94],r[95]),color,thickness)
    if (all(x > 0 for x in [r[118],r[119],r[86],r[87]])):
        cv2.line(frame,(r[118],r[119]),(r[86],r[87]),color,thickness)
    #left arm
    if (all(x > 0 for x in [r[86],r[87],r[88],r[89]])):
        cv2.line(frame,(r[86],r[87]),(r[88],r[89]),color,thickness)
    if (all(x > 0 for x in [r[88],r[89],r[90],r[91]])):
        cv2.line(frame,(r[88],r[89]),(r[90],r[91]),color,thickness)
    if (all(x > 0 for x in [r[90],r[91],r[92],r[93]])):
        cv2.line(frame,(r[90],r[91]),(r[92],r[93]),color,thickness)
    if (all(x > 0 for x in [r[92],r[93],r[120],r[121]])):
        cv2.line(frame,(r[92],r[93]),(r[120],r[121]),color,thickness)
    if (all(x > 0 for x in [r[90],r[91],r[122],r[123]])):
        cv2.line(frame,(r[90],r[91]),(r[122],r[123]),color,thickness)
    #right arm
    if (all(x > 0 for x in [r[94],r[95],r[96],r[97]])):
        cv2.line(frame,(r[94],r[95]),(r[96],r[97]),color,thickness)
    if (all(x > 0 for x in [r[96],r[97],r[98],r[99]])):
        cv2.line(frame,(r[96],r[97]),(r[98],r[99]),color,thickness)
    if (all(x > 0 for x in [r[98],r[99],r[100],r[101]])):
        cv2.line(frame,(r[98],r[99]),(r[100],r[101]),color,thickness)
    if (all(x > 0 for x in [r[100],r[101],r[124],r[125]])):
        cv2.line(frame,(r[100],r[101]),(r[124],r[125]),color,thickness)
    if (all(x > 0 for x in [r[98],r[99],r[126],r[127]])):
        cv2.line(frame,(r[98],r[99]),(r[126],r[127]),color,thickness)
    #left leg
    if (all(x > 0 for x in [r[78],r[79],r[102],r[103]])):
        cv2.line(frame,(r[78],r[79]),(r[102],r[103]),color,thickness)
    if (all(x > 0 for x in [r[102],r[103],r[104],r[105]])):
        cv2.line(frame,(r[102],r[103]),(r[104],r[105]),color,thickness)
    if (all(x > 0 for x in [r[104],r[105],r[106],r[107]])):
        cv2.line(frame,(r[104],r[105]),(r[106],r[107]),color,thickness)
    if (all(x > 0 for x in [r[106],r[107],r[108],r[109]])):
        cv2.line(frame,(r[106],r[107]),(r[108],r[109]),color,thickness)
    #right leg
    if (all(x > 0 for x in [r[78],r[79],r[110],r[111]])):
        cv2.line(frame,(r[78],r[79]),(r[110],r[111]),color,thickness)
    if (all(x > 0 for x in [r[110],r[111],r[112],r[113]])):
        cv2.line(frame,(r[110],r[111]),(r[112],r[113]),color,thickness)
    if (all(x > 0 for x in [r[112],r[113],r[114],r[115]])):
        cv2.line(frame,(r[112],r[113]),(r[114],r[115]),color,thickness)
    if (all(x > 0 for x in [r[114],r[115],r[116],r[117]])):
        cv2.line(frame,(r[114],r[115]),(r[116],r[117]),color,thickness)
    #body label
    #cv2.rectangle(frame,(r[128]-5,r[129]-5),(r[128]+5,r[129]+5),color,-1)
    #if (all(x > 0 for x in [r[137],r[138]])):
    #    cv2.putText(frame,str(i),(r[137]-6,r[138]-30),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    # Draw tracking confdence on joints:
    for joint in [(128,78,79),(129,80,81),(130,82,83),(131,84,85),(132,86,87),(133,88,89),(134,90,91),(135,92,93),(136,94,95),(137,96,97),(138,98,99),(139,100,101),(140,102,103),(141,104,105),(142,106,107),(143,108,109),(144,110,111),(145,112,113),(146,114,115),(147,116,117),(148,118,119),(149,120,121),(150,122,123),(151,124,125),(152,126,127)]:
        if (all(x > 0 for x in [r[joint[1]],r[joint[2]]])):
            if r[joint[0]]=='Tracked':
                #draw tracked joint
                cv2.circle(frame,(r[joint[1]],r[joint[2]]),4,(0,255,0),-1)
            elif r[joint[0]]=='Inferred':
                #draw inferred joint
                cv2.circle(frame,(r[joint[1]],r[joint[2]]),4,(0,0,255),-1)



#for joint in [(3,128,129),(8,131,132),(13,134,135),(18,137,138),(23,140,141),(28,143,144),(33,146,147),(38,149,150),(43,152,153),(48,155,156),(53,158,159),(58,161,162),(63,164,165),(68,167,168),(73,170,171),(78,173,174),(83,176,177),(88,179,180),(93,182,183),(98,185,186),(103,188,189),(108,191,192),(113,194,195),(118,197,198),(123,200,201)]:
#    print(joint[0])
'''           
# Create dictionary of # of bodies: 
skeldict={}
skiplist=[]
for skelfile in skelfiles:
    print('***')
    print(skelfile)
    try:
        # Find the corresponding clap time, start time, and stop time from the timing spreadsheet:
        timingrow=timingdf[(timingdf['kinectFilename']==os.path.splitext(os.path.basename(skelfile))[0]) & (timingdf['Camera']=='kinect')]
        clap=timingrow['claptime'].values[0]
        start=timingrow['starttime'].values[0]
        stop=timingrow['stoptime'].values[0]
        
        
        
        #Load skeleton tracking csv file:
        skeldf=pd.read_csv(skelfile,header=None,dtype=str)
        colist=[0,1,4,5,6,9,10,11,14,15,16,19,20,21,24,25,26,29,30,31,34,35,36,39,40,41,44,45,46,49,50,51,54,55,56,59,60,61,64,65,66,69,70,71,74,75,76,79,80,81,84,85,86,89,90,91,94,95,96,99,100,101,104,105,106,109,110,111,114,115,116,119,120,121,124,125,126,128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]
        for col in colist:
            skeldf[col]=pd.to_numeric(skeldf[col], errors='coerce')
            #skeldf[col]=skeldf[col].dropna().astype(float)
        # sync to clapper time
        # subtract (clap + start) from raw kinect times.
        skeldf['synctime']=skeldf[0].apply(lambda x:x-(clap+start))
        
        # select rows between start and end time
        skeldf=skeldf[skeldf['synctime'].between(0.0,stop)]
        
        
        
        # Rename the unique skeleton IDs to be simpler
        skels=sorted(skeldf[1].unique())
        for i,val in enumerate(skels):
            skeldf.loc[skeldf[1]==val,1]=i
        skelstat=bodystats(skeldf)
        skeldict[os.path.splitext(os.path.basename(skelfile))[0]]=len(skelstat)
    except:
        print('skipping',skelfile)
        skiplist.append(skelfile)
'''

# Create videos with superimposed skeletons: 
# 
# 
# 
#  
skiplist=[]
i=0
for skelfile in skelfiles:
    print(' ')
    print('***')
    runbase=os.path.splitext(os.path.basename(skelfile))[0][0:-11]
    print(runbase)
    #try:
    # find base rgb video        
    #if os.path.splitext(os.path.basename(skelfile))[0][5:-3] in ['1.3.3','1.3.4','1.3.9','1.3.1','2.3.10','6.2.1','6.2.7','6.2.5']:
    #    videofile=videofolder+os.path.splitext(os.path.basename(skelfile))[0][5:-3]+'_kinect_trim.mov'
    #else:
    videofile=videofolder+os.path.splitext(os.path.basename(skelfile))[0][0:-11]+'_kinect_trim.mp4'
    vidout=outdir+os.path.splitext(os.path.basename(skelfile))[0][0:-11]+'_kinect_skel.avi'
    if (os.path.isfile(vidout)):
        print('Already created',vidout)
    else:
        try:
            # Find the corresponding clap time, start time, and stop time from the timing spreadsheet:
            #timingrow=timingdf[(timingdf['kinectFilename']==os.path.splitext(os.path.basename(skelfile))[0]) & (timingdf['Camera']=='kinect')]
            #clap=timingrow['claptime'].values[0]
            #start=timingrow['starttime'].values[0]
            #stop=timingrow['stoptime'].values[0]
            
            
            
            #Load skeleton tracking csv file:
            #skeldf=pd.read_csv(skelfile,header=None,dtype=str)
            skeldf=pd.read_csv(skelfile)
            #columns that should be numeric:
            colist=[0,1,4,5,6,9,10,11,14,15,16,19,20,21,24,25,26,29,30,31,34,35,36,39,40,41,44,45,46,49,50,51,54,55,56,59,60,61,64,65,66,69,70,71,74,75,76,79,80,81,84,85,86,89,90,91,94,95,96,99,100,101,104,105,106,109,110,111,114,115,116,119,120,121,124,125,126,128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]
            #for col in colist:
            #    skeldf[col]=pd.to_numeric(skeldf[col], errors='coerce')
                #skeldf[col]=skeldf[col].dropna().astype(float)
            
            #columns that should be tracked/inferred labels
            #tracklist=[3,8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98,103,108,113,118,123]
            # sync to clapper time
            # subtract (clap + start) from raw kinect times.
            #skeldf['synctime']=skeldf[0].apply(lambda x:x-(clap+start))
            
            # select rows between start and end time
            #skeldf=skeldf[skeldf['synctime'].between(0.0,stop)]
            
            
            
            # Rename the unique skeleton IDs to be simpler
            #skels=sorted(skeldf[1].unique())
            #for i,val in enumerate(skels):
            #    skeldf.loc[skeldf[1]==val,1]=i
                
                
                
            #skelstat=bodystats(skeldf)
            #skeldict[os.path.splitext(os.path.basename(skelfile))[0]]=len(skelstat)
            
            # Create separate df for each body
            #splitdf=bodysplit(skeldf)
            
            #Resample to 25 fps for normal mp4 files, 29.970030 fps for the mov files
            #if os.path.splitext(os.path.basename(skelfile))[0][5:-3] in ['1.3.3','1.3.4','1.3.9','1.3.1','2.3.10','6.2.1','6.2.7','6.2.5']:
            #    splitdf=[skelresample(i,rate='33366667ns') for i in splitdf]
            #else:
            #    splitdf=[skelresample(i,rate='40ms') for i in splitdf]
            # Create videos of 2d skeletons superimposed on RGB video:
            #if os.path.splitext(os.path.basename(skelfile))[0][0:-11] in ['1.3.3','1.3.4','1.3.9','1.3.1','2.3.10','6.2.1','6.2.7','6.2.5']:
            #    skeldf=skelresample(skeldf,rate='33366667ns')
            #else:
            #    skeldf=skelresample(skeldf,rate='40ms')
            #skeldf=skelresample(skeldf,rate='40ms')
            '''
            if small:
                #skeldf=skelresample(skeldf,rate='33333333ns')
                skeldf=skelresample(skeldf,rate='33366667ns')
                #skeldf=skelresample(skeldf,rate='40ms')
            else:
                skeldf=skelresample(skeldf,rate='16683333ns')
            '''
            #correct for asynchrony:
            #lag=skel_corr_df['displacement'][skel_corr_df['video']==runbase]
            #skeldf['sync_time']=skeldf['sync_time']+lag
        
        
            
            video_capture = cv2.VideoCapture()
            #print(skelfile)
            f=0
            skeli=0
            #skelrows=[0]*len(skeldf)
            if video_capture.open(videofile):
                width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))    
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                frate=str(round(1000000000/fps))+'ns'
                skeldf=skelresample(skeldf,rate=frate)
                totalFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(vidout, fourcc, fps,(width,height), True)
                #video_writer = cv2.VideoWriter(vidout, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
                #while video_capture.isOpened():
                #for k in trange(9000):    
                for k in trange(totalFrames):           
                    #print(f,width,height,fps)
                    ret, frame = video_capture.read()
                    if not ret:
                        print('not returned')
                        break
                    s=f/fps
                    #print(s)
                    if skeli < len(skeldf):
                        if s >= skeldf['sync_time'].iloc[skeli]:
                            #draw skeleton on frame
                            drawskel(skeldf.iloc[skeli],frame,i,colors[i])
                            skeli+=1
                    video_writer.write(frame)
                    #if f > 400:
                    #    break
                    f+=1
                video_capture.release()
                video_writer.release()
        except:
            print('ERROR')
    '''
    except:
        print('skipping',skelfile)
        skiplist.append(skelfile)
        '''

#lendict={}
#for k in skeldict.keys():
#    lendict[k[5:-3]]=skeldict[k]


'''
# Write bodies per run to a csv file:

import csv
with open('bodies_per_run.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, lendict.keys())
    w.writeheader()
    w.writerow(lendict)
'''

'''
## Histogram of bodies per run:
plt.figure()
plt.hist(skeldict.values())
plt.xlabel('Number of skeletons')
plt.ylabel('Number of runs')
plt.title('Histogram of skeletons per run')
plt.tight_layout()
plt.show()
plt.savefig('skeleton_histogram.png')

##mean number of bodies per run:
np.mean([int(i) for i in skeldict.values()])
'''
#splitdf=bodysplit(skeldf)
#kdf=splitdf[1]

'''
# Plot inter-frame interval for all skeletons:
for j in splitdf:
    plt.plot(j.index[1:],np.diff(j[0]),markersize=1)
#max(set(np.diff(kdf[0])))
plt.show()
'''

'''
testdf=splitdf[0]
testdf[188]=testdf[188].astype(float)
testdf[189]=testdf[189].astype(float)
testdf[134]=testdf[134].astype(float)
testdf[135]=testdf[135].astype(float)
testdf = testdf.set_index(pd.to_datetime(testdf['synctime'], unit='s'), drop=False)

resample_index = pd.date_range(start=testdf.index[0], end=testdf.index[-1], freq='40ms')
dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
testdf.combine_first(dummy_frame).interpolate('time').iloc[:6]
testoutdf=testdf.combine_first(dummy_frame).interpolate('time').resample('40ms').mean()


# Plot example to check resampling

plt.figure()
plt.scatter(testdf['synctime'].iloc[:60],testdf[201].iloc[:60],alpha=.5,label='30 fps')
plt.scatter(testoutdf['synctime'].iloc[:50],testoutdf[201].iloc[:50],alpha=.5,label='25 fps')

plt.xlabel('Seconds')
plt.ylabel('Right Thumb Y Position')
plt.title('Resampling Example: Right Thumb Y Position')
plt.legend()
plt.tight_layout()
plt.show()

'''



        
        
