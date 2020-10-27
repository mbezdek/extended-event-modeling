#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:29:28 2020

@author: bezdek
"""
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
kinecttrackingfiles=glob('/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/*kinect_merged.csv')
kinecttrackingfiles=['/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.1_kinect_merged.csv']
outdir='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/object_movement'
skelfile="/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_data/2.2.1_skel_clean.csv"
if not os.path.exists(outdir):
    os.makedirs(outdir)
import math  
def calculateDistance(x1,y1,x2,y2):
    if (x1,y1,x2,y2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return distance
def calc_center(df):
    # Input: a dataframe with x,y,w, & h columns.
    # Output: dataframe with added columns for x and y center of each box.
    df['x_cent']=df['x']+(df['w']/2.0)
    df['y_cent']=df['y']+(df['h']/2.0)
    return df

#with smoothing:
for kinecttrackingfile in kinecttrackingfiles:
    try:
        print(os.path.split(kinecttrackingfile)[1][0:-18])
        C1trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C1_merged.csv'
        C2trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C2_merged.csv'      
        #C1trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C1_merged.csv'
        #C2trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C2_merged.csv'
        #kinecttrackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_kinect_merged.csv'
        run=os.path.splitext(os.path.basename(C1trackingfile))[0][0:-10]
        if os.path.exists(outdir+'/'+run):
            break
        else:
            os.makedirs(outdir+'/'+run)
            tdfC1=pd.read_csv(C1trackingfile)
            tdfC2=pd.read_csv(C2trackingfile)
            tdfkinect=pd.read_csv(kinecttrackingfile)
            tdfkinect['x']=960-tdfkinect['x']
            tdfkinect['w']=(-1)*tdfkinect['w']
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
            tdfC1['sec'] = tdfC1['frame']/30.0
            tdfC2['sec'] = tdfC2['frame']/30.0
            tdfkinect['sec'] = tdfkinect['frame']/25.0
            
            objects=tdfC1['name'].unique()
            maxtime=max(tdfC1['sec'].iloc[-1],tdfC2['sec'].iloc[-1],tdfkinect['sec'].iloc[-1])
            
            '''
            # Make plots of labeled object movement, nine per figure.
            k=0
            #while k < len(objects):
            while k < 9:
                j=0
                fig=plt.figure()
                while j < 9:
                    fig.add_subplot(3,3,j+1) 
                    plt.xlim(0,960)
                    plt.ylim(546,0)
                    #n=len(tdfC1[tdfC1['name']==objects[k]]['x_cent'])
                    #C = [i*(255/(n-1)) for i in range(n)]
                    plt.scatter(tdfC1[tdfC1['name']==objects[k]]['x_cent'],tdfC1[tdfC1['name']==objects[k]]['y_cent'],c = (tdfC1[tdfC1['name']==objects[k]]['sec']/maxtime)*255.0)
                    #plt.scatter(tdfC2[tdfC2['name']==objects[k]]['x_cent'],tdfC2[tdfC2['name']==objects[k]]['y_cent'],c = (tdfC2[tdfC2['name']==objects[k]]['frame']/tdfC2['frame'].max())*255.0)
                    #plt.scatter(tdfkinect[tdfkinect['name']==objects[k]]['x_cent'],tdfkinect[tdfkinect['name']==objects[k]]['y_cent'],c = (tdfkinect[tdfkinect['name']==objects[k]]['frame']/tdfkinect['frame'].max())*255.0)
                    plt.title(objects[k]+" C1")
                    j+=1
                    fig.add_subplot(3,3,j+1) 
                    plt.xlim(0,960)
                    plt.ylim(546,0)
                    #n=len(tdfC1[tdfC1['name']==objects[k]]['x_cent'])
                    #C = [i*(255/(n-1)) for i in range(n)]
                    #plt.scatter(tdfC1[tdfC1['name']==objects[k]]['x_cent'],tdfC1[tdfC1['name']==objects[k]]['y_cent'],c = (tdfC1[tdfC1['name']==objects[k]]['frame']/tdfC1['frame'].max())*255.0)
                    plt.scatter(tdfC2[tdfC2['name']==objects[k]]['x_cent'],tdfC2[tdfC2['name']==objects[k]]['y_cent'],c = (tdfC2[tdfC2['name']==objects[k]]['sec']/maxtime)*255.0)
                    #plt.scatter(tdfkinect[tdfkinect['name']==objects[k]]['x_cent'],tdfkinect[tdfkinect['name']==objects[k]]['y_cent'],c = (tdfkinect[tdfkinect['name']==objects[k]]['frame']/tdfkinect['frame'].max())*255.0)
                    plt.title(objects[k]+" C2")
                    j+=1
                    fig.add_subplot(3,3,j+1) 
                    plt.xlim(0,960)
                    plt.ylim(546,0)
                    #n=len(tdfC1[tdfC1['name']==objects[k]]['x_cent'])
                    #C = [i*(255/(n-1)) for i in range(n)]
                    #plt.scatter(tdfC1[tdfC1['name']==objects[k]]['x_cent'],tdfC1[tdfC1['name']==objects[k]]['y_cent'],c = (tdfC1[tdfC1['name']==objects[k]]['frame']/tdfC1['frame'].max())*255.0)
                    #plt.scatter(tdfC2[tdfC2['name']==objects[k]]['x_cent'],tdfC2[tdfC2['name']==objects[k]]['y_cent'],c = (tdfC2[tdfC2['name']==objects[k]]['frame']/tdfC2['frame'].max())*255.0)
                    plt.scatter(tdfkinect[tdfkinect['name']==objects[k]]['x_cent'],tdfkinect[tdfkinect['name']==objects[k]]['y_cent'],c = (tdfkinect[tdfkinect['name']==objects[k]]['sec']/maxtime)*255.0)
                    plt.title(objects[k]+" kinect")
                    j+=1        
                    k+=1
                plt.tight_layout()
                plt.savefig('/Users/bezdek/Documents/traj_'+str(k)+'.png')
                plt.show()
            ''' 
            
             
            l=0
            while l < len(objects):
                obj=objects[l]
                testdf=tdfC1[tdfC1['name']==obj]
                fig=plt.figure()
                fig.add_subplot(3,1,1) 
                m=1
                dist=[]
                frames=[]
                
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist)
                
                ##
                xx = np.linspace(min(frames),max(frames), 25000)
                # interpolate + smooth
                itp = interp1d(frames,dist, kind='linear')
                window_size, poly_order = 501, 3
                yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                ##
                plt.plot(xx, yy_sg, 'k', label= "Smoothed curve")
                plt.xlabel('time (in s)')
                plt.xlim(0,maxtime)
                plt.ylabel('C1 \nmovement') 
                #plt.suptitle('dumbbell_1 travel C1')  
                #plt.savefig('/Users/bezdek/Documents/'+obj+'_travel_C1.png')
                
                testdf=tdfC2[tdfC2['name']==obj]
                fig.add_subplot(3,1,2) 
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist)
                xx = np.linspace(min(frames),max(frames), 25000)
                # interpolate + smooth
                itp = interp1d(frames,dist, kind='linear')
                window_size, poly_order = 501, 3
                yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                ##
                plt.plot(xx, yy_sg, 'k', label= "Smoothed curve")
                plt.xlabel('time (in s)')   
                plt.xlim(0,maxtime)
                plt.ylabel('C2 \nmovement') 
                #plt.suptitle('dumbbell_1 travel C2')  
                #plt.savefig('/Users/bezdek/Documents/'+obj+'_travel_C2.png')
                
                testdf=tdfkinect[tdfkinect['name']==obj]
                fig.add_subplot(3,1,3) 
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist) 
                xx = np.linspace(min(frames),max(frames), 25000)
                # interpolate + smooth
                itp = interp1d(frames,dist, kind='linear')
                window_size, poly_order = 501, 3
                yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                ##
                plt.plot(xx, yy_sg, 'k', label= "Smoothed curve")
                plt.xlabel('time (in s)')  
                plt.xlim(0,maxtime)
                plt.ylabel('Kinect \nmovement') 
                plt.suptitle(obj+' inter-frame movement')  
                #plt.tight_layout()
                plt.savefig(outdir+'/'+run+'/'+obj+'_movement.png')
                plt.close()
                l+=1
    except:
        print('ERROR',kinecttrackingfile)

for kinecttrackingfile in kinecttrackingfiles:
    try:
        print(os.path.split(kinecttrackingfile)[1][0:-18])
        C1trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C1_merged.csv'
        C2trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C2_merged.csv'      
        #C1trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C1_merged.csv'
        #C2trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C2_merged.csv'
        #kinecttrackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_kinect_merged.csv'
        run=os.path.splitext(os.path.basename(C1trackingfile))[0][0:-10]
        if os.path.exists(outdir+'/'+run):
            break
        else:
            os.makedirs(outdir+'/'+run)
            tdfC1=pd.read_csv(C1trackingfile)
            tdfC2=pd.read_csv(C2trackingfile)
            tdfkinect=pd.read_csv(kinecttrackingfile)
            tdfkinect['x']=960-tdfkinect['x']
            tdfkinect['w']=(-1)*tdfkinect['w']
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
            tdfC1['sec'] = tdfC1['frame']/30.0
            tdfC2['sec'] = tdfC2['frame']/30.0
            tdfkinect['sec'] = tdfkinect['frame']/25.0
            
            objects=tdfC1['name'].unique()
            maxtime=max(tdfC1['sec'].iloc[-1],tdfC2['sec'].iloc[-1],tdfkinect['sec'].iloc[-1])        
            l=0
            while l < len(objects):
                obj=objects[l]
                testdf=tdfC1[tdfC1['name']==obj]
                fig=plt.figure()
                fig.add_subplot(3,1,1) 
                m=1
                dist=[]
                frames=[]
                
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist)
             
                plt.xlabel('time (in s)')
                plt.xlim(0,maxtime)
                plt.ylabel('C1 \nmovement') 
                #plt.suptitle('dumbbell_1 travel C1')  
                #plt.savefig('/Users/bezdek/Documents/'+obj+'_travel_C1.png')
                
                testdf=tdfC2[tdfC2['name']==obj]
                fig.add_subplot(3,1,2) 
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist)
                plt.xlabel('time (in s)')   
                plt.xlim(0,maxtime)
                plt.ylabel('C2 \nmovement') 
                #plt.suptitle('dumbbell_1 travel C2')  
                #plt.savefig('/Users/bezdek/Documents/'+obj+'_travel_C2.png')
                
                testdf=tdfkinect[tdfkinect['name']==obj]
                fig.add_subplot(3,1,3) 
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    dist.append(calculateDistance(oldx,oldy,newx,newy))
                    frames.append(testdf['sec'].iloc[m])
                    m+=1
                plt.scatter(frames,dist) 
                plt.xlabel('time (in s)')  
                plt.xlim(0,maxtime)
                plt.ylabel('Kinect \nmovement') 
                plt.suptitle(obj+' inter-frame movement')  
                #plt.tight_layout()
                plt.savefig(outdir+'/'+run+'/'+obj+'_movement.png')
                plt.close()
                l+=1
                
for kinecttrackingfile in kinecttrackingfiles:
    try:
        print(os.path.split(kinecttrackingfile)[1][0:-18])
        C1trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C1_merged.csv'
        C2trackingfile=os.path.split(kinecttrackingfile)[0]+'/'+os.path.split(kinecttrackingfile)[1][0:-17]+'C2_merged.csv'      
        #C1trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C1_merged.csv'
        #C2trackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_C2_merged.csv'
        #kinecttrackingfile='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/from_chpc/merged/2.2.6_kinect_merged.csv'
        run=os.path.splitext(os.path.basename(C1trackingfile))[0][0:-10]
        if os.path.isfile(outdir+'/'+run+'_object_spikes_all.png'):
            pass
        else:
            #os.makedirs(outdir+'/'+run)
            tdfC1=pd.read_csv(C1trackingfile)
            tdfC2=pd.read_csv(C2trackingfile)
            tdfkinect=pd.read_csv(kinecttrackingfile)
            tdfkinect['x']=960-tdfkinect['x']
            tdfkinect['w']=(-1)*tdfkinect['w']
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
            tdfC1['sec'] = tdfC1['frame']/30.0
            tdfC2['sec'] = tdfC2['frame']/30.0
            tdfkinect['sec'] = tdfkinect['frame']/25.0
            
            objects=tdfC1['name'].unique()
            maxtime=max(tdfC1['sec'].iloc[-1],tdfC2['sec'].iloc[-1],tdfkinect['sec'].iloc[-1])        
            l=0
            object_spikes=pd.DataFrame(columns=['camera','name','sec','dist'])
            thresh=20.0
            while l < len(objects):
                obj=objects[l]
                testdf=tdfC1[tdfC1['name']==obj]
                #fig=plt.figure()
                #fig.add_subplot(3,1,1) 
                m=1
                dist=[]
                frames=[]
                
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    newdist=calculateDistance(oldx,oldy,newx,newy)
                    dist.append(newdist)
                    frames.append(testdf['sec'].iloc[m])
                    if newdist > thresh:
                        object_spikes=object_spikes.append({'camera':'C1','name':obj,'sec':testdf['sec'].iloc[m],'dist':newdist},ignore_index=True)
                    m+=1
                testdf=tdfC2[tdfC2['name']==obj]
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    newdist=calculateDistance(oldx,oldy,newx,newy)
                    dist.append(newdist)
                    frames.append(testdf['sec'].iloc[m])
                    if newdist > thresh:
                        object_spikes=object_spikes.append({'camera':'C2','name':obj,'sec':testdf['sec'].iloc[m],'dist':newdist},ignore_index=True)
                    m+=1
              
                testdf=tdfkinect[tdfkinect['name']==obj]
                m=1
                dist=[]
                frames=[]
                while m < len(testdf):
                    oldx=testdf['x_cent'].iloc[m-1]
                    oldy=testdf['y_cent'].iloc[m-1]
                    newx=testdf['x_cent'].iloc[m]
                    newy=testdf['y_cent'].iloc[m]
                    newdist=calculateDistance(oldx,oldy,newx,newy)
                    dist.append(newdist)
                    frames.append(testdf['sec'].iloc[m])
                    if newdist > thresh:
                        object_spikes=object_spikes.append({'camera':'kinect','name':obj,'sec':testdf['sec'].iloc[m],'dist':newdist},ignore_index=True)

                    m+=1
                l+=1
             
            fig=plt.figure()
            #fig.add_subplot(3,1,1) 
            #testdf=object_spikes[object_spikes['camera']=='C1']
            fg = sns.FacetGrid(data=object_spikes, row='camera',hue='name', aspect=1.61)
            fg.map(plt.scatter, 'sec', 'dist',alpha=.5).add_legend()
            #fig.add_subplot(3,1,2) 
            #testdf=object_spikes[object_spikes['camera']=='C2']
            #fg = sns.FacetGrid(data=testdf, hue='name', aspect=1.61)
            #fg.map(plt.scatter, 'sec', 'dist').add_legend()
            #fig.add_subplot(3,1,3) 
            #testdf=object_spikes[object_spikes['camera']=='kinect']
            #fg = sns.FacetGrid(data=testdf, hue='name', aspect=1.61)
            #fg.map(plt.scatter, 'sec', 'dist').add_legend()
            plt.suptitle(run) 
            plt.savefig(outdir+'/'+run+'_object_spikes_all.png')
            plt.close()
    except:
        print('error',kinecttrackingfile)
  
skeldf=pd.read_csv(skelfile)    
      
# sync_time, Right Hand: J11_2D_X, J11_2D_Y
handdf=skeldf[['sync_time', 'J11_2D_X', 'J11_2D_Y']]
fig=plt.figure()
plt.gca().invert_yaxis()
plt.scatter(handdf['J11_2D_X']/2,handdf['J11_2D_Y']/2,c=handdf['sync_time'])
plt.colorbar()
plt.show()

belldf=tdfkinect[tdfkinect['name']=='dumbbell_1']
belldf['x_cent1']=belldf['x_cent'].shift(1)
belldf['y_cent1']=belldf['y_cent'].shift(1)
belldf['move']=belldf[['x_cent1','y_cent1','x_cent','y_cent']].apply(lambda x: calculateDistance(x[0],x[1],x[2],x[3]) if(np.all(pd.notnull(x))) else np.nan, axis = 1)


fig=plt.figure()
plt.gca().invert_yaxis()
plt.scatter(belldf['x_cent'],belldf['y_cent'],c=belldf['frame']/25)
plt.colorbar()
plt.show()

belldf['sync_time']=belldf['frame']/25
testdf=pd.concat([handdf,belldf],ignore_index=True,sort=False)
testdf=testdf.sort_values(by=['sync_time'])
testdf['x_cent']=testdf['x_cent'].interpolate(method='linear')
testdf['y_cent']=testdf['y_cent'].interpolate(method='linear')
testdf['J11_2D_X']=testdf['J11_2D_X'].interpolate(method='linear')
testdf['J11_2D_Y']=testdf['J11_2D_Y'].interpolate(method='linear')
#testdf=testdf.set_index('sync_time')
testdf['J11_2D_X']=testdf['J11_2D_X']/2
testdf['J11_2D_Y']=testdf['J11_2D_Y']/2
testdf=testdf[testdf['sync_time'] < 575.2]

fig=plt.figure()
ax=fig.add_subplot(2,1,1)
# Defining custom 'xlim' and 'ylim' values.
custom_xlim = ([0.0,960.0])
custom_ylim = ([0.0,540.0])

# Setting the values for all axes.
plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

#plt.xlim=([0.0,960.0])
#plt.ylim=([0.0,540.0])
plt.gca().invert_yaxis()
plt.scatter(testdf['J11_2D_X']/2,testdf['J11_2D_Y']/2,c=testdf.index)

plt.colorbar()
ax2=fig.add_subplot(2,1,2)
plt.setp(ax2, xlim=custom_xlim, ylim=custom_ylim)
#plt.xlim=([0.0,960.0])
#plt.ylim=([0.0,540.0])
plt.gca().invert_yaxis()
plt.scatter(testdf['x_cent'],testdf['y_cent'],c=testdf.index)
plt.colorbar()
plt.show()

fig=plt.figure()
i=0
#while i < len(testdf):
while i < 1000:
    plt.plot([testdf['J11_2D_X'].iloc[i]/2+i,testdf['x_cent'].iloc[i]+i],[testdf['J11_2D_Y'].iloc[i]/2,testdf['y_cent'].iloc[i]],marker='o')
    i+=1
#testdf['h2odist']=calculateDistance(testdf['J11_2D_X']/2,testdf['J11_2D_Y']/2,testdf['x_cent'],testdf['y_cent'])
testdf['h2odist']=testdf[['J11_2D_X','J11_2D_Y','x_cent','y_cent']].apply(lambda x: calculateDistance(x[0],x[1],x[2],x[3]) if(np.all(pd.notnull(x))) else np.nan, axis = 1)

fig=plt.figure()
plt.plot(testdf['h2odist'])
plt.plot(testdf[['x_cent','y_cent']])
plt.legend()
plt.show()


# Smooth with Gaussian kernel. Sigma = 3
belldf['moveg'] = gaussian_filter1d(belldf['move'], 3)
fig=plt.figure()

plt.plot(belldf['sec'],belldf['moveg'])
plt.legend()
plt.show()

#movedf=tdfkinect
objs=tdfkinect['name'].unique()
#movedf=movedf.set_index('sec',drop=False)
movedf=pd.DataFrame()
for obj in objs:
    testdf=tdfkinect[tdfkinect['name']==obj]
    testdf['x_cent1']=testdf['x_cent'].shift(1)
    testdf['y_cent1']=testdf['y_cent'].shift(1)
    testdf[obj+'_move']=testdf[['x_cent1','y_cent1','x_cent','y_cent']].apply(lambda x: calculateDistance(x[0],x[1],x[2],x[3]) if(np.all(pd.notnull(x))) else np.nan, axis = 1)
    testdf=testdf.set_index('sec',drop=False)
    movedf[obj+'_move']=testdf[obj+'_move']

movedf['sync_time']=movedf.index
'''
fig=plt.figure()
for obj in objs:
    plt.plot(movedf['sec'],movedf[obj+'_move'])
plt.legend()
plt.show()
'''

# Apply Gaussian filter to object movement:
for obj in objs:
    movedf[obj+'_moveg']= gaussian_filter1d(movedf[obj+'_move'], 3)
    movedf=movedf.drop([obj+'_move'], axis=1)

''' 
fig=plt.figure()
for obj in objs:
    plt.scatter(movedf['sec'],movedf[obj+'_moveg'])
plt.legend()
plt.show()
'''
# 0.08 time step. resample to .33
#Resampling function:
def skelresample(testdf,rate='40ms'):
    #tracklist=[2,3,7,8,12,13,17,18,22,23,27,28,32,33,37,38,42,43,47,48,52,53,57,58,62,63,67,68,72,73,77,78,82,83,87,88,92,93,98,103,108,113,118,123]
    #tracklist=[i for i in testdf.columns if i[-5:]=='moveg']
    tracklist=[i for i in testdf.columns]
    #testdf=df.apply(pd.to_numeric, errors='coerce')
    testdf = testdf.set_index(pd.to_datetime(testdf['sync_time'], unit='s'), drop=False)
    resample_index = pd.date_range(start=testdf['sync_time'].iloc[0], end=testdf['sync_time'].iloc[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
    #df.combine_first(dummy_frame).interpolate('time').iloc[:6]
    outdf=testdf.combine_first(dummy_frame).interpolate('time').resample(rate).last().fillna(0)
    #for col in tracklist:
    #    outdf[col]=testdf[col].interpolate('time').resample(rate).nearest()
    #for col in [128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]:
            #outdf[col]=outdf[col].dropna().astype(np.int64)
    #        outdf[col]=outdf[col].fillna(value=-1).astype(int)
    #outdf[3]=testdf[3].interpolate('time').resample(rate).nearest()
    #outdf = outdf.reindex(columns=sorted(outdf.columns))
    return outdf

def objresample(testdf,rate='40ms'):
    #tracklist=[2,3,7,8,12,13,17,18,22,23,27,28,32,33,37,38,42,43,47,48,52,53,57,58,62,63,67,68,72,73,77,78,82,83,87,88,92,93,98,103,108,113,118,123]
    #tracklist=[i for i in testdf.columns if i[-5:]=='moveg']
    tracklist=[i for i in testdf.columns]
    #testdf=df.apply(pd.to_numeric, errors='coerce')
    testdf = testdf.set_index(pd.to_datetime(testdf['sync_time'], unit='s'), drop=False)
    resample_index = pd.date_range(start=testdf.index[0], end=testdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
    #df.combine_first(dummy_frame).interpolate('time').iloc[:6]
    outdf=testdf.combine_first(dummy_frame).interpolate('time').resample(rate).last().fillna(0)
    #for col in tracklist:
    #    outdf[col]=testdf[col].interpolate('time').resample(rate).nearest()
    #for col in [128,129,131,132,134,135,137,138,140,141,143,144,146,147,149,150,152,153,155,156,158,159,161,162,164,165,167,168,170,171,173,174,176,177,179,180,182,183,185,186,188,189,191,192,194,195,197,198,200,201]:
            #outdf[col]=outdf[col].dropna().astype(np.int64)
    #        outdf[col]=outdf[col].fillna(value=-1).astype(int)
    #outdf[3]=testdf[3].interpolate('time').resample(rate).nearest()
    #outdf = outdf.reindex(columns=sorted(outdf.columns))
    return outdf
resampledf=objresample(movedf,rate='333ms')


fig=plt.figure()
for obj in objs:
    plt.scatter(resampledf['sync_time'],resampledf[obj+'_moveg'])
plt.legend()
plt.show()

# Problem: linear interpolation across long gaps in object positions.
# possible solution: log start/stop times for objects. Make all other times 0 or NA


# Calc limb distance from center
# calc limb velocity
# calc limb acceleration

def calc_limb_dist(df):
    # key joints: spine_mid:J1,left hand:J7,right hand: J11, foot left J15, foot right: J19
    # compute distances between hands/feet and spine mid:
    df['left_hand_dist'] = np.sqrt( (df.J7_3D_X-df.J1_3D_X)**2 + (df.J7_3D_Y-df.J1_3D_Y)**2 + (df.J7_3D_Z-df.J1_3D_Z)**2)
    df['right_hand_dist'] = np.sqrt( (df.J11_3D_X-df.J1_3D_X)**2 + (df.J11_3D_Y-df.J1_3D_Y)**2 + (df.J11_3D_Z-df.J1_3D_Z)**2)
    df['left_foot_dist'] = np.sqrt( (df.J15_3D_X-df.J1_3D_X)**2 + (df.J15_3D_Y-df.J1_3D_Y)**2 + (df.J15_3D_Z-df.J1_3D_Z)**2)
    df['right_foot_dist'] = np.sqrt( (df.J19_3D_X-df.J1_3D_X)**2 + (df.J19_3D_Y-df.J1_3D_Y)**2 + (df.J19_3D_Z-df.J1_3D_Z)**2)   
    df['right_hand_move'] = np.sqrt( (df.J11_3D_X-df.J11_3D_X.shift(1))**2 + (df.J11_3D_Y-df.J11_3D_Y.shift(1))**2 + (df.J11_3D_Z-df.J11_3D_Z.shift(1))**2)
    
    return df
skeldf=calc_limb_dist(skeldf)

fig=plt.figure()
plt.plot(skeldf['left_hand_dist'])
plt.plot(skeldf['right_hand_dist'])
plt.plot(skeldf['right_hand_move'])
plt.plot(skeldf['left_foot_dist'])
plt.plot(skeldf['right_foot_dist'])
plt.legend()
plt.show()
skelresampledf=skelresample(skeldf,rate='333ms')

#combine obj and skel dfs:
sdf=skelresampledf[['right_hand_dist','right_hand_move','sync_time']].set_index('sync_time',drop=False)
odf=resampledf.set_index('sync_time',drop=False)
#odf['sync_time']=odf['sec']
alldf=pd.concat([sdf,odf],ignore_index=True,sort=False)
alldf=alldf.sort_values(by=['sync_time'])
for col in alldf.columns:
    alldf[col]=alldf[col].interpolate(method='linear')
alldf=skelresample(alldf,rate='333ms')    

fig=plt.figure()
for col in alldf.columns:
    if col != 'sync_time':
        plt.scatter(alldf['sync_time'],alldf[col])
plt.legend()
plt.show()
    
alldf.to_csv('/Users/bezdek/Documents/tracking/scripts/2.2.1_features.csv',index=False)
testdf['J11_2D_X']=testdf['J11_2D_X']/2
testdf['J11_2D_Y']=testdf['J11_2D_Y']/2
testdf=testdf[testdf['sync_time'] < 575.2]