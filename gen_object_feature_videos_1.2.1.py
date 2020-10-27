#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create object movement videos

Created on Sun Sep 20 12:45:45 2020

@author: bezdek
"""
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
#from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent
#from sem.event_models import RecurrentEvent, GRUEvent, LSTMEvent
#from sem import sem_run
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.animation as animation
#from IPython.display import HTML

featuresin='/Users/bezdek/Documents/tracking/scripts/1.2.1_features.csv'
featuresout='/Users/bezdek/Documents/tracking/scripts/1.2.1_feature_test.mp4'
corrout='/Users/bezdek/Documents/tracking/scripts/1.2.1_feature_corrs.png'

fdf=pd.read_csv(featuresin)
for col in fdf.columns:
    if col != 'sync_time':
        fdf[col]=zscore(fdf[col])
 

fig=plt.figure()
for col in fdf.columns:
    if col != 'sync_time':
        plt.scatter(fdf['sync_time'],fdf[col])
plt.legend()
plt.show()       
fdfcorr=fdf.drop(['sync_time'],axis=1).corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(fdfcorr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(fdfcorr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title('1.2.1 Correlations between features')
plt.savefig(corrout)
plt.close()        
        
        
'''
colors = dict(zip(
    ["India", "Europe", "Asia", "Latin America", "Middle East", "North America", "Africa"],
    ["#adb0ff", "#ffb3ff", "#90d595", "#e48381", "#aafbff", "#f7bb5f", "#eafb50"]
))
group_lk = df.set_index('name')['group'].to_dict()
'''

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=1800)

valcols=[i for i in fdf.columns if i not in ['sync_time']]
fdflong=pd.melt(fdf,id_vars=['sync_time'],value_vars=valcols)
fdflong=fdflong.sort_values('sync_time')

times=fdflong['sync_time'].unique()
current_time=times[5]
NUM_COLORS = len(fdflong['variable'].unique())

cm = plt.get_cmap('gist_rainbow')
colors=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

fig, ax = plt.subplots(figsize=(15, 8))

def draw_barchart(current_time):
    fdff = fdflong[fdflong['sync_time'].eq(current_time)].sort_values('variable')
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
    plt.xlim([min(fdflong['value']),max(fdflong['value'])])
    #ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    #ax.tick_params(axis='x', colors='#777777', labelsize=12)
    #ax.set_yticks([])
    #ax.margins(0, 0.01)
    #ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, 'Object Movement',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    #plt.box(False)
    
#draw_barchart(times[10])

fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_barchart, frames=times)
#HTML(animator.to_jshtml())
animator.save(featuresout, writer=writer)
# or use animator.to_html5_video() or animator.save()