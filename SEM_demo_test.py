#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:36:14 2020

@author: bezdek & code from Nick Franklin
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent
from sem.event_models import RecurrentEvent, GRUEvent, LSTMEvent
from sem import sem_run
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


segpath='/Users/bezdek/Box/DCL_ARCHIVE/Documents/Events/exp148_Corpus/segmentation/Analysis/allData.csv'
grain='coarse'
movie='2.2.1_C1_trim'
'''
def build_2d_data(N, beta=0.1):
    """ 
    2d, Mulitvariate Guassian data
    """
    mu = [1, 1]
    Sigma = np.eye(2) * beta
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        x[n, :] = np.random.multivariate_normal(mu, Sigma)
        
    return x
X = build_2d_data(100)
plt.figure(figsize=(3,3))
plt.scatter(X[:,0], X[:, 1])

D = 2
X_train = X[0:-1]; y_train = X[1:]

def build_static_dataset(N, beta=0.1):
    """ 
    2 events, modeled as gaussians with different means
    """
    
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(int(N/2)):
        x[n, :] = np.random.multivariate_normal(mus[0], np.eye(2) * beta)
    for n in range(int(N/2), N):
        x[n, :] = np.random.multivariate_normal(mus[1], np.eye(2) * beta)
        
    return x
def build_static_dataset(N, beta=0.1):
    """ 
    2 events, modeled as gaussians with different means
    """
    
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1], [1, 4],[-1, -4],[3, 3],[-3, -3],[5, 5]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(30):
        x[n, :] = np.random.multivariate_normal(mus[0], np.eye(2) * beta)
    for n in range(30,60):
        x[n, :] = np.random.multivariate_normal(mus[1], np.eye(2) * beta)
    for n in range(60,90):
        x[n, :] = np.random.multivariate_normal(mus[2], np.eye(2) * beta)
    for n in range(90,120):
        x[n, :] = np.random.multivariate_normal(mus[3], np.eye(2) * beta)
    for n in range(120,150):
        x[n, :] = np.random.multivariate_normal(mus[4], np.eye(2) * beta)
    for n in range(150,180):
        x[n, :] = np.random.multivariate_normal(mus[5], np.eye(2) * beta)
    for n in range(180,210):
        x[n, :] = np.random.multivariate_normal(mus[6], np.eye(2) * beta)
    return x
x_train = build_static_dataset(210, 0.01)
y = np.concatenate([np.zeros((30), dtype=int), np.ones((30), dtype=int), 2 * np.ones((30), dtype=int), 3 * np.ones((30), dtype=int), 4 * np.ones((30), dtype=int), 5 * np.ones((30), dtype=int), 6 * np.ones((30), dtype=int)])
plt.plot(x_train[:, 0], x_train[:, 1])
'''
# SEM parameters
lmda = 10.0  # stickyness parameter
alfa = .01  # concentration parameter
#lmda = 20.0  # stickyness parameter
#alfa = 2.0  # concentration parameter

# define plotting function


def plot_segmentation(post, y):
    cluster_id = np.argmax(post, axis=1)
    cc = sns.color_palette('Dark2', post.shape[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2]))
    for clt in cluster_id:
        idx = np.nonzero(cluster_id == clt)[0]
        axes[0].scatter(x_train[idx, 0], x_train[idx, 1], color=cc[clt], alpha=.5)
    axes[0].set_xlabel(r'$\mathbf{x}_{s,1}$')
    axes[0].set_ylabel(r'$\mathbf{x}_{s,2}$')

    sns.set_palette('Dark2')
    axes[1].plot(post)
    y_hat = np.argmax(post, axis=1)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Posterior Probability')
    print("Adjusted Rand Score: {}\n".format(metrics.adjusted_rand_score(y, y_hat)))
    print(np.argmax(post, axis=1))
    
# Initialize keras model
sem_kwargs2 = dict(
    lmda=lmda, alfa=alfa, f_class=LinearEvent, 
    f_opts=dict(l2_regularization=0.4)  
)
# using the f_opts kwargs, we can set options for the event model as well.  Here, 
# we've added l2 regularization to the network, but normally this can just be left empty


#results = sem_run(x_train, sem_kwargs2)
#plot_segmentation(results.post, y)

'''
# Right hand = J11_3D_X,J11_3D_Y,J11_3D_Z
# 2.2.1 as example:
skelfile='/Users/bezdek/Box/DCL_ARCHIVE/Project-SEM-Corpus/clean_skeleton_data/2.2.1_skel_clean.csv'
skeldf=pd.read_csv(skelfile)
#x_train=skeldf[['J11_3D_X','J11_3D_Y','J11_3D_Z']].to_numpy()
#testresults=sem_run(x_train, sem_kwargs2)
skeldf['J11_change']=np.linalg.norm(skeldf[['J11_3D_X', 'J11_3D_Y','J11_3D_Z']]-skeldf[['J11_3D_X', 'J11_3D_Y','J11_3D_Z']].shift(1), axis=1)

#testdf=df.apply(pd.to_numeric, errors='coerce')
rate='1s'
testdf = skeldf.set_index(pd.to_datetime(skeldf['sync_time'], unit='s'), drop=False)
resample_index = pd.date_range(start=skeldf.index[0], end=skeldf.index[-1], freq=rate)
dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=testdf.columns)
#df.combine_first(dummy_frame).interpolate('time').iloc[:6]
outdf=testdf.combine_first(dummy_frame).interpolate('time').resample(rate).mean()
x_train=outdf[['J11_3D_X','J11_3D_Y','J11_3D_Z','J11_change']].iloc[1:].to_numpy()
'''
fdf=pd.read_csv('/Users/bezdek/Documents/tracking/scripts/2.2.1_features.csv')
for col in fdf.columns:
    if col != 'sync_time':
        fdf[col]=zscore(fdf[col])

fdfcorr=fdf.corr()

x_train=fdf.drop(['sync_time','paper towel_4_moveg'],axis=1).to_numpy()
#cluster_id = np.argmax(testresults.post, axis=1)
breaks=[83,137,191,256,304,373,412,460]
j=0
k=0
y=[]
while k < len(x_train):
    y.append(j)
    if j < len(breaks):
        if k==breaks[j]:
            j+=1
    k+=1
'''
j=0
k=0
x_train=np.zeros([573,9])
while k < 573:
    x_train[k,j]=1
    if j < len(breaks):
        if k==breaks[j]:
            j+=1
    k+=1
'''   
testresults=sem_run(x_train, sem_kwargs2)
plot_segmentation(testresults.post,y)


def bin_times(array, bin_size=1.0):
    """ Helper function to learn the bin the subject data"""
    max_seconds=max(array)
    cumulative_binned = [np.sum(array <= t0) for t0 in np.arange(bin_size, max_seconds + bin_size, bin_size)]
    binned = np.array(cumulative_binned)[1:] - np.array(cumulative_binned)[:-1]
    binned = np.concatenate([[cumulative_binned[0]], binned])
    return binned


def string_to_segments(raw_string):
    raw_string = raw_string.split('\n')
    list_segments = [float(x.split(' ')[1]) for x in raw_string if 'BreakPoint' in x]
    return list_segments

segnorm=pd.read_csv(segpath)
seg = segnorm[(segnorm['movie1'] == '2.2.1_C1_trim') & (segnorm['condition'] == 'coarse')]
    # parse annotations, from string to a list of breakpoints for each annotation
seg['segment_processed'] = seg['segment1']
seg['segment_processed'] = seg['segment_processed'].apply(string_to_segments)

times = np.sort(list(set(seg['segment_processed'].sum()))).astype(np.float32)
n_subjs=30
bin_size=1
binned_times = bin_times(times, bin_size) / np.float(n_subjs)
binned_times=np.append(binned_times,0.0)
# Point biserial correlation:
def get_binned_boundaries(posterior, bin_size=1.0, frequency=3.0):
    
    e_hat = np.argmax(posterior, axis=1)
    
    frame_time = np.arange(1, len(e_hat) + 1) / float(frequency)
    index = np.arange(0, np.max(frame_time), bin_size)

    boundaries = np.concatenate([[0], e_hat[1:] !=e_hat[:-1]])

    boundaries_binned = []
    for t in index:
        boundaries_binned.append(np.sum(
            boundaries[(frame_time >= t) & (frame_time < (t + bin_size))]
        ))
    return np.array(boundaries_binned, dtype=bool) 

def get_point_biserial(boundaries_binned, binned_comp):
    M_1 = np.mean(binned_comp[boundaries_binned == 1])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned == 1)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n)**2))
    return r_pb

boundaries = get_binned_boundaries(testresults.post,1.0,3.0)
bicorr=get_point_biserial(boundaries,binned_times)

plt.figure(figsize=(4.5, 2.0))
plt.plot(binned_times, label='Subject Boundaries')
plt.xlabel('Time (seconds)')
plt.ylabel('Boundary Probability')

b = np.arange(len(boundaries))[boundaries][0]
plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75)
for b in np.arange(len(boundaries))[boundaries][1:]:
    plt.plot([b, b], [0, 1], 'k:', alpha=0.75)

plt.legend(loc='upper left')
plt.ylim([0, 0.6])
sns.despine()


# Plot stacked posterior probabilities:
idx = np.argwhere(np.all(testresults.post[..., :] == 0, axis=0))
active_events = np.delete(testresults.post, idx, axis=1)

plt.figure(figsize=(4.5, 2.0))
for i in range(np.shape(active_events)[1]):
    plt.bar(list(range(len(active_events))),active_events[:,i])
'''
binned=pd.read_csv('/Users/bezdek/Documents/seg_agreement/2.2.1_binned.csv')
binned[['index', 'time', 'count']] = pd.DataFrame([ x.split() for x in binned['. "Freq"'].tolist() ])
binned['time']=binned['time'].str.replace('"', '').astype(int)
binned['index']=binned['index'].astype(int)
binned['count']=binned['count'].astype(int)
binned=binned.drop(['. "Freq"','index'],axis=1)
bins=np.zeros(binned['time'].max())

plt.figure(figsize=(4.5, 2.0))
#plt.plot(binned_dishes, label='Subject Boundaries')
plt.xlabel('Time (seconds)')
plt.ylabel('Boundary Probability')

b = np.arange(len(boundaries))[boundaries][0]
plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75)
for b in np.arange(len(boundaries))[boundaries][1:]:
    plt.plot([b, b], [0, 1], 'k:', alpha=0.75)

plt.legend(loc='upper left')
plt.ylim([0, 0.6])
sns.despine()


get_point_biserial(boundaries, binned_dishes)
'''
# Boundary probability:
from scipy.special import logsumexp
def get_binned_boundary_prop(log_post, bin_size=1.0, frequency=30.0):
    """
    :param results: sem_results
    :param bin_size: seconds
    :param frequency: in Hz
    :return:
    """
    
    e_hat = np.argmax(log_post, axis=1)

    # normalize
    log_post0 = log_post - np.tile(np.max(log_post, axis=1).reshape(-1, 1), (1, log_post.shape[1]))
    log_post0 -= np.tile(logsumexp(log_post0, axis=1).reshape(-1, 1), (1, log_post.shape[1]))

    boundary_probability = [0]
    for ii in range(1, log_post0.shape[0]):
        idx = list(range(log_post0.shape[0]))
        idx.remove(e_hat[ii - 1])
        boundary_probability.append(logsumexp(log_post0[ii, idx]))
    boundary_probability = np.array(boundary_probability)

    frame_time = np.arange(1, len(boundary_probability) + 1) / float(frequency)

    index = np.arange(0, np.max(frame_time), bin_size)
    boundary_probability_binned = []
    for t in index:
        boundary_probability_binned.append(
            # note: this operation is equivalent to the log of the average boundary probability in the window
            logsumexp(boundary_probability[(frame_time >= t) & (frame_time < (t + bin_size))]) - \
            np.log(bin_size * 30.)
        )
    boundary_probability_binned = pd.Series(boundary_probability_binned, index=index)
    return boundary_probability_binned

boundary_prop = get_binned_boundary_prop(testresults.log_prior + testresults.log_like,1.0,3.0)

plt.figure(figsize=(3.5, 3.5))
plt.scatter(boundary_prop, binned_times, color='k', alpha=0.2)
plt.ylabel('Subject Boundaries Frequency')
plt.xlabel('Model log Boundary Probability')
x = boundary_prop
y = binned_times
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

# get the permutation r_pb
_boundaries = np.copy(boundaries)
n_permute = 500
r_pb_permuted = []
for ii in range(n_permute):
    np.random.shuffle(_boundaries)
    r_pb_permuted.append(get_point_biserial(_boundaries, binned_times))

def get_subjs_rpb(data, bin_size=1.0):
    # get the grouped data
    grouped_data = data.segment_processed.sum()
    
    r_pbs = []
    
    for sj in set(data.workerId):
        _binned_times =  bin_times(np.append(np.asarray(data.loc[(data.workerId == sj), 'segment_processed'].values.tolist(),dtype=np.int).flatten(),0),1.0)       
        r_pbs.append(get_point_biserial(_binned_times, grouped_data))
    return r_pbs

r_pbs = get_subjs_rpb(seg)

data=seg
sj=data.workerId.unique()[0]

import statsmodels.api as sm
from scipy import stats


plt.figure(figsize=(5.0, 5.0))
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)

plt.plot(binned_times, label='Subject Boundaries')
plt.xlabel('Time (seconds)')
plt.ylabel('Boundary Probability')

b = np.arange(len(boundaries))[boundaries][0]
plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75)
for b in np.arange(len(boundaries))[boundaries][1:]:
    plt.plot([b, b], [0, 1], 'k:', alpha=0.75)

plt.legend(loc='upper right', framealpha=1.0)
plt.ylim([0, 0.6])
plt.title('"2.2.1 - Exercise"')
sns.despine()

ax = plt.subplot2grid((2, 2), (1, 0), colspan=1)
plt.scatter(boundary_prop, binned_times, color='k', s=10, alpha=0.1)
plt.ylabel('Boundaries Frequency')
plt.xlabel('Model log Boundary Probability')
x = boundary_prop
y = binned_times
y = y[np.argsort(x)]
x = np.sort(x)
X = sm.add_constant(x)
mod = sm.OLS(y, X)
res = mod.fit()
y_hat = res.predict(X)
plt.plot(x, res.predict(X))

n = len(y_hat)
dof = n - res.df_model - 1
t = stats.t.ppf(1-0.025, df=dof)
s_err = np.sum(np.power(y - y_hat, 2))
conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x-np.mean(x)),2) /
            ((np.sum(np.power(x,2))) - n*(np.power(np.mean(x),2))))))

upper = y_hat + abs(conf)
lower = y_hat - abs(conf)
plt.fill_between(x, lower, upper, alpha=0.25)

ax = plt.subplot2grid((2, 2), (1, 1), colspan=1)
x = 'Model Subjects'.split()
sns.distplot(r_pbs,  ax=ax, norm_hist=False, label='Subjects', bins=10, color='k')
r_pb_model = get_point_biserial(boundaries, binned_times)
lb, ub = ax.get_ylim()
plt.plot([r_pb_model, r_pb_model], [0, ub], 'r', label='Model', lw='3')
plt.xlabel(r'Point-biserial correlation')
plt.ylabel('Frequency')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
sns.despine()



colors=['red','yellow','green','blue','pink','orange','purple','brown']
for x,c in zip(x_train,y):
    plt.plot(x[0],x[1],'o',color=colors[c],picker=True)
plt.show()
y_hat = np.argmax(testresults.post, axis=1)
fig,ax=plt.subplots(1)
for i,x in enumerate(testresults.post.T[0:8]):
    ax.plot(x,color=colors[i],picker=True)
    ax.set_xlabel('Time')
    ax.set_ylabel('Posterior Probability')

'''
parameters from everyday videos demo
'''    

# these are the parameters for the event model itself.
f_opts=dict(
    var_df0=10., 
    var_scale0=0.06, 
    l2_regularization=0.0, 
    dropout=0.5,
    n_epochs=10,
    t=4
)

lmda = 10**4  # stickyness parameter (prior)
alfa = 10**-1 # concentration parameter (prior)

sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'f_opts': f_opts}

results=sem_run(x_train,sem_init_kwargs)
plot_segmentation(results.post,y)

colors=['red','yellow','green','blue','pink','orange','purple','brown']

fig = plt.figure()
ax = fig.gca(projection='3d')
for x,c in zip(x_train,y):
    ax.scatter(x[0],x[1],'o',color=colors[c],picker=True)
    #ax.plot(x, y, z, label='parametric curve')
#ax.legend()

plt.show()