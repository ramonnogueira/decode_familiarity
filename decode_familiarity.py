# Ramon Nogueira January 2022

# Load all packages
import os
import h5py
import csv
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import tables
import pandas as pd
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from google.colab import files
import io

#######################################################
# Function to create figures
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

############################################
# Load Data from a google colab project
files_all=files.upload()
fkeys=list(files_all.keys())

# Individual sessions
# Parameters
n_cv=20
n_sh=1000
reg=1

# Loop on the classifier
perf_pre=nan*np.zeros((len(fkeys),n_cv,2))
perf_sh_pre=nan*np.zeros((len(fkeys),n_sh,n_cv))
for i in range(len(fkeys)):
  print (fkeys[i])
  data=pd.read_csv(io.BytesIO(files_all[fkeys[i]]))
  clase=np.array(data['trial'],dtype=np.int16)
  features=nan*np.zeros((len(clase),1))
  features[:,0]=data['amplitude']

  # Evaluate Decoding Performance
  cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=0.2)
  g=-1
  for train_index, test_index in cv.split(features,clase):
    g=(g+1)
    cl=LinearSVC(C=1/reg,class_weight='balanced')
    cl.fit(features[train_index],clase[train_index])
    perf_pre[i,g,0]=cl.score(features[train_index],clase[train_index])
    perf_pre[i,g,1]=cl.score(features[test_index],clase[test_index])

  # Evaluate Shuffled Decoding Performance
  for k in range(n_sh):
    clase_sh=np.random.permutation(clase)
    cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=0.2)
    g=-1
    for train_index, test_index in cv.split(features,clase_sh):
      g=(g+1)
      cl=LinearSVC(C=1/reg,class_weight='balanced')
      cl.fit(features[train_index],clase_sh[train_index])
      perf_sh_pre[i,k,g]=cl.score(features[test_index],clase_sh[test_index])

perf=np.mean(perf_pre,axis=1)
perf_m=np.mean(perf,axis=0)
perf_sh=np.mean(perf_sh_pre,axis=2)
perf_sh_m=np.mean(perf_sh,axis=0)
plt.hist(perf_sh_m)
plt.axvline(perf_m[1])

print (perf)
print (perf_m)

# Pseudo-population analysis
reg=1
nt=100
n_rand=100
n_sh=1000
perc_tr=0.8

feat_tr=nan*np.zeros((n_rand,2*nt,len(fkeys)))
feat_te=nan*np.zeros((n_rand,2*nt,len(fkeys)))
clase_pseudo=np.zeros(2*nt)
clase_pseudo[0:nt]=1
# Loop on the classifier
for i in range(len(fkeys)):
  #print (fkeys[i])
  data=pd.read_csv(io.BytesIO(files_all[fkeys[i]]))
  clase=data['trial']
  act=data['amplitude']
  ind1=np.where(clase==1)[0]
  ind0=np.where(clase==0)[0]
  nt1=int(len(ind1)*perc_tr)  
  nt0=int(len(ind0)*perc_tr)  
  
  for ii in range(n_rand):
    ind1_p=np.random.permutation(ind1)
    ind0_p=np.random.permutation(ind0)
    feat_tr[ii,0:nt,i]=np.random.choice(act[ind1_p][0:nt1],nt,replace=True)
    feat_tr[ii,nt:,i]=np.random.choice(act[ind0_p][0:nt0],nt,replace=True)
    feat_te[ii,0:nt,i]=np.random.choice(act[ind1_p][nt1:],nt,replace=True)
    feat_te[ii,nt:,i]=np.random.choice(act[ind0_p][nt0:],nt,replace=True)

# Evaluate Decoding Performance
perf_pseudo=nan*np.zeros((n_rand,2))
for k in range(n_rand):
    cl=LinearSVC(C=1/reg,class_weight='balanced')
    cl.fit(feat_tr[k],clase_pseudo)
    perf_pseudo[k,0]=cl.score(feat_tr[k],clase_pseudo)
    perf_pseudo[k,1]=cl.score(feat_te[k],clase_pseudo)

###################################################
#Evaluate Shuffle perf
perf_ps_sh=nan*np.zeros((n_sh,n_rand))
for kk in range(n_sh):
  print (kk)
  feat_tr=nan*np.zeros((n_rand,2*nt,len(fkeys)))
  feat_te=nan*np.zeros((n_rand,2*nt,len(fkeys)))
  clase_pseudo=np.zeros(2*nt)
  clase_pseudo[0:nt]=1
  for i in range(len(fkeys)):
    #print (fkeys[i])
    data=pd.read_csv(io.BytesIO(files_all[fkeys[i]]))
    clase=data['trial']
    clase_sh=np.random.permutation(clase)
    act=data['amplitude']
    ind1=np.where(clase_sh==1)[0]
    ind0=np.where(clase_sh==0)[0]
    nt1=int(len(ind1)*perc_tr)  
    nt0=int(len(ind0)*perc_tr)  

    for ii in range(n_rand):
      ind1_p=np.random.permutation(ind1)
      ind0_p=np.random.permutation(ind0)
      feat_tr[ii,0:nt,i]=np.random.choice(act[ind1_p][0:nt1],nt,replace=True)
      feat_tr[ii,nt:,i]=np.random.choice(act[ind0_p][0:nt0],nt,replace=True)
      feat_te[ii,0:nt,i]=np.random.choice(act[ind1_p][nt1:],nt,replace=True)
      feat_te[ii,nt:,i]=np.random.choice(act[ind0_p][nt0:],nt,replace=True)
  
  for k in range(n_rand):
    cl=LinearSVC(C=1/reg,class_weight='balanced')
    cl.fit(feat_tr[k],clase_pseudo)
    perf_ps_sh[kk,k]=cl.score(feat_te[k],clase_pseudo)

perf_ps=np.mean(perf_pseudo,axis=0)
perf_ps_sh_m=np.mean(perf_ps_sh,axis=1)
print (perf_ps)

##########################################
# Plot the results
alpha_sig=0.05
sim_sh_sort=np.sort(perf_sh_m)
ps_sh_sort=np.sort(perf_ps_sh_m)
int_sim=int(alpha_sig/2*n_sh)

vmin=-0.3
vmax=0.3
fig=plt.figure(figsize=(1.5,2))
ax=fig.add_subplot(1,1,1)
adjust_spines(ax,['left','bottom'])
ax.set_ylim([0.35,1.0])
ax.set_xlim([-0.5,1.5])
ax.set_ylabel('Decoding Performance')
# Simultaneous
ax.scatter(np.random.normal(0,0.04,len(perf)),perf[:,1],s=1,color='black')
ax.scatter([0],perf_m[1],s=20,color='black')
plt.fill_between([vmin,vmax],[sim_sh_sort[int_sim],sim_sh_sort[int_sim]],[sim_sh_sort[-int_sim],sim_sh_sort[-int_sim]],alpha=0.5,color='black')
# Pseudosimultaneous
ax.scatter([1],perf_ps[1],s=20,color='black')
plt.fill_between([1+vmin,1+vmax],[ps_sh_sort[int_sim],ps_sh_sort[int_sim]],[ps_sh_sort[-int_sim],ps_sh_sort[-int_sim]],alpha=0.5,color='black')
#
plt.xticks([0,1],['Individual','Pseudo-population'],rotation=90)
ax.plot(np.linspace(-0.5,1.5,10),0.5*np.ones(10),color='black',linestyle='--')
fig.savefig('../decoding_familiarity_ls_protocol.pdf',dpi=500,bbox_inches='tight')


