# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:28:54 2023

@author: eliza
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
figure_fp = config['filepaths']['figures']
#function for plotting legend without duplicate labels.
def legend_without_duplicate_labels(ax, loc_):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc = loc_, fontsize=20)

#Load in file with trajectory data interpolated to time cadence of radio data.
total_cassini_traj = pd.read_csv(input_data_fp + '/interped_traj_df_allyears.csv',parse_dates=['datetime_ut'])
#Load file with the start and stop times of LFE events.
lfe_df=pd.read_csv(output_data_fp + "/lfe_timestamps.csv",parse_dates=['start','end'])
#Load in file with the start and stop times of non-LFE events.
nolfe_df = pd.read_csv(output_data_fp + "/nolfe_timestamps.csv",parse_dates=['start','end'])

#Find the sections of trajectory data that occur during LFE eventsand concatenate into single dataframe.
lfe_traj_df=[]
for i,j in zip(lfe_df['start'], lfe_df['end']):
    a=total_cassini_traj.loc[total_cassini_traj["datetime_ut"].between(i,j,inclusive='left'),:]
    lfe_traj_df.append(a)
lfe_traj_df=pd.concat(lfe_traj_df, axis=0)
#Find the sections of trajectory data that occur during Non-LFE events and concatenate into single dataframe.
nolfe_traj_df=[]
for i,j in zip(nolfe_df['start'], nolfe_df['end']):
    a=total_cassini_traj.loc[total_cassini_traj["datetime_ut"].between(i,j,inclusive='left'),:]
    nolfe_traj_df.append(a)
nolfe_traj_df=pd.concat(nolfe_traj_df, axis=0)

# Start figure
plt.ioff()
fig2 = plt.figure(figsize=(36,30))
plt.rcParams.update({'font.size': 35})

#2 panel plot with 2 columns in the first panel and 3 in the lower. 
spec2 = gridspec.GridSpec(ncols=15, nrows=2, figure=fig2)
ax1 = fig2.add_subplot(spec2[0, 1:7])
ax2 = fig2.add_subplot(spec2[0, 8:14])
ax3 = fig2.add_subplot(spec2[1, 0:5])
ax4 = fig2.add_subplot(spec2[1, 5:10])
ax5 = fig2.add_subplot(spec2[1, 10:])


#========= Panel 1 ===============#
#Define spacecraft location in x, y , x for total cassini trajectory data.
xr = total_cassini_traj['xpos_ksm']
yr = total_cassini_traj['ypos_ksm']
zr = total_cassini_traj['zpos_ksm']

# establish parameters for local time separation in plots
r_fill=100
th_fill = np.arange(24)*2*np.pi/24
th_c = np.arange(100)*2*np.pi/100
x_fill=r_fill*np.cos(th_fill)
y_fill=r_fill*np.sin(th_fill)

# fill background with local time and radial bins
for i in range(0,len(x_fill),2):
    ax1.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="#555555", alpha = 0.1)
    ax2.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="#555555", alpha = 0.1)       
    for i in range(10,100,10):
        rc = i
        xc = rc*np.cos(th_c)
        yc = rc*np.sin(th_c)
        ax1.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
        ax2.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
 
# plot orbital path in X-Y and X-Z
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)

#Plot Cassini's orbits
ax1.plot(xr,yr,linewidth = 0.8, color = '#999999',alpha=0.7)
ax2.plot(xr,zr,linewidth=0.8, color = '#999999',alpha=0.7)

#Overplot the LFE events. 
for i, j in zip(lfe_df['start'], lfe_df['end']):
    traj = total_cassini_traj.loc[total_cassini_traj['datetime_ut'].between(i, j,inclusive='left'),:]
    ax1.plot(traj['xpos_ksm'], traj['ypos_ksm'], linewidth=3, color='steelblue', label='LFE')
    ax2.plot(traj['xpos_ksm'], traj['zpos_ksm'], linewidth=3, color='steelblue', label='LFE')
#Overplot the non-LFE events.     
for i, j in zip(nolfe_df['start'], nolfe_df['end']):
    traj = total_cassini_traj.loc[total_cassini_traj['datetime_ut'].between(i, j,inclusive='left'),:]
    ax1.plot(traj['xpos_ksm'], traj['ypos_ksm'], linewidth=3,alpha=0.5, color='darkorange', label = 'Non-LFE')
    ax2.plot(traj['xpos_ksm'], traj['zpos_ksm'], linewidth=3,alpha=0.5, color='darkorange', label = 'Non-LFE')

# limits field of view in plot
ax1.set_xlim([-70,50])
ax1.set_ylim([-80,70])
ax2.set_xlim([-50,50])
ax2.set_ylim([-50,50])
# label x and y axis
ax1.set_xlabel('$X_{KSM}  (R_S)$', fontsize=30)
ax2.set_xlabel('$X_{KSM}  (R_S)$', fontsize=30)
ax1.set_ylabel('$Y_{KSM}  (R_S)$', fontsize=30)
ax2.set_ylabel('$Z_{KSM}  (R_S)$', fontsize=30)

# put legend on plot
legend_without_duplicate_labels(ax1, 'lower left')
legend_without_duplicate_labels(ax2, 'upper right')

#========== Panel 2: histograms of Radius, Latitude, Local Time ================#
#=======(c) Radius============#
#establish radial bins
radial_bins = np.arange(10)*10 
#Bin total Cassini data into radial bins
n,x = np.histogram(total_cassini_traj['Range_ksm'],bins = radial_bins)
#get bin centers coordinates
r_bin_cents = 0.5*(x[1:]+x[:-1])
#normalise to total counts of Cassini data.
n_norm = n/len(total_cassini_traj)
#Plot curve of total Cassini data
ax3.plot(r_bin_cents,n_norm,linestyle=(0, (1, 1)),linewidth=5,color = '#999999',drawstyle='steps')
#Bin LFE data into radial bins
n_lfe,x_lfe = np.histogram(lfe_traj_df['Range_ksm'],bins = radial_bins)
#Normalise to total LFE counts.
n_lfe = np.array(n_lfe)/len(lfe_traj_df['Range_ksm'])
#Bin non-LFE data into radial bins
n_nolfe,x_nolfe = np.histogram(nolfe_traj_df['Range_ksm'],bins = radial_bins)
#normalise to total non-lfe counts.
n_nolfe =np.array(n_nolfe)/len(nolfe_traj_df)
#Plot
ax3.plot(r_bin_cents,n_lfe,label='LFE', linewidth=5, color='steelblue',drawstyle='steps')
ax3.plot(r_bin_cents,n_nolfe, color = 'darkorange',alpha=0.5,label='Non-LFE', linewidth=5,drawstyle='steps')
#Formatting Axis
ax3.set_xlim(0, 90)
ax3.tick_params(labelsize=25)
ax3.xaxis.set_minor_locator(MultipleLocator(10))
ax3.tick_params('both', length=10, width=1, which='major')
ax3.tick_params('both', length=5, width=1, which='minor')
ax3.xaxis.set_major_locator(MultipleLocator(20))
ax3.xaxis.set_major_formatter('{x:.0f}')
ax3.set_xlabel('$R  (R_S)$', fontsize=30)
ax3.set_ylabel('Normalised Counts', fontsize=30)
legend_without_duplicate_labels(ax3, 'upper right')

#========(d)Latitude==========#
#Establish Latitude bins
theta_bins = (np.arange(19)-9)*10
#Bin total cassini data into latitude bins
n,x = np.histogram(total_cassini_traj['Latitude_ksm'],bins = theta_bins)
#normalise to total cassini counts
n_norm = n/len(total_cassini_traj)
#get bin centers coordinates
t_bin_cents = 0.5*(x[1:]+x[:-1])
#Plot histogram of normalised total cassini trajectory data.
ax4.plot(t_bin_cents,n_norm,linestyle=(0, (1, 1)), color = '#999999', linewidth = 5,drawstyle='steps')
#Bin LFE events into latitude bins
n_lfe,x_lfe = np.histogram(lfe_traj_df['Latitude_ksm'],bins = theta_bins)
#normalise to total lfe counts
n_lfe = np.array(n_lfe)/len(lfe_traj_df)
#Plot normalised data during LFE events.
ax4.plot(t_bin_cents,n_lfe, color = 'steelblue',label='LFE', linewidth=5,drawstyle='steps')
#Bin non-lfe events into Latitude bins
n_nolfe,x_nolfe = np.histogram(nolfe_traj_df['Latitude_ksm'],bins = theta_bins)
#normalise to total non-lfe counts
n_nolfe =np.array(n_nolfe)/len(nolfe_traj_df)
#Plot normalised data during non-LFE events.
ax4.plot(t_bin_cents,n_nolfe, color = 'darkorange',alpha=0.5,label='Non-LFE', linewidth=5,drawstyle='steps')

#Formatting axis
ax4.xaxis.set_minor_locator(MultipleLocator(10))
ax4.tick_params('both', length=10, width=1, which='major')
ax4.tick_params('both', length=5, width=1, which='minor')
ax4.xaxis.set_major_locator(MultipleLocator(25))
ax4.xaxis.set_major_formatter('{x:.0f}')
ax4.set_xlabel('Latitude ($^{\circ}$)', fontsize=30)
ax4.tick_params(labelsize=25)
legend_without_duplicate_labels(ax4, 'upper left')

#========(e)Local Time===============#
#define bins for local time
lt_bins = np.arange(25)
#get centers of each local time bin.
lt_bin_cents = 0.5*(lt_bins[1:]+lt_bins[:-1])
#Bin total cassini data into local time bins
n,x = np.histogram(total_cassini_traj['LT_ksm'],bins = lt_bins)
#normalise to total counts
n_norm = n/len(total_cassini_traj)
#Plot histogram of normalised total cassini data.
ax5.plot(lt_bin_cents,n_norm,linestyle=(0, (1, 1)), color = '#999999', linewidth = 5,drawstyle='steps')
#Bin data during LFE events into local time bins and normalise.
n_lfe,x_lfe = np.histogram(lfe_traj_df['LT_ksm'],bins = lt_bins)
n_lfe = np.array(n_lfe)/len(lfe_traj_df)
#Plot normalised LFE data.
ax5.plot(lt_bin_cents,n_lfe, color = 'steelblue',label='LFE', linewidth=5,drawstyle='steps')
#Bin data during Non-LFE events into local time bins and normalise.
n_nolfe,x_nolfe = np.histogram(nolfe_traj_df['LT_ksm'],bins = lt_bins)
n_nolfe =np.array(n_nolfe)/len(nolfe_traj_df)
#Plot normalised non-lfe data.
ax5.plot(lt_bin_cents,n_nolfe, color = 'darkorange',alpha=0.5,label='Non-LFE', linewidth=5,drawstyle='steps')

#Format axis
ax5.xaxis.set_minor_locator(MultipleLocator(2))
ax5.tick_params('both', length=10, width=1, which='major')
ax5.tick_params('both', length=5, width=1, which='minor')
ax5.xaxis.set_major_locator(MultipleLocator(5))
ax5.xaxis.set_major_formatter('{x:.0f}')
ax5.tick_params(labelsize=25)
ax5.set_xlabel('LT (hr)', fontsize=30)
legend_without_duplicate_labels(ax5, 'upper left')

#Label each plot.
ax1.text(0, 1.05, 'a', horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes,fontsize=35, weight='bold')
ax2.text(0, 1.05, 'b', horizontalalignment='center',verticalalignment='center',transform = ax2.transAxes,fontsize=35, weight='bold')
ax3.text(0, 1.05, 'c', horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=35, weight='bold')
ax4.text(0, 1.05, 'd', horizontalalignment='center',verticalalignment='center',transform = ax4.transAxes,fontsize=35, weight='bold')
ax5.text(0, 1.05, 'e', horizontalalignment='center',verticalalignment='center',transform = ax5.transAxes,fontsize=35, weight='bold')

#save figure
plt.subplots_adjust(wspace=2.0)
plt.savefig(figure_fp + '/lfe_spread_steps_.png', bbox_inches='tight')
plt.clf()

