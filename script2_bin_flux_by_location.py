# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:07:21 2022

@author: eliza
"""

'''This is code to split the Cassini RPWS data (processed acc. to Lamy et al [2008])
into 1hr Local Time bins, and then further split by latitude (>|5|degrees, <|5|degrees) for each
year. The result will be a flux of flux density values recorded at each lat., lt bin.
These lists can be used to create a sliding window for flux density colorbar normalisation
by finding the 10th and 80th percentile of the total (sum of flux density at contributing bins).'''

#import modules
import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

#Load dataframe with flux+trajectory data, find associated LT bin for each row.
def load_df(bins_, step):
    fp= input_data_fp + '/skr_traj_df_allyears.csv'
    df = pd.read_csv(fp)
    df = df[['Latitude_krtp', 'LT', 'flux']]
    df['LT']=df['LT'].round(decimals = step)
    LT= df['LT'].copy()
    bin_indices = np.digitize(LT, bins_)
    df['bin'] = bin_indices
    return df  
#Split dataframe into data for one bin.
def split_by_LT(bin_,df):
    df_lt=df.copy()
    df_lt = df_lt.loc[df_lt['bin']==bin_, :].reset_index(drop=True)
    return df_lt
#split data for one bin into two, for different latitude regions.
def split_by_Lat(bin_, df):
    df_lat = split_by_LT(bin_, df.copy())
    fluxlowlat = np.array(df_lat.loc[(df_lat['Latitude_krtp'] > -5) & (df_lat['Latitude_krtp'] <5), 'flux'].dropna().reset_index(drop=True))
    fluxhighlat = np.array(df_lat.loc[(df_lat['Latitude_krtp'] <= -5) | (df_lat['Latitude_krtp'] >= 5), 'flux'].dropna().reset_index(drop=True))
    return fluxlowlat, fluxhighlat
#Do this for all data. 
def split_all(bins_, step, df):
    lowlat=[]
    highlat=[]
    bin_indices = np.arange(1, len(bins_)+1, step)
    for i in bin_indices:
        fluxlowlat, fluxhighlat = split_by_Lat(i, df)
        lowlat.append(fluxlowlat)
        highlat.append(fluxhighlat)
        print(i)
    return lowlat, highlat

step=1
bins_ = np.arange(0,24.1,step)
bin_indices = np.arange(1, len(bins_)+1,step)
df= load_df(bins_, step)
lowlat, highlat=split_all(bins_, step, df)

lowlat = np.array(lowlat,dtype=object)
highlat=np.array(highlat,dtype=object)
np.save(input_data_fp + '/lowlat_flux.npy', lowlat)
np.save(input_data_fp + '/highlat_flux.npy', lowlat)
