# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os, sys
import scipy
import mne
from mne.filter import filter_data, notch_filter
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pickle
import json
import warnings
import glob
from joblib import Parallel, delayed
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache",exist_ok=True)
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))
from utils import *
from feat_funcs import *

def load_edf_file(file_name):
    # returns raw mne object, dataframe of data, dataframe of labels, sampling frequency
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose = 0)
    fs = raw.info['sfreq']
    df = raw.to_data_frame().set_index('time')
    times = raw.times
    annotations = raw.annotations
    label = np.zeros(len(times)).astype(int)
    if annotations:
        for anno in annotations:
            sz_onset = anno['onset']
            sz_dura = anno['duration']
            sz_end = sz_onset+sz_dura
            label[(times >= sz_onset)&(times <= sz_end)] = 1
    label_df = pd.DataFrame({'time':times,'labels':label})
    return raw, df, label_df, fs

import torch
import torch.nn as nn
import torch.nn.functional as F

svm_path = os.path.join(working_dir,'SVM')
sys.path.append(svm_path)
from utils_baseline import (
        extract_features, train_one_class_svm, compute_novelty_scores,
        estimate_outlier_fraction, detect_seizure, apply_persistence
    )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_cnn = torch.load(svm_path+"/model_1130.pt", map_location=torch.device(device), weights_only=False)
# model_cnn.eval()

feat_setting = {'name':'svm',
                'win':1, 'stride':0.5,
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40} # in seconds

def sparcnet_single(data, fs):
    """Do seizure prediciton on a 10-second clip.
    Data should be a pd dataframe

    Args:
        data (_type_): _description_
        fs (_type_): _description_
    """
    if 'Fz-Cz' in data.columns:
        data = data.drop(columns=['Fz-Cz'])
    if 'Cz-Pz' in data.columns:
        data = data.drop(columns=['Cz-Pz'])
    data = data.values
    data = bandpass_filter(data, fs, lo = feat_setting['lowcut'], hi = feat_setting['highcut'])
    data = downsample(data, fs, feat_setting['resample'])
    data = np.where(data<=500, data, 500)
    data = np.where(data>=-500, data, -500)
    data = torch.from_numpy(data).float()
    data = data.T.unsqueeze(0)
    data = data.to(device)
    output, _ = model_cnn(data)
    sz_prob = F.softmax(output,1).detach().cpu().numpy().flatten()
    return sz_prob

def custom_bipolar(df,pairs):
    filtered = df['filtered']
    columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
               'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    data = {}
    for p in pairs:
        ch1, ch2 = p.split('-')
        # try:
        ch_data = filtered[ch1]-filtered[ch2]
        data[p] = ch_data
        # except:
        #     pass
    return pd.DataFrame(data,index=filtered.index)


montage_dict = {'full':['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
                'uneeg_left_front':['F7-T3'],
                'uneeg_left_back':['T3-T5'],
                'uneeg_right_front':['F8-T4'],
                'uneeg_right_back':['T4-T6'],
                # 'uneeg_right':['F8-T4','T4-T6'],
                # 'uneeg_left':['F7-T3','T3-T5'],
                # 'uneeg_bilateral4':['F7-T3','T3-T5','F8-T4', 'T4-T6'],
                'uneeg_bilateral_back2':['T3-T5','T4-T6'],
                'uneeg_bilateral_front2':['F7-T3','F8-T4'],
                'uneeg_vert_left':lambda df: custom_bipolar(df,['C3-T3']),
                'uneeg_vert_right':lambda df: custom_bipolar(df,['C4-T4']),
                'uneeg_diag_left_front':lambda df: custom_bipolar(df,['F3-T3']),
                'uneeg_diag_left_back':lambda df: custom_bipolar(df,['P3-T3']),
                'uneeg_diag_right_front':lambda df: custom_bipolar(df,['F4-T4']),
                'uneeg_diag_right_back':lambda df: custom_bipolar(df,['P4-T4']),
                'uneeg_diag_bilateral_front':lambda df: custom_bipolar(df,['F3-T3','F4-T4']),
                'uneeg_diag_bilateral_back':lambda df: custom_bipolar(df,['P3-T3','P4-T4']),
                'uneeg_vert_bilateral':lambda df: custom_bipolar(df,['C3-T3','C4-T4']),
                'epiminder_2':['C3-P3','C4-P4'],
                'ceribell':['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
                # 'epiminder_4':['C3-P3','C4-P4','T3-T5','T4-T6'],
                }

def process_pat(pat, group):
    warnings.filterwarnings('ignore')
    group = group.sort_values('file')
    iic_file = group[group['type']=='iic'].iloc[0]['file']
    raw, df, label_df, fs = load_edf_file(iic_file)
    prepro = Preprocessor()
    prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    preprocessed = prepro.preprocess(df)
    for m in montage:
        montage_processor = montage_dict[m]
        if isinstance(montage_processor,list):
            data_df = preprocessed['BIPOLAR']
            data_df = data_df[montage_dict[m]]
        else:
            data_df = montage_processor(preprocessed)
        train_data = data_df.iloc[:int(fs*60),:].values
        clf_list = []
        for i in range(train_data.shape[1]):
            X_train = extract_features(train_data[:, i], fs=fs)
            X_train = np.nan_to_num(X_train)
            clf = train_one_class_svm(X_train)
            clf_list.append(clf)

        for _, row in group.iterrows():
            file_name = row['file']
            raw, df, label_df, fs = load_edf_file(file_name)
            prepro = Preprocessor()
            prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
            preprocessed = prepro.preprocess(df)
            if isinstance(montage_processor,list):
                data_df = preprocessed['BIPOLAR']
                data_df = data_df[montage_dict[m]]
            else:
                data_df = montage_processor(preprocessed)

            prob_path = os.path.join(prob_folder,m,file_name.split('/')[-1].replace('.edf','.csv'))
            if os.path.exists(prob_path) and not force:
                continue
            
            len_feat = extract_features(data_df.iloc[:, 0].values, fs=fs)
            start_time_s = data_df.index.min()
            time_vals = start_time_s + feat_setting['stride'] + np.arange(0, len(len_feat)) * feat_setting['stride']
            feat_labels = [label_df.loc[(data_df.index >= win_stop-feat_setting['win']) & (data_df.index < win_stop),'labels'].any().astype(int) for win_stop in time_vals]
            pred_df_final = pd.DataFrame(index=time_vals)
            pred_df_final['label'] = feat_labels
            for i in range(data_df.shape[1]):
                X_test = extract_features(data_df.iloc[:, i].values, fs=fs)
                X_test = np.nan_to_num(X_test)

                y_pred = compute_novelty_scores(clf_list[i], X_test)
                nu_hat = estimate_outlier_fraction(y_pred, n=20)
                
                smoothing_sigma = 2 * int(len(nu_hat) / 1000 + 1)
                nu_filt = np.round(gaussian_filter1d(nu_hat, smoothing_sigma), 100)
                pred_df_final['nu_hat_'+data_df.columns[i]] = nu_hat
                pred_df_final['sz_prob_'+data_df.columns[i]] = nu_filt

            
            pred_df_final.index = pd.to_datetime(pred_df_final.index, unit='s')
            pred_df_final.to_csv(prob_path)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run SPaRCNet prediction on folder of edf files, store probabilities"
    )

    # Define arguments
    parser.add_argument("-f", "--folder", type=str, default='', help="Folder containing files to process, if not specified, all seizure and interictal clips in emu dataset")
    parser.add_argument("-o", "--output", type=str, default='svm_results/prob', help="Folder to store sparcnet probability files, one file per edf file")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montage code to run, e.g epiminder_2, if 'all', all available montages in montage_dict")
    parser.add_argument("--force", action='store_true', help="Force re-running, even if probability file already exist")
    
    # Parse the arguments
    params = vars(parser.parse_args())

    if not params['folder']:
        sz_folder = '/mnt/sauce/littlab/users/haoershi/emu_dataset/seizure'
        sz_files = sorted(glob.glob(sz_folder+'/*.edf'))
        iic_folder = '/mnt/sauce/littlab/users/haoershi/emu_dataset/interictal'
        iic_files = sorted(glob.glob(iic_folder+'/*.edf'))
        all_files = sz_files+iic_files
    else:
        all_files = glob.glob(params['folder'])

    prob_folder = params['output']
    os.makedirs(prob_folder,exist_ok=True)

    if params['montage'] == 'all':
        montage = list(montage_dict.keys())
    else:
        montage = params['montage'].split(',')
    for m in montage:
        os.makedirs(os.path.join(prob_folder,m),exist_ok=True)
    force = params['force']
    # with tqdm(total=len(all_files),desc = 'Processing block'):
    #     # for file_name in all_files:
    #     #     process_file(file_name)
    #     results = Parallel(n_jobs=40)(delayed(process_file)(file_name) for file_name in all_files)
    # for file_name in tqdm(all_files,total=len(all_files)):
    file_df = pd.DataFrame({'file':all_files})
    file_df['patient'] = file_df['file'].apply(lambda x: x.split('/')[-1].split('_')[0])
    file_df['type'] = file_df['file'].apply(lambda x: x.split('/')[-1].split('_')[1])
    n_pat = len(file_df['patient'].unique())
    with tqdm(total=n_pat,desc = 'Processing block'):
        # for file_name in all_files:
        #     process_file(file_name)
        results = Parallel(n_jobs=20)(delayed(process_pat)(pat, group) for pat, group in file_df.groupby('patient'))
        
