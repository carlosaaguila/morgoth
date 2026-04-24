# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os, sys
import scipy
import mne
from mne.filter import filter_data, notch_filter
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
from calc_metrics_update import *

feat_settings = {'sparcnet':{
                    'win':int(10), 'stride':int(2),
                    'reref':'BIPOLAR', 'resample':200,
                    'lowcut':1, 'highcut':40}, # in seconds,
                'svm':{
                'win':int(1), 'stride':int(1),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40}, # in seconds
                'dynasd':{
                'win':1, 'stride':0.5,
                'reref':'BIPOLAR', 
                'lowcut':1, 'highcut':40},
                'oncet':{
                'win':int(1), 'stride':int(1),
                'reref':'BIPOLAR', 
                'lowcut':1, 'highcut':40},
                'feat':{
                'win':int(5), 'stride':int(5),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40},} # in seconds

montage_list = ['full',
                'uneeg_left_front',
                'uneeg_left_back',
                'uneeg_right_front',
                'uneeg_right_back',
                # 'uneeg_right',
                # 'uneeg_left',
                # 'uneeg_bilateral4',
                'uneeg_bilateral_back2',
                'uneeg_bilateral_front2',
                'uneeg_vert_left',
                'uneeg_vert_right',
                'uneeg_diag_left_front',
                'uneeg_diag_left_back',
                'uneeg_diag_right_front',
                'uneeg_diag_right_back',
                'uneeg_diag_bilateral_front',
                'uneeg_diag_bilateral_back',
                'uneeg_vert_bilateral',
                'epiminder_2',
                'ceribell',
                # 'epiminder_4',
                # 'epiminder_simulate',
                # 'zero'
                ]


# def average_metric(df):
#     summary = {}
#     for col in df.columns:
#         if col in ['total_dura','tn_min','fp_min','total_sz_dura']:
#             summary[col] = df[col].sum(skipna=True)
#         elif col == 'tn':
#             summary[col] = (df[col]*df['tn_min']).sum(skipna=True) / df['tn_min'].sum(skipna=True)
#         elif col in ['avg_sz_dura','auroc_sample','auprc_sample','recall_event','precision_event', 'precision_sample','fp', 'f1_event','f1_sample']:
#             summary[col] = df[col].mean(skipna=True)
#         elif col in ['recall_sample']:
#             summary[col] = (df[col]*df['avg_sz_dura']).sum(skipna=True) / df['avg_sz_dura'].sum(skipna=True)
#     summary_df = pd.DataFrame([summary])
#     return summary_df

def patient_metrics(pred_file_df):
    all_metrics = []
    for key, group in pred_file_df.groupby('admission_id'):
        if group['is_sz'].sum() == 0:
            continue
        segment_metrics = []
        auc_prob = []
        auc_label = []
        auc_pred = []
        for _, row in group.iterrows():
            try:
                pred_df = pd.read_csv(row['pred_file'],index_col=0)
                if len(pred_df) == 0:
                    print(row['pred_file'].split('/')[-1][:-4])
                    continue
            except:
                continue
            label = pred_df.iloc[:,-1].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['pred'].values
            event_id = row['pred_file'].split('/')[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=feat_setting['stride'])
            metric_row = pd.DataFrame([metrics],index=[event_id])
            segment_metrics.append(metric_row)
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        segment_metrics = pd.concat(segment_metrics,axis=0).sort_index()
        patient_metrics = compute_metrics(auc_label, auc_pred, auc_prob, stride=feat_setting['stride'])
        patient_metrics['avg_sz_dura'] = np.nansum(segment_metrics['total_sz_dura'].values)/np.nansum(segment_metrics['num_sz'].values)
        patient_metrics['num_sz'] = np.nansum(segment_metrics['num_sz'].values)
        patient_metrics['num_pred'] = np.nansum(segment_metrics['num_pred'].values)
        patient_metrics['recall_event'] = np.nanmean(segment_metrics['recall_event'].values)
        patient_metrics['fp'] = np.nanmean(segment_metrics['fp'].values)
        patient_metrics['precision_event'] = np.nansum(segment_metrics['precision_event'] * segment_metrics['num_pred']) / np.nansum(segment_metrics['num_pred'])
        patient_metrics['f1_event'] = 2*patient_metrics['recall_event']*patient_metrics['precision_event']/(patient_metrics['recall_event']+patient_metrics['precision_event'])
        all_metrics.append(pd.DataFrame([patient_metrics],index=[key]))
    all_metrics = pd.concat(all_metrics,axis=0).sort_index() # this is per patient metrics
    all_metrics[['precision_event','f1_event']] = all_metrics[['precision_event','f1_event']].fillna(0.0)
    return all_metrics

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run SPaRCNet prediction on folder of edf files, store probabilities"
    )

    # Define arguments
    parser.add_argument("--model", type=str, default='sparcnet', help="Folder to store sparcnet prediction files, one file per edf file")
    # parser.add_argument("-o", "--output", type=str, default='dynasd_results/metrics', help="Folder to store sparcnet metric files, one file per threshold setting")
    # parser.add_argument("--pred_folder", type=str, default='dynasd_results/pred', help="Folder of sparcnet prediction files, one file per edf file")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montage code to run, e.g epiminder_2, if 'all', all available montages in montage_list")
    parser.add_argument("-f", "--force", action='store_true', help="Force re-running")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Use prediction generated from threshold x")
    parser.add_argument("-s", "--setting", type=str, default='', help="Use prediction generated from setting x")
    
    # Parse the arguments
    params = vars(parser.parse_args())
    model = params['model']
    metric_folder = f'{model}_results/metrics'#params['output']
    pred_folder = f'{model}_results/pred'#params['pred_folder']
    force = params['force']
    thres = params['thres']
    model_name = pred_folder.split('/')[0].split('_')[0]
    feat_setting = feat_settings[model_name]

    setting_folder = params['setting']
    print(setting_folder)
    metric_folder = os.path.join(metric_folder, setting_folder)
    pred_folder = os.path.join(pred_folder, setting_folder)
    
    if params['montage'] == 'all':
        montage = montage_list
        montage = [m for m in montage if os.path.exists(f'{pred_folder}/{m}')]
    else:
        montage = params['montage'].split(',')


    for m in montage:
        # metric folder for specific setting and montage
        os.makedirs(os.path.join(metric_folder,m),exist_ok=True)
        
        pred_files = glob.glob(os.path.join(pred_folder, m,  '*.csv'))
        out_file = os.path.join(metric_folder,m,'segment_metrics.csv')
        full_metrics = []
        for f in pred_files:
            pred_df = pd.read_csv(f,index_col=0)
            label = pred_df['label'].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['pred'].values
            event_id = f.split('/')[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=feat_setting['stride'])
            metric_row = pd.DataFrame([metrics],index=[event_id])
            full_metrics.append(metric_row)
        full_metrics = pd.concat(full_metrics,axis=0).sort_index()
        full_metrics[['precision_event','f1_event']] = full_metrics[['precision_event','f1_event']].fillna(0.0)
        full_metrics.to_csv(out_file)

        out_file = os.path.join(metric_folder,m,'patient_metrics.csv')
        if not force and os.path.exists(out_file):
            continue
        pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
        pred_file_df['admission_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1].split('_')[0])
        pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1][:-4])
        pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x)
        patient_map = pd.read_csv('../emu_dataset/dataset_admission_info.csv', dtype={'patient_id':str})
        # pred_file_df = pred_file_df.merge(patient_map, on='admission_id', how='left')
        # pred_file_df['patient_id'] = pred_file_df['patient_id'].fillna(pred_file_df['admission_id'])
        # pred_file_df['is_detected'] = pred_file_df['event_id'].apply(lambda x: x in eligible_ids)

        all_metrics = patient_metrics(pred_file_df)
        all_metrics.to_csv(out_file)
        
        # filtered_metrics = patient_metrics(pred_file_df[pred_file_df['is_detected']])
        # filtered_metrics.to_csv(out_file.replace('.csv','_filtered.csv'))
        

        
