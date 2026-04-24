# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import warnings
import glob
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache",exist_ok=True)
warnings.filterwarnings("ignore")

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))
from utils import *
from feat_funcs import *

svm_path = os.path.join(working_dir,'SVM')
sys.path.append(svm_path)
from utils_baseline import (
        extract_features, train_one_class_svm, compute_novelty_scores,
        estimate_outlier_fraction, detect_seizure, apply_persistence
    )


feat_settings = {'sparcnet':{
                'win':int(10), 'stride':int(2),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40},
                'dynasd':{'win':1, 'stride':0.5,
                'reref':'BIPOLAR', 
                'lowcut':1, 'highcut':40},
                'svm':{'win':1, 'stride':0.5,
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
                'ceribell'
                # 'epiminder_4',
                # # 'epiminder_simulate',
                # 'zero'
                ]

def process_file(file_name, thres=None):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder,setting_folder,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name,index_col=0)
    prob = prob_df.iloc[:,:6].values
    sz_prob = prob[:,1]
    if thres is None:
        pred = get_prediction(sz_prob, mul=0.2, gap_num=4, min_event_num=20)
    else:
        # pred = (prob_mat >= thres).astype(int)
        # perc_chan = pred.sum(axis=1)/pred.shape[1]
        # pred = perc_chan >= 1
        pred = (sz_prob >= thres).astype(int)
        pred = get_event_smoothed_pred(smooth_pred(pred), sz_prob, gap_num=int(4/feat_setting['stride']), min_event_num=int(20/feat_setting['stride']))  #int(4/feat_setting['stride'])
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    # pred_df['smoothed_pred'] = get_event_smoothed_pred(smooth_pred(pred_df['pred'].values)) 
    pred_df = pd.concat([pred_df, prob_df.iloc[:,-1]],axis=1)
    pred_df.to_csv(out_file)

def process_file_feat(file_name, thres=None):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder,setting_folder,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name)
    sz_prob = prob_df['sz_prob'].values
    if thres is None:
        pred = get_prediction(sz_prob, mul=0.2, gap_num=4, min_event_num=20)
    else:
        # pred = (prob_mat >= thres).astype(int)
        # perc_chan = pred.sum(axis=1)/pred.shape[1]
        # pred = perc_chan >= 1 
        pred = (sz_prob >= thres).astype(int)
        pred = get_event_smoothed_pred(smooth_pred(pred), sz_prob, gap_num=int(4/feat_setting['stride']), min_event_num=int(20/feat_setting['stride']))  #int(4/feat_setting['stride'])
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    # pred_df['smoothed_pred'] = get_event_smoothed_pred(smooth_pred(pred_df['pred'].values)) 
    pred_df = pd.concat([pred_df, prob_df[['label']]],axis=1)
    pred_df.to_csv(out_file)


def get_optimal_thres(prob_files):
    all_prob = []
    all_label = []
    for f in prob_files:
        prob_df = pd.read_csv(f, index_col=0)
        sz_prob = prob_df['LPD'].values
        label = prob_df['label'].values
        all_prob.extend(sz_prob)
        all_label.extend(label)
    fpr, tpr, thres = roc_curve(all_label, all_prob)
    opt_thres = thres[np.argmax(tpr-fpr)]
    return opt_thres

from timescoring.annotations import Annotation
from timescoring import scoring 
from joblib import Parallel, delayed

def compute_eventwise_f1(true, prob, t, stride):
    pred = (prob >= t).astype(int)
    labels = Annotation(true, 1/stride)
    preds = Annotation(pred, 1/stride)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90)
    scores = scoring.EventScoring(labels, preds, param)
    return scores.f1
        
def get_optimal_thres_f1(prob_files, stride):
    true = []
    prob = []
    for f in prob_files:
        prob_df = pd.read_csv(f, index_col=0)
        sz_prob = prob_df['LPD'].values
        label = prob_df['label'].values
        prob.extend(sz_prob)
        true.extend(label)
    _, _, thres = roc_curve(true, prob)
    N = 200
    if len(thres) > N:
        idx = np.linspace(0, len(thres) - 1, N).astype(int)
        thres = thres[idx]
    results = Parallel(n_jobs=40)(
            delayed(compute_eventwise_f1)(true, prob, t, stride) for t in thres
        )
    opt_ind = np.argmax(results)
    opt_thres = thres[opt_ind] 
    return opt_thres


def process_file_ndd(file_name, thres=None, avg = True):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder,setting_folder,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name,index_col=0)
    prob_mat = prob_df[[c for c in prob_df.columns if c.startswith('prob')]].values
    sz_prob = prob_mat.mean(axis=1)
    if thres is None:
        pred = ieeg_get_prediction(prob_mat, mul=1.2, gap_num=4, min_event_num=20)
    else:
        if avg:
            pred = (sz_prob >= thres).astype(int)
        else:
            pred = (prob_mat >= thres).astype(int)
            # perc_chan = pred.sum(axis=1)/pred.shape[1]
            # pred = perc_chan >= 1
            pred = pred.sum(axis=1) >= min(2,pred.shape[1])
        pred = get_event_smoothed_pred(smooth_pred(pred), gap_num=int(4/feat_setting['stride']), min_event_num=int(20/feat_setting['stride'])) #int(4/feat_setting['stride'])
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    pred_df = pd.concat([pred_df, prob_df[['label']]],axis=1)
    pred_df.to_csv(out_file)

def process_file_svm(file_name, thres=0.99, avg = True):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder,setting_folder,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name,index_col=0)
    prob_mat = prob_df[[c for c in prob_df.columns if c.startswith('nu_hat')]].values
    sz_prob = prob_mat.mean(axis=1)
    if avg:
        pred = detect_seizure(sz_prob, threshold=thres)
    else:
        pred_mat = (prob_mat >= thres).astype(int)
        # perc_chan = pred_mat.sum(axis=1)/pred_mat.shape[1]
        # pred = perc_chan >= 1
        pred = pred_mat.sum(axis=1) >= min(2,pred_mat.shape[1])
    pred = get_event_smoothed_pred(smooth_pred(pred), gap_num=int(4/feat_setting['stride']), min_event_num=int(20/feat_setting['stride'])) #int(4/feat_setting['stride'])
    pred = apply_persistence(pred) 
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    pred_df = pd.concat([pred_df, prob_df[['label']]],axis=1)
    pred_df.to_csv(out_file)
        
        


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Get sparcnet predictions from stored probability files, this step allow quick iteration and test on threshold values"
    )

    # Define arguments
    parser.add_argument("--model", type=str, default='sparcnet', help="Folder to store sparcnet prediction files, one file per edf file")
    # parser.add_argument("-o", "--output", type=str, default='sparcnet_results/pred', help="Folder to store sparcnet prediction files, one file per edf file")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montage code to run, e.g epiminder_2, if 'all', all available montages in montage_list")
    parser.add_argument("--force", action='store_true', help="Force re-running, even if there're already prediction files")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Threshold to appy")
    parser.add_argument("-s", "--setting", type=str, default='', help="A code for the current setting, default to none")
    parser.add_argument("--avg", action='store_true', help="Force re-running, even if there're already prediction files")
    
    # Parse the arguments
    params = vars(parser.parse_args())
    
    model = params['model']
    feat_setting = feat_settings[model]
    model_label_map = {'dynasd':'NDD','sparcnet':'SPaRCNet','svm':'SVM', 'feat':'Feat'}
    model_label = model_label_map[model]
    if model == 'dynasd':
        prob_folder = f"/mnt/sauce/littlab/users/joekoji/montage-proj/{model}_results_fixed/prob"
    elif model == 'sparcnet':
        prob_folder = f"/mnt/sauce/littlab/users/joekoji/montage-proj/{model}_results/prob"
    elif model == 'svm':
        prob_folder = f"/mnt/sauce/littlab/users/haoershi/limited_montage/{model}_results/prob"
    elif model == 'feat':
        prob_folder = f"/mnt/sauce/littlab/users/haoershi/limited_montage/{model}_results/prob"
    pred_folder = f"{params['model']}_results/pred"
    force = params['force']
    thres = params['thres']
    avg = params['avg']
    thres_folder = f"{model}_results/thres"
    os.makedirs(thres_folder, exist_ok=True)
    
    if params['montage'] == 'all':
        montage = montage_list
    else:
        montage = params['montage'].split(',')

    setting_folder = params['setting']
    threses = pd.read_csv('manuscript/f1_ver/threses_all.csv')#pd.read_csv('manuscript/f1_ver/threses_all.csv')
        

    all_threses = []
    for m in montage:
        prob_files = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        os.makedirs(os.path.join(pred_folder,setting_folder,m),exist_ok=True)
        if 'thres_optimal_f1' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_f1'].iloc[0]
            # thres = get_optimal_thres_f1(prob_files, feat_setting['stride'])
        elif 'thres_optimal' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_yodenj'].iloc[0]
        elif 'autothres' in setting_folder:
            thres = None
        elif 'thres_fixedfar_0.5' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_far0.5'].iloc[0]
        elif 'thres_fixedfar_1' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_far1'].iloc[0]
        elif 'thres_fixedfar_2' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_far2'].iloc[0]
        elif 'thres_fixedfar_5' in setting_folder:
            thres = threses[(threses['model']==model_label)&(threses['montage']==m)]['thres_far5'].iloc[0]
        else:
            pass
        all_threses.append({'montage':m,'thres':thres})
        pd.DataFrame(all_threses).to_csv(f'{thres_folder}/{setting_folder}.csv',index=False)
        print(f'Model {model}, montage {m}, using threshold {thres}')
        if model == 'dynasd':
            with tqdm(total=len(prob_files),desc = 'Processing file'):
                results = Parallel(n_jobs=40)(delayed(process_file_ndd)(file_name, thres, avg) for file_name in prob_files)
        elif model == 'sparcnet':
            with tqdm(total=len(prob_files),desc = 'Processing file'):
                results = Parallel(n_jobs=40)(delayed(process_file)(file_name, thres) for file_name in prob_files)
        elif model == 'svm':
            with tqdm(total=len(prob_files),desc = 'Processing file'):
                results = Parallel(n_jobs=40)(delayed(process_file_svm)(file_name, thres, avg) for file_name in prob_files)
        elif model == 'feat':
            with tqdm(total=len(prob_files),desc = 'Processing file'):
                results = Parallel(n_jobs=40)(delayed(process_file_feat)(file_name, thres) for file_name in prob_files)

