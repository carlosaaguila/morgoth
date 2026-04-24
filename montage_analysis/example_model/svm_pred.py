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


feat_setting = {'name':'sparcnet',
                'win':int(10), 'stride':int(2),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40} # in seconds


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
                # 'epiminder_4',
                # # 'epiminder_simulate',
                # 'zero'
                ]

def process_file(file_name):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder,setting_folder,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name,index_col=0)
    sz_prob = prob_df['sz_prob'].values
    pred = get_prediction(sz_prob, fs=1)
    # func = lambda x: ((x[:,1] >= thres).astype(int), x[:,1])
    # pred, sz_prob = func(prob)
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    # pred_df['smoothed_pred'] = get_event_smoothed_pred(smooth_pred(pred_df['pred'].values)) 
    pred_df = pd.concat([pred_df, prob_df.iloc[:,-1]],axis=1)
    pred_df.to_csv(out_file)

def get_optimal_thres(prob_files):
    all_prob = []
    all_label = []
    for f in prob_files:
        prob_df = pd.read_csv(f, index_col=0)
        sz_prob = prob_df.iloc[:,1].values
        label = prob_df.iloc[:,-1].values
        all_prob.extend(sz_prob)
        all_label.extend(label)
    fpr, tpr, thres = roc_curve(all_label, all_prob)
    opt_thres = thres[np.argmax(tpr-fpr)]
    return opt_thres

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Get sparcnet predictions from stored probability files, this step allow quick iteration and test on threshold values"
    )

    # Define arguments
    parser.add_argument("-o", "--output", type=str, default='svm_results/pred', help="Folder to store sparcnet prediction files, one file per edf file")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montage code to run, e.g epiminder_2, if 'all', all available montages in montage_list")
    parser.add_argument("--force", action='store_true', help="Force re-running, even if there're already prediction files")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Threshold to appy")
    parser.add_argument("-s", "--setting", type=str, default='', help="A code for the current setting, default to none")
    
    # Parse the arguments
    params = vars(parser.parse_args())

    prob_folder = '/mnt/sauce/littlab/users/joekoji/montage-proj/svm_results/prob'
    pred_folder = params['output']
    force = params['force']
    thres = params['thres']
    
    if params['montage'] == 'all':
        montage = montage_list
    else:
        montage = params['montage'].split(',')

    setting_folder = f"autothres"
    if params['setting']:
        setting_folder += f"_{params['setting']}"
        if params['setting'] == 'optimal':
            setting_folder = 'thres_optimal'

    for m in montage:
        prob_files = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        os.makedirs(os.path.join(pred_folder,setting_folder,m),exist_ok=True)
        with tqdm(total=len(prob_files),desc = 'Processing file'):
            results = Parallel(n_jobs=40)(delayed(process_file)(file_name) for file_name in prob_files)
        
