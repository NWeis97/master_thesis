from typing import List
import matplotlib.pyplot as plt
import shortuuid
import logging
import os
import configparser

# tSNE and PCA plot 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import pdb
import warnings
import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_PCA_tSNE_plot(data: np.array ,qvars: np.array, classes: list, epoch: int,
                        varP: float, mode: str, size: int=100, perp: int=10, n_iter: int=1000,
                        lr: int=200, early_ex: int=12, init: str='pca'):
    """This function makes a scatterplot of the embedding space of *data* with pseudo-variances
       *qvars*
       The classes are given by a list and epoch defines the epoch at call-time (for naming).

    Args:
        data (np.array): A (out_dim-1,num_quries) tensor with embedding means
        qvars (np.array): A (num_quries,) tensor with embedding variances
        classes (list): List of classes with length *num_quries*
        epoch (int): Epoch number at call-time
        mode (str): train/val for saving-location
        size (int, optional): Size of dots in scatterplot. Defaults to 100.
        perp (int, optional): tSNE optional. Defaults to 10.
        n_iter (int, optional): tSNE optional. Defaults to 1000.
        lr (int, optional): tSNE optional. Defaults to 200.
        early_ex (int, optional): tSNE optional. Defaults to 12.
        init (str, optional): tSNE optional. Defaults to 'pca'.
    """
    if (mode != 'train') & (mode != 'val'):
        raise ValueError('Unknown mode!')
    
    sizes = np.array([size]*len(classes))
    df_subset = pd.DataFrame()
    df_subset['size'] = sizes
    df_subset['y'] = classes
    df_subset['var'] = qvars/varP

    # get tSNE embeddings
    tsne = TSNE(n_components=2, verbose=0, perplexity=perp,
                n_iter=n_iter,init=init,learning_rate=lr,early_exaggeration=early_ex,)
    tsne_results = tsne.fit_transform(data.T)
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    # PCA components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.T)
    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1]

    # tSNE plot
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(pd.unique(classes))),
        data=df_subset,
        legend="full",
        alpha=1,
        s=size,
        ax=ax
    )
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(pd.unique(classes))),
        data=df_subset,
        alpha=0.5,
        s=df_subset['var']*size*2,
        ax=ax,
        legend = False
    )
    ax.set_xlabel('tSNE-2d-one', fontsize=15)
    ax.set_ylabel('tSNE-2d-two', fontsize=15)
    ax.set_title(f'tSNE epoch {epoch}', fontsize=20)
    ax.legend(fontsize=15)

    plt.close()
    
    #PCA
    fig2, ax2 = plt.subplots(1,1,figsize=(12,7))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", len(pd.unique(classes))),
        data=df_subset,
        legend="full",
        alpha=1,
        s=size,
        ax=ax2
    )
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", len(pd.unique(classes))),
        data=df_subset,
        alpha=0.5,
        s=df_subset['var']*size*2,
        ax=ax2,
        legend = False
    )
    ax2.set_xlabel('PCA-2d-one', fontsize=15)
    ax2.set_ylabel('PCA-2d-two', fontsize=15)
    ax2.set_title(f'PCA epoch {epoch}', fontsize=20)
    ax2.legend(fontsize=15)
    
    plt.close()
    return fig, fig2


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]], [batch[0][2]]
    return ([batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))],
            [batch[i][2] for i in range(len(batch))])
    
    
def check_config(config):
    """This function checks if the specific config being run at call-time has been run before.
       If so, return the name of said config file, otherwise generate new filename and save
       config to version history.

    Args:
        config (config): Config file of type configParser

    Returns:
        config_filename (str): name of either existing config file or new config file.
        config_run_no (int): number of times this config has been run
    """
    # get logger
    logger = logging.getLogger('__main__')
    
    # assign directory
    directory = './configs/configs_hist'
    
    # bool for checking if config already has been run
    config_exists = False
    
    # iterate over configs in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            config_hist =  configparser.ConfigParser()
            config_hist.read(f)
            break_outer = False
            
            for key1 in config:
                for key2 in config[key1]:
                    try:
                        if config[key1][key2] != config_hist[key1][key2]:
                            break_outer = True
                            break;
                    except:
                        break_outer = True
                        break;
                
                if break_outer:
                    break;

            if break_outer == False:
                config_exists = True
                break;
     
    if config_exists:
        
        config_filename = filename[:-4]
        
        # Define config run number
        config_run_no = 0
        while True:
            if os.path.exists(f'./logs/training_test/train_model/{str(config_filename)}'+
                            f'/run_{str(config_run_no)}.log'):
                config_run_no += 1
            else:
                break;
    else:
        config_filename = str(shortuuid.uuid())
        os.makedirs(f'./logs/training_test/train_model/{str(config_filename)}',exist_ok=True) 
        with open(f'configs/configs_hist/{config_filename}.ini', 'w') as configfile:
            config.write(configfile)  
        config_run_no = 0
    
    return config_filename, config_run_no
    
    
def get_logger(config_filename: str, config_run_no: str):
    """This function defines and returns logger that save output to file in path:
       './logs/training_test/train_model/{config_filename}/run_{config_run_no}.logs'

    Args:
        mode (str): training or testing
        config_filename (str): config filename
        config_run_no (str): config run number

    Returns:
       logger (Logger): A logger for logging
    """
    # Define logger
    log_fmt = '%(message)s'
    log_file_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger() 
    file_handler = logging.FileHandler(f'./logs/training_test/train_model/{str(config_filename)}/'+
                                       f'run_{str(config_run_no)}.log')
    file_handler.setFormatter(logging.Formatter(log_file_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    
    return logger



def get_logger_test(name: str):
    """This function defines and returns logger that save output to file in path:
       './logs/training_test/test_model/{name}.logs'

    Args:
        name (str): name of test

    Returns:
       logger (Logger): A logger for logging
    """
    # Define logger
    log_fmt = '%(message)s'
    log_file_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger() 
    file_handler = logging.FileHandler(f'./logs/training_test/test_model/{str(name)}.log')
    file_handler.setFormatter(logging.Formatter(log_file_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    
    return logger