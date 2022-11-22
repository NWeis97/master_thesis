from typing import List
import matplotlib.pyplot as plt
import shortuuid
import logging
import os
import configparser
from PIL import Image, ImageDraw
from matplotlib.lines import Line2D

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
warnings.simplefilter("ignore", UserWarning)

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
                        varP: float, mode: str, size: int=100, perp: int=15, n_iter: int=2000,
                        lr: int=200, early_ex: int=10, init: str='pca'):
    """This function makes a scatterplot of the embedding space of *data* with pseudo-variances
       *qvars*
       The classes are given by a list and epoch defines the epoch at call-time (for naming).

    Args:
        data (np.array): A (out_dim-1,num_quries) array with embedding means
        qvars (np.array): A (num_quries,) array with embedding variances
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
    if (mode != 'train') & (mode != 'val') & (mode != 'test'):
        raise ValueError('Unknown mode!')
    
    # Marker sizes
    sizes = np.array([size]*len(classes))
    df_subset = pd.DataFrame()
    df_subset['size'] = sizes
    df_subset['class'] = classes
    
    max_size = 10*size*2
    min_size = size/4*2
    df_subset['var'] = qvars/varP
    df_subset['var'] = (df_subset['var']-np.min(df_subset['var']))/(np.max(df_subset['var'])-
                                                                    np.min(df_subset['var']))
    df_subset['var'] = min_size + df_subset['var'] * (max_size-min_size)

    # Marker types
    marker_t = []
    for i in range(len(classes)):
        if '_OOD' in classes[i]:
            marker_t.append('OOD')
        else:
            marker_t.append('ID')
    df_subset['type'] = marker_t
    markers = {"ID": "o", "OOD": "s"}
    
    # Palette
    if len(pd.unique(classes)) <= 10:
        palette = sns.color_palette("hls",10)
    elif len(pd.unique(classes)) <= 20:
        pal1 = sns.color_palette("hls",10)
        pal2 = sns.color_palette("husl",10)
        palette = []
        for i in range(10):
            palette.append(pal1[i])
            palette.append(pal2[i])
    else:
        palette = sns.color_palette("hls", len(pd.unique(classes)))
        
    
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
    g=sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=palette,
        data=df_subset,
        legend="full",
        style="type",
        alpha=1,
        s=size,
        ax=ax,
        markers=markers
    )
    g.legend(loc='center left', bbox_to_anchor=(2.5, 0.5), ncol=1)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=palette,
        data=df_subset,
        alpha=0.5,
        s=df_subset['var'],
        style="type",
        ax=ax,
        legend = False,
        markers=markers
    )
    ax.set_xlabel('tSNE-2d-one', fontsize=15)
    ax.set_ylabel('tSNE-2d-two', fontsize=15)
    if mode == 'test':
        ax.set_title(f'tSNE test (var_prior={varP:.3f})', fontsize=20)
    else:
        ax.set_title(f'tSNE epoch {epoch} (var_prior={varP:.3f})', fontsize=20)
    ax.legend(fontsize=15)
    
    # Move legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])  
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    plt.close()
    
    #PCA
    fig2, ax2 = plt.subplots(1,1,figsize=(12,7))
    g2 = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="class",
        palette=palette,
        data=df_subset,
        legend="full",
        style="type",
        alpha=1,
        s=size,
        ax=ax2,
        markers=markers
    )
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="class",
        palette=palette,
        data=df_subset,
        alpha=0.5,
        s=df_subset['var'],
        style="type",
        ax=ax2,
        legend = False,
        markers=markers
    )
    ax2.set_xlabel('PCA-2d-one', fontsize=15)
    ax2.set_ylabel('PCA-2d-two', fontsize=15)
    if mode == 'test':
        ax2.set_title(f'PCA test (var_prior={varP:.3f})', fontsize=20)
    else:
        ax2.set_title(f'PCA epoch {epoch} (var_prior={varP:.3f})', fontsize=20)
    ax2.legend(fontsize=15)
    
    # Move legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])  
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    
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


def print_tSNE_plot_with_images(data: np.array, objects: list, bbox: list, classes: list, 
                                img_size: int, path: str, size: float=0.15, perp: int=15, 
                                n_iter: int=2000, lr: int=200, early_ex: int=10, init: str='pca'):
    """This function makes a scatterplot of the embedding space of *data* with pseudo-variances
       *qvars*
       The classes are given by a list and epoch defines the epoch at call-time (for naming).

    Args:
        data (np.array): A (out_dim-1,num_quries) array with embedding means
        objects (list): A (num_quries,) list with object names
        bbox (list): A (num_quries,) list with object names
        classes (list): List of classes with length *num_quries*
        img_size (int): size of image
        path (str): path to images
        size (float, optional): Size of images in plot. Default 0.15 of axis limits
        perp (int, optional): tSNE optional. Defaults to 10.
        n_iter (int, optional): tSNE optional. Defaults to 1000.
        lr (int, optional): tSNE optional. Defaults to 200.
        early_ex (int, optional): tSNE optional. Defaults to 12.
        init (str, optional): tSNE optional. Defaults to 'pca'.
    """

    # Marker sizes
    df_subset = pd.DataFrame()

    # Marker types
    marker_t = []
    for i in range(len(classes)):
        if '_OOD' in classes[i]:
            marker_t.append('OOD')
        else:
            marker_t.append('ID')
    df_subset['type'] = marker_t
    markers = {"ID": "s", "OOD": "o"}
    
    # Palette
    if len(pd.unique(classes)) <= 10:
        palette = sns.color_palette("hls",10)
    elif len(pd.unique(classes)) <= 20:
        pal1 = sns.color_palette("hls",10)
        pal2 = sns.color_palette("husl",10)
        palette = []
        for i in range(10):
            palette.append(pal1[i])
            palette.append(pal2[i])
    else:
        palette = sns.color_palette("hls", len(pd.unique(classes)))
        
    unique_classes = np.unique(classes)
    class_to_color_map = {unique_classes[i]:i for i in range(len(unique_classes))}
    
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
    
    # randomize 
    rand_idx = np.random.permutation(range(len(objects)))

    # tSNE plot
    fig, ax = plt.subplots(1,1,figsize=(12,7))
    min_x = min(df_subset['tsne-2d-one'])*1.15
    max_x = max(df_subset['tsne-2d-one'])*1.15
    min_y = min(df_subset['tsne-2d-two'])*1.15
    max_y = max(df_subset['tsne-2d-two'])*1.15
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)
    
    x_axis_range = max_x-min_x
    y_axis_range = max_y-min_y
    
    for i in rand_idx:
        color = palette[class_to_color_map[classes[i]]]
        color = tuple(np.round(np.array(color)*256).astype(int).tolist())
        img = Image.open(path+'/'+objects[i]).convert("RGB")
        img_bbox = img.crop(bbox[i])
        img_resize = img_bbox.resize((img_size,img_size))
        img_full = ImageDraw.Draw(img_resize)
        img_full.rectangle(((0, 0), (img_size, img_size)), fill=None, outline=color, width=10)
        img_full = img_full._image  # type: ignore
        
        c_x = (df_subset['tsne-2d-one'][i]-min_x)/x_axis_range*0.8+0.025
        c_y = (df_subset['tsne-2d-two'][i]-min_y)/y_axis_range*0.8+0.025
        
        newax = fig.add_axes([c_x-size/2,c_y-size/2,size,size], anchor='NE', zorder=1)
        newax.imshow(img_full)
        newax.axis('off')
        
        
    ax.set_xlabel('tSNE-2d-one', fontsize=15)
    ax.set_ylabel('tSNE-2d-two', fontsize=15)
    ax.set_title(f'tSNE embedding space with images', fontsize=20)
    ax.axis('off')
    
    # Move legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])  
    
    legend_elements = []
    for i in range(len(unique_classes)):
        col = palette[class_to_color_map[unique_classes[i]]]
        line = Line2D([0], [0], color=col, label=unique_classes[i], markersize=0,linewidth=3)
        legend_elements.append(line)
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.91, 0.55))

    plt.close()
    
    # PCA plot
    fig2, ax2 = plt.subplots(1,1,figsize=(12,7))
    min_x = min(df_subset['pca-one'])*1.15
    max_x = max(df_subset['pca-one'])*1.15
    min_y = min(df_subset['pca-two'])*1.15
    max_y = max(df_subset['pca-two'])*1.15
    ax2.set_xlim(min_x,max_x)
    ax2.set_ylim(min_y,max_y)
    
    x_axis_range = max_x-min_x
    y_axis_range = max_y-min_y
    
    for i in rand_idx:
        color = palette[class_to_color_map[classes[i]]]
        color = tuple(np.round(np.array(color)*256).astype(int).tolist())
        img = Image.open(path+'/'+objects[i]).convert("RGB")
        img_bbox = img.crop(bbox[i])
        img_resize = img_bbox.resize((img_size,img_size))
        img_full = ImageDraw.Draw(img_resize)
        img_full.rectangle(((0, 0), (img_size, img_size)), fill=None, outline=color, width=10)
        img_full = img_full._image  # type: ignore
        
        c_x = (df_subset['pca-one'][i]-min_x)/x_axis_range*0.8+0.025
        c_y = (df_subset['pca-two'][i]-min_y)/y_axis_range*0.8+0.025
        
        newax = fig2.add_axes([c_x-size/2,c_y-size/2,size,size], anchor='NE', zorder=1)
        newax.imshow(img_full)
        newax.axis('off')
        
        
    ax2.set_xlabel('PCA-one', fontsize=15)
    ax2.set_ylabel('PCA-two', fontsize=15)
    ax2.set_title(f'PCA embedding space with images', fontsize=20)
    ax2.axis('off')
    
    # Move legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])  
    
    legend_elements = []
    for i in range(len(unique_classes)):
        col = palette[class_to_color_map[unique_classes[i]]]
        line = Line2D([0], [0], color=col, label=unique_classes[i], markersize=0,linewidth=3)
        legend_elements.append(line)
    
    fig2.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.91, 0.55))

    plt.close()
    
    return fig, fig2