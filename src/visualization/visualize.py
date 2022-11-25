# Imports
from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pdb
import warnings
import matplotlib.lines as mlines

# tSNE and PCA plot 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# PIL
from PIL import Image, ImageFont, ImageDraw

# Own imports
from src.models.image_classifier import ImageClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)



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


def image_object_full_bbox_resize_class(path,bbox,img_size,t_class,p_class,probs,var=None):
    """
    This function is a generic PIL image loader of an object on
    an image with path 'path' and bounding box 'bbox'.
    The function returns the origin PIL image with bbox annotation,
    the bbox image, and the bbox image resized.
    The three images will be put side-by-side
    """
    
    img = Image.open(path).convert("RGB")
    img_bbox = img.crop(bbox)
    img_resize = img_bbox.resize((img_size,img_size))
    
    img_full = ImageDraw.Draw(img)
    img_full.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='Red', width=3)
    img_full = img_full._image  # type: ignore
    
    images = [img_full,img_bbox,img_resize]
    widths, heights = zip(*(i.size for i in images))

    total_width = int(sum(widths)*1.1)
    max_height = max(heights)
    
    probs_space = 0
    if probs is not None:
        probs_space = 200
        

    concat_img = Image.new('RGB', (max(total_width+20,1400)+probs_space, max_height+70))

    x_offset = 10 + max(int((1400-total_width-20)/2),0) + probs_space
    y_offset = 70
    x_space_between = int(np.floor((total_width-sum(widths))/2))
    for im in images:
        concat_img.paste(im, (x_offset,y_offset))
        x_offset += im.size[0]+x_space_between


    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', 40)
    # Add Text to an image
    concat_img = ImageDraw.Draw(concat_img)
    concat_img.text((10, 10), f"True class: {t_class}, Pred class: {p_class}, variance: {var:.4f}",
                    font=myFont, fill =(255, 255, 255))
    
    # add probs if parsed
    if probs is not None:
        myFont2 = ImageFont.truetype('FreeMono.ttf', 20)
        probs_top20 = probs.sort_values()[::-1][:20]
        probs_top20_idx = []
        for i in range(len(probs_top20)):
            if probs_top20.index[i] == t_class:
                probs_top20_idx.append('*'+probs_top20.index[i]+'*')
            else:
                probs_top20_idx.append(probs_top20.index[i])
        probs_top20.index = probs_top20_idx
        concat_img.text((10, 80), f"{probs_top20}", font=myFont2, fill =(255, 255, 255))
    concat_img = concat_img._image #type: ignore

    return concat_img


def visualize_embedding_space(classifier: ImageClassifier):
    # Extract embeddings for 30 objects from same class in embedding space
    num_classes_samples = 30  
    classes_unique = (np.sort(pd.unique(classifier.classes)))
    classes = []
    
    means_list = classifier.means.to('cpu').numpy()
    vars_list = classifier.vars.to('cpu').numpy()
    classes_list = np.array(classifier.classes)
    
    means = np.zeros((means_list.shape[0], num_classes_samples*len(classes_unique)))
    vars = np.zeros((num_classes_samples*len(classes_unique),))
    count = 0
    for class_ in classes_unique:
        class_idxs = np.where(classes_list == class_)[0]
        rand_idx_of_class = np.random.choice(class_idxs,num_classes_samples)
        for idx in rand_idx_of_class:
            means[:,count] = means_list[:,idx]
            vars[count] = np.mean(vars_list[:,idx])
            classes.append(classes_list[idx])
            count += 1
    
    
    
    # print embedding space
    fig_tsne_test, fig_pca_test = print_PCA_tSNE_plot(means,vars,classes,0,
                                                      classifier.params['var_prior'],'test')
    return fig_tsne_test, fig_pca_test


def visualize_var_dist_per_class(vars):
    for class_ in vars.keys():
        vars[class_] = list(vars[class_].values())
    
    # Create dataframe
    vars = (pd.DataFrame.from_dict(vars,'index').stack().reset_index(name='Variance')
                        .drop('level_1',1).rename(columns={'level_0':'Class'}))
    
    # Save fig
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    sns.boxplot(data=vars, x="Variance", y="Class", ax=ax)
    ax.set_title('Distribution of variances', fontsize=15)
    return fig


def calibration_plots(cali_plot_df_acc: pd.DataFrame,
                      cali_plot_df_conf: pd.DataFrame,
                      num_samples_bins: pd.DataFrame):
    num_classes = len(cali_plot_df_acc.columns)-2
    
    fig, axes = plt.subplots(nrows=int(num_classes/5)+1,ncols=5, figsize=(16,12),
                             sharex=True, sharey=True)
    
    size_min = 1
    size_max = 100
    # only these two lines are calibration curves
    for i, class_ in enumerate(cali_plot_df_acc.columns):
        ax = axes.flatten()[i]
        # get sizes 
        sizes = num_samples_bins[class_]
        sizes[sizes>100] = 100
        sizes = size_min + (size_max-size_min)*sizes/100
        sizes = sizes.astype(float)
        
        if (class_=='All') or (class_=='Class_mean'):
            color = '#008000'
        elif class_[-4:]=='_OOD':
            color = '#8C000F'
        else:
            color = '#1f77b4'
        x = cali_plot_df_conf[class_]
        y = cali_plot_df_acc[class_]
        std = cali_plot_df_acc[class_]*(1-cali_plot_df_acc[class_])/num_samples_bins[class_]
        std = np.sqrt(std.fillna(100))
        ax.plot(x,y,marker='o', linewidth=1, markersize=0,color=color)
        ax.scatter(x,y,marker='o',s=sizes, edgecolors='k', c=color,linewidth=0.5)
        ax.fill_between(x.tolist(), np.maximum((y-std).tolist(),0), 
                        np.minimum((y+std).tolist(),1),alpha=0.5,
                        edgecolor=color,facecolor=color,linestyle='-')

        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='--')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        ax.set_title(f'{class_}',fontsize=15)

    fig.supxlabel('Predicted probability',fontsize=20, y=0.03)
    fig.supylabel('True probability in each bin',fontsize=20, x=0.03)
    fig.suptitle('Calibration plots for each class',fontsize=20, y=0.96)
    
    legend_elements = [Line2D([0], [0], marker='o', color='k', label='#Samples = 1',
                                        markerfacecolor='#1f77b4', markersize=1,linewidth=0,
                                        markeredgewidth=0.5),
                       Line2D([0], [0], marker='o', color='k', label='#Samples = 50',
                                        markerfacecolor='#1f77b4', markersize=5,linewidth=0,
                                        markeredgewidth=0.5),
                       Line2D([0], [0], marker='o', color='k', label='#Samples > 100',
                                        markerfacecolor='#1f77b4', markersize=10,linewidth=0,
                                        markeredgewidth=0.5)]
    
    plt.subplots_adjust(left=0.1, right=0.82, top=0.9, bottom=0.1)
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.91, 0.55))
    
    # Add legend for coloring
    legend_elements2 = [Line2D([0], [0], marker='', color='#1f77b4', label='Classes ID',
                                        markerfacecolor='#1f77b4', markersize=0,linewidth=1,
                                        markeredgewidth=0),
                       Line2D([0], [0], marker='', color='#8C000F', label='Classes OOD',
                                        markersize=0,linewidth=1,markeredgewidth=0),
                       Line2D([0], [0], marker='', color='#008000', label='Classes combined',
                                        markersize=0,linewidth=1,markeredgewidth=0)]

    legend2 = fig.legend(legend_elements2,['Classes ID','Classes OOD','Classes combined'], 
                         loc='center', bbox_to_anchor=(0.91, 0.45))
    fig.add_artist(legend2)
    
    return fig


def precision_recall_plots(aP_plotting: pd.DataFrame,
                           aP_df: pd.DataFrame):
    num_classes = len(aP_plotting['precision'].keys())
    
    fig, axes = plt.subplots(nrows=int(np.ceil(num_classes/5)),ncols=5, figsize=(16,12),
                             sharex=True, sharey=True)
    
    # only these two lines are calibration curves
    for i, class_ in enumerate(aP_plotting['precision'].keys()):
        ax = axes.flatten()[i]
        
        if class_[-4:]=='_OOD':
            color = '#8C000F'
        else:
            color = '#1f77b4'
            
        ax.plot(aP_plotting['recall'][class_],aP_plotting['precision'][class_], linewidth=1, 
                markersize=0,color='k')
        ax.step(aP_plotting['rec_AUC'][class_],aP_plotting['prec_AUC'][class_],linewidth=2,
                markersize=0,color=color, where='pre')

        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0.5, 0.5], color='gray', linestyle='--', linewidth=1)
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        ax.set_title(f'{class_}',fontsize=15)
        ax.text(0, 0.03, f'aP = {aP_df[class_]:.3f}', fontsize=13)

    fig.supxlabel('Recall',fontsize=20, y=0.03)
    fig.supylabel('Precision',fontsize=20, x=0.03)
    fig.suptitle('Precision/recall plots',fontsize=20, y=0.96)
    
    # Add legend for coloring
    legend_elements  = [Line2D([0], [0], marker='', color='#1f77b4', label='Classes ID',
                                        markerfacecolor='#1f77b4', markersize=0,linewidth=2,
                                        markeredgewidth=0),
                       Line2D([0], [0], marker='', color='#8C000F', label='Classes OOD',
                                        markersize=0,linewidth=2,markeredgewidth=0),
                       Line2D([0], [0], marker='', color='gray', label='No skill',
                                        markersize=0,linewidth=1,markeredgewidth=0,
                                        linestyle='--')]
    
    plt.subplots_adjust(left=0.1, right=0.82, top=0.9, bottom=0.1)
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.91, 0.5))
    
    return fig


def UCE_plots(uncert_cali_plot_df: pd.DataFrame):
    # Plot UCE plot
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(12,6))
    
    size_min = 1
    size_max = 100
    # only these two lines are calibration curves
    sizes = uncert_cali_plot_df['Count']
    sizes = (sizes-np.min(sizes))/(np.max(sizes)-np.min(sizes))*100
    sizes = size_min + (size_max-size_min)*sizes/100
    sizes = sizes.astype(float)
    color = '#008000'
    x = uncert_cali_plot_df['Uncertainty']
    y = uncert_cali_plot_df['True error']
    
    std = (uncert_cali_plot_df['True error']*(1-uncert_cali_plot_df['True error']) /
           uncert_cali_plot_df['Count'])
    y_lower = y-np.sqrt(std.fillna(100))
    y_upper = y+np.sqrt(std.fillna(100))

    ax.flatten()[0].plot(x,y, marker='o', linewidth=1, markersize=0,color=color)
    ax.flatten()[0].scatter(x,y,marker='o',s=sizes, edgecolors='k', c=color, linewidth=0.5)
    ax.flatten()[0].fill_between(x.tolist(), np.maximum(y_lower.tolist(),0), 
                                 np.minimum(y_upper.tolist(),1),alpha=0.5,
                                 edgecolor=color,facecolor=color,linestyle='-')
    ax.flatten()[1].plot(x,y, marker='o', linewidth=1, markersize=0,color=color)
    ax.flatten()[1].scatter(x,y,marker='o',s=sizes, edgecolors='k', c=color, linewidth=0.5)
    ax.flatten()[1].fill_between(x.tolist(), np.maximum(y_lower.tolist(),0), 
                                 np.minimum(y_upper.tolist(),1),alpha=0.5,
                                 edgecolor=color,facecolor=color,linestyle='-')
    
    # Add reference lines
    y2 = uncert_cali_plot_df['Optimal']
    ax.flatten()[0].plot(x,y2, linewidth=1, color='black', linestyle='--', markersize=0)
    ax.flatten()[1].plot(x,y2, linewidth=1, color='black', linestyle='--', markersize=0)
    ax.flatten()[1].set_yscale('log')

    fig.supxlabel('Uncertainty',fontsize=20, y=0.03)
    fig.supylabel('True error',fontsize=20, x=0.03)
    fig.suptitle('Uncertainty calibration plot',fontsize=20, y=0.96)
    
    legend_elements = [Line2D([0], [0], marker='o', color='k', label='#Samples < 100',
                                        markerfacecolor='#008000', markersize=1,linewidth=0,
                                        markeredgewidth=0.5),
                       Line2D([0], [0], marker='o', color='k', label='#Samples = 500',
                                        markerfacecolor='#008000', markersize=5,linewidth=0,
                                        markeredgewidth=0.5),
                       Line2D([0], [0], marker='o', color='k', label='#Samples > 1000',
                                        markerfacecolor='#008000', markersize=10,linewidth=0,
                                        markeredgewidth=0.5)]
    
    plt.subplots_adjust(left=0.1, right=0.82, top=0.86, bottom=0.14)
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.91, 0.6))
    
     # Add legend for coloring
    legend_elements2 = [Line2D([0], [0], marker='', color='black', label='Optimal Calibration',
                                        markerfacecolor='#1f77b4', markersize=0,linewidth=1,
                                        markeredgewidth=0, linestyle='--'),
                       Line2D([0], [0], marker='', color='#008000', label='Model',
                                        markersize=0,linewidth=1,markeredgewidth=0)]

    legend2 = fig.legend(legend_elements2,['Optimal Calibration','Model'], 
                         loc='center', bbox_to_anchor=(0.91, 0.4))
    fig.add_artist(legend2)
    
    return fig


def visualize_embedding_space_with_images(classifier: ImageClassifier):
    # Extract embeddings for 10 objects from same class in embedding space
    num_classes_samples = 3
    classes_unique = (np.sort(pd.unique(classifier.classes)))
    classes = []
    
    objects_list = classifier.objects
    bboxs_list = classifier.bbox
    means_list = classifier.means.to('cpu').numpy()
    vars_list = classifier.vars.to('cpu').numpy()
    classes_list = np.array(classifier.classes)
    
    means = np.zeros((means_list.shape[0], num_classes_samples*len(classes_unique)))
    objects = []
    bboxs = []
    
    count = 0
    for class_ in classes_unique:
        class_idxs = np.where(classes_list == class_)[0]
        rand_idx_of_class = np.random.choice(class_idxs,num_classes_samples)
        for idx in rand_idx_of_class:
            means[:,count] = means_list[:,idx]
            classes.append(classes_list[idx])
            objects.append(objects_list[idx])
            bboxs.append(bboxs_list[idx])
            count += 1
    
    
    path_to_img = './data/raw/JPEGImages/'
    fig1, fig2 = print_tSNE_plot_with_images(means, objects, bboxs, classes, 
                                             classifier.params['img_size'], path_to_img)

    return fig1, fig2


