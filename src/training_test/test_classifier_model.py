from turtle import color
from src.models.image_classifier import ImageClassifier
import wandb
import os
import pdb
import argparse
import numpy as np
import pandas as pd
import ast
from numpy import linalg as LA
from src.utils.helper_functions import print_PCA_tSNE_plot
from src.loaders.generic_loader import image_object_full_bbox_resize_class
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import seaborn as sns



def main(args):
    # Extract args
    classifier_model = args.model_name
    database_model = args.model_database
    balanced_classes = int(args.balanced_dataset)
    test_dataset = args.test_dataset
    num_NN = int(args.num_NN)
    num_MC = int(args.num_MC)
    method = args.method
    with_OOD = ast.literal_eval(args.with_OOD)
    
    # Init wandb
    wandb.init(
                project="master_thesis",
                entity="nweis97",
                config={"classifier_model": classifier_model, 
                        "database_model": database_model,
                        "balanced_classes": balanced_classes,
                        "test_dataset": test_dataset,
                        "num_NN": num_NN,
                        "num_MC": num_MC,
                        "method": method},
                job_type="Test"
        )
    
    # Extract classifier model
    classifier = ImageClassifier(classifier_model,database_model,balanced_classes,with_OOD)
    
    # Calculate probabilities
    (probs_df, 
     objects, 
     bboxs, 
     true_classes) = classifier.get_probability_dist_dataset(test_dataset,num_NN,num_MC,method)
    
    # Get measures
    MaP, aP_df, aP_plotting = classifier.calc_MaP(probs_df,true_classes)
    acc_top1, acc_top2, acc_top3, acc_top5, acc_df = classifier.calc_accuracy(probs_df,true_classes)
    (cali_plot_df, ECE_class_mean, ECE_class_mean_m0, SCE, 
                   SCE_m0, CECE, CECE_df, num_samples_bins) = classifier.calc_ECE(probs_df,true_classes)
    
    # Log simple measures
    wandb.log({'MaP':MaP,
               'Accuracy top1': acc_top1,
               'Accuracy top2': acc_top2,
               'Accuracy top3': acc_top3,
               'Accuracy top5': acc_top5,
               'ECE_class_mean': ECE_class_mean,
               'SCE': SCE,
               'ECE_class_mean_m0': ECE_class_mean_m0,
               'SCE_m0': SCE_m0,
               'CECE': CECE,})
 
    # Log class specific metrics
    class_metrics = pd.DataFrame({'Average precision': aP_df,
                                  'Class Expected Calibration Error': CECE_df},
                                 index=aP_df.index)
    class_metrics = pd.concat([class_metrics,acc_df],axis=1)
    class_metrics.loc['**mean**'] = class_metrics.mean()
    class_metrics = class_metrics.reset_index()
    
    wandb.log({'Class Metrics': wandb.Table(dataframe=class_metrics),
               'Class Calibration': wandb.Table(dataframe=cali_plot_df.reset_index())})
    
    # Get calibration plots
    cali_plots = calibration_plots(cali_plot_df,num_samples_bins)
    
    # Log illustration of embedding space with classes
    fig_tsne_test, fig_pca_test = visualize_embedding_space(classifier)
    wandb.log({'fig_tsne_test': wandb.Image(fig_tsne_test),
               'fig_pca_test' : wandb.Image(fig_pca_test),
               'Calibration plots': wandb.Image(cali_plots)})
    
    
    # Get images for each class of for correct is misclassied images
    path_to_img = './data/raw/JPEGImages/'
    pred_classes = probs_df.idxmax()
    idx_true, idx_false = get_idxs_of_true_and_false_classifications(pred_classes, true_classes)
              
    for i in range(len(idx_true)):   
        i_true = idx_true[i]
        i_false = idx_false[i]    
        _, img_true_var = classifier.get_embedding(objects[i_true],bboxs[i_true])  
        _, img_false_var = classifier.get_embedding(objects[i_false],bboxs[i_false])  
        img_true = image_object_full_bbox_resize_class(path_to_img+objects[i_true],
                                                       bboxs[i_true],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_true],
                                                       pred_classes[i_true],
                                                       probs_df.iloc[:,i_true],
                                                       img_true_var)
        img_false = image_object_full_bbox_resize_class(path_to_img+objects[i_false],
                                                       bboxs[i_false],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_false],
                                                       pred_classes[i_false],
                                                       probs_df.iloc[:,i_false],
                                                       img_false_var)
        
        wandb.log({'Correct classification': wandb.Image(img_true),
                   'Misclassification': wandb.Image(img_false)})
    
    
    # Extract variances for dataset
    _, vars = classifier.get_embeddings_of_dataset(test_dataset,num_NN,num_MC,method)
    dict_vars = sort_idx_on_var_per_class(vars, true_classes)
    
    for class_ in dict_vars:
        # sample one image with small (top 10%) and large (top 10%) var for each class
        idx_class = list(dict_vars[class_].keys())
        idx_num_small = np.random.randint(0,int(len(idx_class)/10),1)
        idx_num_large = np.random.randint(0,int(len(idx_class)/10),1)
        i_small = idx_class[idx_num_small.item()]
        i_large = idx_class[-idx_num_large.item()-1] 
        small_var = dict_vars[class_][i_small]  
        large_var = dict_vars[class_][i_large]  
        img_small_var = image_object_full_bbox_resize_class(path_to_img+objects[i_small],
                                                       bboxs[i_small],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_small],
                                                       pred_classes[i_small],
                                                       probs_df.iloc[:,i_small],
                                                       small_var)
        img_large_var = image_object_full_bbox_resize_class(path_to_img+objects[i_large],
                                                       bboxs[i_large],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_large],
                                                       pred_classes[i_large],
                                                       probs_df.iloc[:,i_large],
                                                       large_var)
        
        wandb.log({'Small variance examples': wandb.Image(img_small_var),
                   'Large variance examples': wandb.Image(img_large_var)})
    
    fig_vars = visualize_var_dist_per_class(dict_vars)
    wandb.log({'Distribution over variances': wandb.Image(fig_vars)})
    
    # Create precision/recall plots
    fig_aP = precision_recall_plots(aP_plotting, aP_df)
    wandb.log({'Preicison/recall plots': wandb.Image(fig_aP)})
    
    # Confidence vs. accuracy table
    conf_vs_acc_df = confidence_vs_accuracy(probs_df, true_classes)
    for i in range(len(conf_vs_acc_df['Confidence'])):
        wandb.log({'Confidence vs. accuracy_conf': conf_vs_acc_df['Confidence'][i],
                   'Confidence vs. accuracy_acc': conf_vs_acc_df['Accuracy'][i],
                   'Confidence vs. accuracy_count': conf_vs_acc_df['Count'][i]})
    
    # Accurate when certain table
    acc_cert_df = accruate_when_certain(probs_df, true_classes)
    for i in range(len(acc_cert_df['uncertainty'])):
        wandb.log({'Accurate when certain_unc_thresh': acc_cert_df['uncertainty'][i],
                   'Accurate when certain_acc': acc_cert_df['Accuracy'][i],
                   'Accurate when certain_count': acc_cert_df['Count'][i]})
    
    # Uncertain when inaccurate
    uncert_inacc_df = uncertain_when_inaccurate(probs_df, true_classes)
    for i in range(len(uncert_inacc_df['uncertainty'])):
        wandb.log({'Uncertain when inaccurate_thresh_unc': uncert_inacc_df['uncertainty'][i],
                   'Uncertain when inaccurate_frac_unc': uncert_inacc_df['Frac_uncertain'][i]})
    

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
            vars[count] = vars_list[idx]
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
    
    
    

def calibration_plots(cali_plot_df: pd.DataFrame,
                      num_samples_bins: pd.DataFrame):
    num_classes = len(cali_plot_df.columns)-2
    
    fig, axes = plt.subplots(nrows=int(num_classes/5)+1,ncols=5, figsize=(12,12),
                             sharex=True, sharey=True)
    
    x = cali_plot_df.index.tolist()
    
    size_min = 1
    size_max = 100
    # only these two lines are calibration curves
    for i, class_ in enumerate(cali_plot_df.columns):
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
        ax.plot(x,cali_plot_df[class_], marker='o', linewidth=1, markersize=0,color=color)
        ax.scatter(x,cali_plot_df[class_],marker='o',s=sizes, edgecolors='k', c=color, 
                                                              linewidth=0.5)

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
    
    fig, axes = plt.subplots(nrows=int(np.ceil(num_classes/5)),ncols=5, figsize=(12,12),
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
                markersize=0,color=color, where='post')

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
    



def get_idxs_of_true_and_false_classifications(pred_classes: list, true_classes: list):
    img_true = []
    img_false = []
    idx_true = []
    idx_false = []
    for j in range(len(pred_classes)):
        i = int(np.random.randint(0,len(pred_classes),1))
        class_ = true_classes[i]
        if pred_classes[i] == class_:
            if class_ not in img_true:
                img_true.append(class_)
                idx_true.append(i)
        else:
            if class_ not in img_false:
                img_false.append(class_)
                idx_false.append(i)
                
    idx_true_sorted = np.argsort(img_true)  
    idx_false_sorted = np.argsort(img_false)  
    idx_true = np.array(idx_true)[idx_true_sorted]
    idx_false = np.array(idx_false)[idx_false_sorted]
    
    return idx_true, idx_false
    



def sort_idx_on_var_per_class(vars, classes):
    classes_unique = np.sort(np.unique(classes))
    dict_vars = {class_:{} for class_ in classes_unique}
    for i, class_ in enumerate(classes):
        dict_vars[class_][i]=vars[i].item()
        
    for key in dict_vars.keys():
        dict_vars[key] = dict(sorted(dict_vars[key].items(), key=lambda item: item[1]))
    
    return dict_vars



def confidence_vs_accuracy(probs_df: pd.DataFrame,
                           true_classes: list):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    pred_probs = probs_df.max(0)
    probs_unique = np.array([0]+np.unique(pred_probs).tolist())
    
    conf_acc_df = pd.DataFrame(index=probs_unique)
    conf_acc_df['Accuracy'] = np.zeros((len(probs_unique),))
    conf_acc_df['Count'] = np.zeros((len(probs_unique),))
    for prob in probs_unique:
        idx_conf_true = pred_probs >= prob
        idx_prob_true = pred_classes[idx_conf_true] == true_classes[idx_conf_true]
        count = np.sum(idx_conf_true)
        conf_acc_df.loc[prob,'Accuracy'] = np.sum(idx_prob_true)/count
        conf_acc_df.loc[prob,'Count'] = count
    
    conf_acc_df = conf_acc_df.reset_index().rename(columns={'index':'Confidence'})
    return conf_acc_df


def accruate_when_certain(probs_df: pd.DataFrame,
                          true_classes: list):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    uncertainty = []
    for i in range(len(probs_df.columns)):
        ui = np.abs(-np.sum(probs_df.iloc[:,i]*np.log(probs_df.iloc[:,i])))
        uncertainty.append(ui)
    
    uncertainty = np.array(uncertainty)
    uncertainty_unique = np.unique(uncertainty)
    
    acc_cert_df = pd.DataFrame({'uncertainty':uncertainty_unique})
    acc_cert_df['Accuracy'] = np.zeros((len(uncertainty_unique),))
    acc_cert_df['Count'] = np.zeros((len(uncertainty_unique),))

    for i,u in enumerate(uncertainty_unique):
        idx_unc_true = uncertainty <= u
        idx_prob_true = pred_classes[idx_unc_true] == true_classes[idx_unc_true]
        count = np.sum(idx_unc_true)
        acc_cert_df.loc[i,'Accuracy'] = np.sum(idx_prob_true)/count
        acc_cert_df.loc[i,'Count'] = count
    
    return acc_cert_df


def uncertain_when_inaccurate(probs_df: pd.DataFrame,
                              true_classes: list):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    uncertainty = []
    for i in range(len(probs_df.columns)):
        ui = np.abs(-np.sum(probs_df.iloc[:,i]*np.log(probs_df.iloc[:,i])))
        uncertainty.append(ui)
    
    idx_inacc = pred_classes != true_classes
    uncertainty = np.array(uncertainty)[idx_inacc]
    uncertainty_unique = np.unique(uncertainty)
    
    acc_cert_df = pd.DataFrame({'uncertainty':uncertainty_unique})
    acc_cert_df['Frac_uncertain'] = np.zeros((len(uncertainty_unique),))
    acc_cert_df['Count'] = np.zeros((len(uncertainty_unique),))

    for i,u in enumerate(uncertainty_unique):
        idx_unc_true = uncertainty >= u
        count = np.sum(idx_inacc)
        acc_cert_df.loc[i,'Frac_uncertain'] = np.sum(idx_unc_true)/count
        acc_cert_df.loc[i,'Count'] = count
    
    return acc_cert_df


if __name__ == '__main__':
    
    # Get configs
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--model-name", help="Name of model to test")
    args_parser.add_argument("--model-database", help="Name set of images to build model upon")
    args_parser.add_argument("--balanced-dataset", help="Balance the classes in model embedding space")
    args_parser.add_argument("--test-dataset", help="What dataset to test model on")
    args_parser.add_argument("--num-NN", help="Number of NN to base probs on")
    args_parser.add_argument("--num-MC", help="Number of MC samples to base probs on")
    args_parser.add_argument("--method", help="Probability calculation method")
    args_parser.add_argument("--with_OOD", help="Include OOD classes")
    args = args_parser.parse_args()
    
    
    main(args)

