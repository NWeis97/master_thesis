""" This script tests a classic classifier model. This includes extracting probability disitrbutions
for the entire test dataset, extracting metrics, create visualizations, and more. Most is saved to 
WandB (so in order to make full use of this, WandB is recommended), while some of the metrics and
graph values are saved to a pickle file. To use and extract the information from the pickle files,
see results/concat_results.py
"""

# Imports
import wandb
import pdb
import argparse
import numpy as np
import pandas as pd
import torch
import ast
import pickle
import json
import os
import shutil

# Own imoprts
from src.models.image_classifier import ImageClassifier, init_classifier_model
from src.visualization.visualize import (image_object_full_bbox_resize_class,
                                         precision_recall_plots,
                                         calibration_plots,
                                         UCE_plots_classewise,
                                         visualize_metric_dist_per_class)
from src.utils.helper_functions_test import (get_idxs_of_true_and_false_classifications,
                                             sort_idx_on_metric_per_class)
from src.utils.performance_measures import (confidence_vs_accuracy,
                                            accruate_when_certain,
                                            uncertain_when_inaccurate,
                                            get_UCE,
                                            get_AvU,
                                            calc_MaP,
                                            calc_accuracy,
                                            calc_ECE,
                                            calc_ACE,
                                            get_entropies)


def main(args):
    #* --------------------------------------------------------
    #* ------ Initialize model and extract probabilities ------
    #* --------------------------------------------------------
    
    # Define path to images
    path_to_img = './data/raw/JPEGImages/'
    
    # Extract args
    classifier_model = args.model_name
    database_model = args.model_database
    test_dataset = args.test_dataset
    calibration_method = args.calibration_method
    
    # Init wandb
    wandb.init(
                project="master_thesis",
                entity="nweis97",
                config={"classifier_model": classifier_model, 
                        "database_model": database_model,
                        "test_dataset": test_dataset,
                        "calibration_method": calibration_method},
                job_type="Test"
        )
    
    
    # Extract classifier model
    if calibration_method == 'Ensemble':
        classifier = init_classifier_model(classifier_model,
                                       database_model,
                                       calibration_method='None')
    else:
        classifier = init_classifier_model(classifier_model,
                                        database_model,
                                        calibration_method=calibration_method)
        
    
    
    if calibration_method != 'Ensemble':
        # Calculate probabilities
        (probs_df, 
        objects, 
        bboxs, 
        true_classes) = classifier.get_probability_dist_dataset(test_dataset)
    else:
        # Get objects, bboxs, and true_classes
        # Calculate probabilities
        (probs_df, 
        objects, 
        bboxs, 
        true_classes) = classifier.get_probability_dist_dataset(test_dataset)
        
        # Extract probabilities from same classifiers 
        # Get names of runs with same settings:
        file_names = os. listdir('models/state_dicts_classic/.')
        
        # Get current files params
        with open(f'./models/params/{classifier_model}.json', 'r') as f:
            data = json.load(f)
        
        data.pop('seed')
        data.pop('temp_opt')
        
        # Open params and check if same settings
        count = 1
        for file in file_names:
            with open(f'./models/params/{file[:-3]}.json', 'r') as f:
                cur_data = json.load(f)
            cur_data.pop('seed',None)
            cur_data.pop('temp_opt',None)

            if (cur_data == data) & (file[:-3] != classifier_model):
                probs_file_name = classifier._get_test_file_storing_name_()
                probs_file_name = probs_file_name.split('_')[1:]
                probs_file_name = file[:-3] + '_' + '_'.join(probs_file_name) + '.csv'
                cur_probs_df = pd.read_csv(f'./reports/probability_distributions_classic/{probs_file_name}',index_col=0)
                
                probs_df = (probs_df*count+cur_probs_df)/(count+1)
                count += 1
                
        
    
    #* -------------------------------------------
    #* ------ Remove classes not trained on ------
    #* -------------------------------------------
    OOD_true_classes_idx = ([i for i in range(len(true_classes)) 
                                 if '_OOD' in true_classes[i]])
    OOD_probs_df = probs_df.iloc[:,OOD_true_classes_idx]
    OOD_probs_df.columns = np.arange(0,len(OOD_probs_df.columns),1)
    #OOD_objects = [objects[i] for i in range(len(true_classes)) if i in OOD_true_classes_idx]
    #OOD_bboxs = [bboxs[i] for i in range(len(true_classes)) if i in OOD_true_classes_idx]
    #OOD_true_classes = ([true_classes[i] for i in range(len(true_classes)) 
    #                    if i in OOD_true_classes_idx])
    
    ID_true_classes_idx = [i for i in range(len(true_classes)) if i not in OOD_true_classes_idx]
    probs_df = probs_df.iloc[:,ID_true_classes_idx]
    objects = [objects[i] for i in range(len(true_classes)) if i in ID_true_classes_idx]
    bboxs = [bboxs[i] for i in range(len(true_classes)) if i in ID_true_classes_idx]
    true_classes = ([true_classes[i] for i in range(len(true_classes)) 
                        if i in ID_true_classes_idx])
    probs_df.columns = np.arange(0,len(probs_df.columns),1)
        
        
    
    #* --------------------------------------
    #* ------ Get Performance Measures ------
    #* --------------------------------------
    
    # Get MaP
    MaP, aP_df, aP_plotting = calc_MaP(probs_df,true_classes)
    
    # Get Accuracies
    acc_top1, acc_top2, acc_top3, acc_top5, acc_df = calc_accuracy(probs_df,true_classes)
    
    # Get ECE
    ACE, ACE_df, ACE_acc_df, ACE_conf_df, ACE_num_each_bin = calc_ACE(probs_df,true_classes)
    (cali_plot_df_acc, cali_plot_df_conf, WECE, 
           AECE, CECE, CECE_df, num_each_bin_df) = calc_ECE(probs_df,true_classes)
    
    # Get UCE
    (UCE, UCE_fig, UCE_df, 
     uncertainties,
     uncert_class_df_uncert,
     uncert_class_df_error, 
     uncert_class_df_count) = get_UCE(probs_df, true_classes,len(classifier.unique_classes))
    
    # Get AvU
    (AvU_simple_thresh, simple_thresh, AvU_simple_df, 
     AvU_best_thresh, best_thresh, AvU_best_df,
     AUC_AvU, ACr_list, ICr_list) = get_AvU(probs_df, true_classes,len(classifier.unique_classes))
    
    # Get entropies of OOD objects
    entropies_OOD = get_entropies(OOD_probs_df)
    entropies_ID = get_entropies(probs_df)
    
    
    
    #* ---------------------------
    #* ------ Create Tables ------
    #* ---------------------------
    
    # Define class-wise metrics table
    class_metrics = pd.DataFrame({'Average precision': aP_df,
                                  'Class Expected Calibration Error': CECE_df,
                                  'Class Adaptive Calibration Error': ACE_df},
                                 index=aP_df.index)
    class_metrics = pd.concat([class_metrics,acc_df],axis=1)
    class_metrics.loc['**mean**'] = class_metrics.mean()
    class_metrics = class_metrics.reset_index()
    
    # Confidence vs. accuracy table 
    conf_vs_acc_df = confidence_vs_accuracy(probs_df, true_classes)
    
    # Accurate when certain table
    acc_cert_df = accruate_when_certain(probs_df, true_classes,len(classifier.unique_classes))
    
    # Uncertain when inaccurate
    uncert_inacc_df = uncertain_when_inaccurate(probs_df, true_classes,
                                                len(classifier.unique_classes))
    
    
    #* ---------------------------------
    #* ------ Make Visualizations ------
    #* ---------------------------------
    
    # Make calibration plots
    cali_plots = calibration_plots(cali_plot_df_acc,cali_plot_df_conf,num_each_bin_df, False)
    cali_plots_ace = calibration_plots(ACE_acc_df, ACE_conf_df, ACE_num_each_bin, True)
    
    # Make uncertainty plots plots
    uncert_plots = UCE_plots_classewise(uncert_class_df_error, uncert_class_df_uncert, 
                                        uncert_class_df_count)
    
    # Create precision/recall plots
    fig_aP = precision_recall_plots(aP_plotting, aP_df)
    
    
    #********* Exctract examples of images with low and high variance *********#
    
    # Extract variances for dataset
    pred_classes = probs_df.idxmax()
    dict_uncertainties = sort_idx_on_metric_per_class(uncertainties, true_classes)
    
    # Make figure with distribution over variances for each class
    fig_uncertainties = visualize_metric_dist_per_class(dict_uncertainties.copy(),'uncertainties')
    
    img_small_uncert = {class_: None for class_ in dict_uncertainties}
    img_large_uncert = {class_: None for class_ in dict_uncertainties}

    for class_ in dict_uncertainties:
        # sample one image with small (top 10%) and large (top 10%) var for each class
        idx_class = list(dict_uncertainties[class_].keys())
        idx_num_small = np.random.randint(0,int(len(idx_class)/10),1)
        idx_num_large = np.random.randint(0,int(len(idx_class)/10),1)
        i_small = idx_class[idx_num_small.item()]
        i_large = idx_class[-idx_num_large.item()-1] 
        
        # Uncertainty images
        small_uncert = dict_uncertainties[class_][i_small]  
        large_uncert = dict_uncertainties[class_][i_large]  
        img_small_uncert[class_] = image_object_full_bbox_resize_class(path_to_img+objects[i_small],
                                                       bboxs[i_small],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_small],
                                                       pred_classes[i_small],
                                                       probs_df.iloc[:,i_small],
                                                       small_uncert,
                                                       'uncertainty')
        img_large_uncert[class_] = image_object_full_bbox_resize_class(path_to_img+objects[i_large],
                                                       bboxs[i_large],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_large],
                                                       pred_classes[i_large],
                                                       probs_df.iloc[:,i_large],
                                                       large_uncert,
                                                       'uncertainty')

    
    #* Extract, for each class, one correctly and on misclassified image *#  
    
    # Extract predicted classes and indecies of true and false classifications
    idx_true, idx_false = get_idxs_of_true_and_false_classifications(pred_classes, true_classes)
       
    img_true = {i: None for i in idx_true}
    img_false = {i: None for i in idx_false}    
          
    for i in range(len(idx_true)):   
        i_true = idx_true[i]
        i_false = idx_false[i]    
        img_true_uncert = uncertainties[i_true]
        img_false_uncert = uncertainties[i_false]
        img_true[i] = image_object_full_bbox_resize_class(path_to_img+objects[i_true],
                                                          bboxs[i_true],
                                                          int(classifier.params['img_size']),
                                                          true_classes[i_true],
                                                          pred_classes[i_true],
                                                          probs_df.iloc[:,i_true],
                                                          img_true_uncert,
                                                          'uncertainty')
        img_false[i] = image_object_full_bbox_resize_class(path_to_img+objects[i_false],
                                                           bboxs[i_false],
                                                           int(classifier.params['img_size']),
                                                           true_classes[i_false],
                                                           pred_classes[i_false],
                                                           probs_df.iloc[:,i_false],
                                                           img_false_uncert,
                                                           'uncertainty')
    
    
    
    #* ---------------------------------------
    #* ------ Log Results to local file ------
    #* ---------------------------------------
    method = 'Vanilla'
    metrics_dict = {'Model Type': classifier.model_type,
                    'Method': method,
                    'Calibration Method': calibration_method,
                    'With_OOD': False,
                    'Seed': classifier.params['seed'],
                    'MaP': MaP,
                    'Accuracy top1': acc_top1,
                    'Accuracy top2': acc_top2,
                    'Accuracy top3': acc_top3,
                    'Accuracy top5': acc_top5,
                    'AECE': AECE,
                    'WECE': WECE,
                    'CECE': CECE,
                    'ACE': ACE,
                    'UCE': UCE,
                    'AvU_simple_thresh': AvU_simple_thresh,
                    'AvU_best_thresh': AvU_best_thresh,
                    'Uncertainty_simple_thresh': simple_thresh,
                    'Uncertainty_best_thresh': best_thresh,
                    'AUC_AvU': AUC_AvU}
    
    graphs_dict = {'Model Type': classifier.model_type,
                   'Method': method,
                   'Calibration Method': calibration_method,
                   'With_OOD': False,
                   'Seed': classifier.params['seed'],
                   'conf_vs_acc': conf_vs_acc_df,
                   'Accurate when certain_unc_thresh': acc_cert_df,
                   'Uncertain when inaccurate_thresh_unc': uncert_inacc_df,
                   'Accurat_certain_ratio': ACr_list,
                   'Inaccurat_certain_ratio': ICr_list,
                   'Class Metrics': class_metrics,
                   'Class Calibration Acc': cali_plot_df_acc,
                   'Class Calibration Conf': cali_plot_df_conf,
                   'Class Calibration count': num_each_bin_df,
                   'Class Calibration Acc (ACE)': ACE_acc_df,
                   'Class Calibration Conf (ACE)': ACE_conf_df,
                   'Class Calibration count (ACE)': ACE_num_each_bin,
                   'Uncertainty Calibration': UCE_df,
                   'AvU_simple_df': AvU_simple_df,
                   'AvU_best_df': AvU_best_df,
                   'Entropies_ID': entropies_ID,
                   'Entropies_OOD': entropies_OOD}
    
     # Extract args
    file_name = f"{classifier_model}_{method}_{calibration_method}"
    with open(f'reports/test_results/{file_name}_metrics_dict.pickle', 'wb') as f:
        pickle.dump(metrics_dict, f)
    with open(f'reports/test_results/{file_name}_graphs_dict.pickle', 'wb') as f:
        pickle.dump(graphs_dict, f)
    
    
    #* ----------------------------------
    #* ------ Log Results to WandB ------
    #* ----------------------------------
    
    # Log simple performance measures
    wandb.log(metrics_dict)
 
 
    # Log class-specific metrics
    wandb.log({'Class Metrics': wandb.Table(dataframe=class_metrics),
               'Class Calibration Acc': wandb.Table(dataframe=cali_plot_df_acc.reset_index()),
               'Class Calibration Conf': wandb.Table(dataframe=cali_plot_df_conf.reset_index()),
               'Class Calibration Acc (ACE)': wandb.Table(dataframe=ACE_acc_df.reset_index()),
               'Class Calibration Conf (ACE)': wandb.Table(dataframe=ACE_conf_df.reset_index()),
               'Uncertainty Calibration': wandb.Table(dataframe=UCE_df),
               'AvU_simple_df': wandb.Table(dataframe=AvU_simple_df
                                                      .reset_index()
                                                      .rename(columns={'index':'Accuracy'})),
               'AvU_best_df': wandb.Table(dataframe=AvU_best_df
                                                    .reset_index()
                                                    .rename(columns={'index':'Accuracy'}))})
    
    # Log confusion matrix
    pred_classes = probs_df.idxmax(0)
    mapper = {classifier.unique_classes[i]:i for i in range(len(classifier.unique_classes))}
    t_class = [mapper[class_] for class_ in true_classes]
    p_class = [mapper[class_] for class_ in pred_classes]
    wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,y_true=t_class, 
                                                            preds=p_class,
                                                            class_names=classifier.unique_classes)})
    
    # Log illustration of embedding space with classes and other figures
    wandb.log({'Uncertainty plots': wandb.Image(uncert_plots),
               'Calibration plots': wandb.Image(cali_plots),
               'Calibration plots (ACE)': wandb.Image(cali_plots_ace),
               'Uncertainty calibration plots': wandb.Image(UCE_fig),
               'Distribution over uncertainties': wandb.Image(fig_uncertainties),
               'Preicison/recall plots': wandb.Image(fig_aP)})
    
    # Log images with low and high variances   
    for class_ in dict_uncertainties:
        wandb.log({'Low uncertainty examples': wandb.Image(img_small_uncert[class_]),
                   'High uncertainty examples': wandb.Image(img_large_uncert[class_])})
        
    # Log correctly and misclassified images
    for i in range(len(idx_true)):   
        wandb.log({'Correct classification': wandb.Image(img_true[i]),
                   'Misclassification': wandb.Image(img_false[i])})
    
    # Confidence vs. accuracy table 
    for i in range(len(conf_vs_acc_df['Confidence'])):
        wandb.log({'Confidence vs. accuracy_conf': conf_vs_acc_df['Confidence'][i],
                   'Confidence vs. accuracy_acc': conf_vs_acc_df['Accuracy'][i],
                   'Confidence vs. accuracy_count': conf_vs_acc_df['Count'][i]})
    
    # Accurate when certain table
    for i in range(len(acc_cert_df['uncertainty'])):
        wandb.log({'Accurate when certain_unc_thresh': acc_cert_df['uncertainty'][i],
                   'Accurate when certain_acc': acc_cert_df['Accuracy'][i],
                   'Accurate when certain_count': acc_cert_df['Count'][i]})
    
    # Uncertain when inaccurate
    for i in range(len(uncert_inacc_df['uncertainty'])):
        wandb.log({'Uncertain when inaccurate_thresh_unc': uncert_inacc_df['uncertainty'][i],
                   'Uncertain when inaccurate_frac_unc': uncert_inacc_df['Frac_uncertain'][i]})
    
    # Accuray_Certain_rate vs Inaccurate_Certain_rate
    for i in range(len(ACr_list)):
        wandb.log({'Accurat_certain_ratio': ACr_list[i],
                   'Inaccurat_certain_ratio': ICr_list[i]})


    # Remove local files
    wandb_local_dir = wandb.run.dir[:-5]
    wandb.finish()
    shutil.rmtree(wandb_local_dir, ignore_errors=True)





if __name__ == '__main__':
    
    # Get configs
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--model-name", help="Name of model to test")
    args_parser.add_argument("--model-database", help="Name set of images to build model upon")
    args_parser.add_argument("--test-dataset", help="What dataset to test model on")
    args_parser.add_argument("--calibration_method", help="What type of calibration should be used")
    args = args_parser.parse_args()
    
    
    main(args)

