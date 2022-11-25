# Imports
import wandb
import pdb
import argparse
import numpy as np
import pandas as pd
import torch
import ast

# Own imoprts
from src.models.image_classifier import ImageClassifier
from src.visualization.visualize import (visualize_embedding_space_with_images,
                                         image_object_full_bbox_resize_class,
                                         precision_recall_plots,
                                         calibration_plots,
                                         visualize_var_dist_per_class,
                                         visualize_embedding_space)
from src.utils.helper_functions_test import (get_idxs_of_true_and_false_classifications,
                                             sort_idx_on_var_per_class)
from src.utils.performance_measures import (confidence_vs_accuracy,
                                            accruate_when_certain,
                                            uncertain_when_inaccurate,
                                            get_UCE,
                                            get_AvU)


def main(args):
    #* --------------------------------------------------------
    #* ------ Initialize model and extract probabilities ------
    #* --------------------------------------------------------
    
    # Define path to images
    path_to_img = './data/raw/JPEGImages/'
    
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
    
    
    #* --------------------------------------
    #* ------ Get Performance Measures ------
    #* --------------------------------------
    
    # Get MaP
    MaP, aP_df, aP_plotting = classifier.calc_MaP(probs_df,true_classes)
    
    # Get Accuracies
    acc_top1, acc_top2, acc_top3, acc_top5, acc_df = classifier.calc_accuracy(probs_df,true_classes)
    
    # Get ECE
    (cali_plot_df_acc, cali_plot_df_conf, WECE, 
           AECE, CECE, CECE_df, num_each_bin_df) = classifier.calc_ECE(probs_df,true_classes)
    
    # Get UCE
    UCE, UCE_fig, UCE_df = get_UCE(probs_df, true_classes,len(classifier.unique_classes))
    
    # Get AvU
    (AvU_simple_thresh, simple_thresh, AvU_simple_df, 
     AvU_best_thresh, best_thresh, AvU_best_df,
     AUC_AvU, ACr_list, ICr_list) = get_AvU(probs_df, true_classes,len(classifier.unique_classes))
    
    
    
    #* ---------------------------
    #* ------ Create Tables ------
    #* ---------------------------
    
    # Define class-wise metrics table
    class_metrics = pd.DataFrame({'Average precision': aP_df,
                                  'Class Expected Calibration Error': CECE_df},
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
    cali_plots = calibration_plots(cali_plot_df_acc,cali_plot_df_conf,num_each_bin_df)
    
    # Make illustrations of embedding space with classes
    fig_tsne_test, fig_pca_test = visualize_embedding_space(classifier)
    fig_tsne_images, fig_pca_images = visualize_embedding_space_with_images(classifier)
    
    # Make figure with distribution over variances for each class
    fig_vars = visualize_var_dist_per_class(dict_vars)
    
    # Create precision/recall plots
    fig_aP = precision_recall_plots(aP_plotting, aP_df)
    
    
    #********* Exctract examples of images with low and high variance *********#
    
    # Extract variances for dataset
    _, vars = classifier.get_embeddings_of_dataset(test_dataset,num_NN,num_MC,method)
    dict_vars = sort_idx_on_var_per_class(vars, true_classes)
    
    img_small_var = {class_: None for class_ in dict_vars}
    img_large_var = {class_: None for class_ in dict_vars}

    for class_ in dict_vars:
        # sample one image with small (top 10%) and large (top 10%) var for each class
        idx_class = list(dict_vars[class_].keys())
        idx_num_small = np.random.randint(0,int(len(idx_class)/10),1)
        idx_num_large = np.random.randint(0,int(len(idx_class)/10),1)
        i_small = idx_class[idx_num_small.item()]
        i_large = idx_class[-idx_num_large.item()-1] 
        small_var = dict_vars[class_][i_small]  
        large_var = dict_vars[class_][i_large]  
        img_small_var[class_] = image_object_full_bbox_resize_class(path_to_img+objects[i_small],
                                                       bboxs[i_small],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_small],
                                                       pred_classes[i_small],
                                                       probs_df.iloc[:,i_small],
                                                       small_var)
        img_large_var[class_] = image_object_full_bbox_resize_class(path_to_img+objects[i_large],
                                                       bboxs[i_large],
                                                       int(classifier.params['img_size']),
                                                       true_classes[i_large],
                                                       pred_classes[i_large],
                                                       probs_df.iloc[:,i_large],
                                                       large_var)



    #* Extract, for each class, one correctly and on misclassified image *#  
    
    # Extract predicted classes and indecies of true and false classifications
    pred_classes = probs_df.idxmax()
    idx_true, idx_false = get_idxs_of_true_and_false_classifications(pred_classes, true_classes)
       
    img_true = {i: None for i in idx_true}
    img_false = {i: None for class_ in idx_false}    
          
    for i in range(len(idx_true)):   
        i_true = idx_true[i]
        i_false = idx_false[i]    
        _, img_true_var = classifier.get_embedding(objects[i_true],bboxs[i_true])  
        _, img_false_var = classifier.get_embedding(objects[i_false],bboxs[i_false])  
        img_true[i] = image_object_full_bbox_resize_class(path_to_img+objects[i_true],
                                                          bboxs[i_true],
                                                          int(classifier.params['img_size']),
                                                          true_classes[i_true],
                                                          pred_classes[i_true],
                                                          probs_df.iloc[:,i_true],
                                                          torch.mean(img_true_var))
        img_false[i] = image_object_full_bbox_resize_class(path_to_img+objects[i_false],
                                                           bboxs[i_false],
                                                           int(classifier.params['img_size']),
                                                           true_classes[i_false],
                                                           pred_classes[i_false],
                                                           probs_df.iloc[:,i_false],
                                                           torch.mean(img_false_var))
    
    
    
    
    #* ----------------------------------
    #* ------ Log Results to WandB ------
    #* ----------------------------------
    
    # Log simple performance measures
    wandb.log({'MaP':MaP,
               'Accuracy top1': acc_top1,
               'Accuracy top2': acc_top2,
               'Accuracy top3': acc_top3,
               'Accuracy top5': acc_top5,
               'AECE': AECE,
               'WECE': WECE,
               'CECE': CECE,
               'UCE': UCE,
               'AvU_simple_thresh': AvU_simple_thresh,
               'AvU_best_thresh': AvU_best_thresh,
               'Uncertainty_simple_thresh': simple_thresh,
               'Uncertainty_best_thresh': best_thresh,
               'AUC_AvU': AUC_AvU})
 
 
    # Log class-specific metrics
    wandb.log({'Class Metrics': wandb.Table(dataframe=class_metrics),
               'Class Calibration Acc': wandb.Table(dataframe=cali_plot_df_acc.reset_index()),
               'Class Calibration Conf': wandb.Table(dataframe=cali_plot_df_conf.reset_index()),
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
    
    # Log illustration of embedding space with classes
    wandb.log({'fig_tsne_test': wandb.Image(fig_tsne_test),
               'fig_pca_test' : wandb.Image(fig_pca_test),
               'fig_tsne_images': wandb.Image(fig_tsne_images),
               'fig_pca_images' : wandb.Image(fig_pca_images),
               'Calibration plots': wandb.Image(cali_plots),
               'Uncertainty calibration plots': wandb.Image(UCE_fig)})
    
    # Log images with low and high variances   
    for class_ in dict_vars:
        wandb.log({'Small variance examples': wandb.Image(img_small_var),
                   'Large variance examples': wandb.Image(img_large_var)})
        
    # Log correctly and misclassified images
    for i in range(len(idx_true)):   
        wandb.log({'Correct classification': wandb.Image(img_true),
                   'Misclassification': wandb.Image(img_false)})
    
    # Log distribution over variances
    wandb.log({'Distribution over variances': wandb.Image(fig_vars)})
    
    # Log precision/recall plots
    wandb.log({'Preicison/recall plots': wandb.Image(fig_aP)})
    
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

