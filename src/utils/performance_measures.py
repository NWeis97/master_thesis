# Imports
import pandas as pd
import numpy as np
import pdb

# Own Imports
from src.utils.performance_measures_helper_functions import ( _get_uncertainties_,
                                                             _get_AUC_AvU_,
                                                             _calc_AUC_,
                                                             _calc_aP_for_class_,
                                                             _calc_ECE_,
                                                             _calc_CECE_)
from src.visualization.visualize import UCE_plots


def calc_MaP(probs_df: pd.DataFrame, 
             true_classes: list):
        """Calculates both the mean average precision of the model, and the class specific 
           average precisions (area under precision/recall curve)

        Args:
            probs_df (pd.DataFrame): probability distributions for all samples
            true_classes (list): true classes

        Returns:
            MaP, acc_df: The mean average precision (MaP), and class specific precisions (aP_df)
        """
        true_classes_unique = pd.unique(true_classes)
        aP_df = pd.Series(index=true_classes_unique)
        aP_plotting = {'precision':{}, 'recall':{}, 'prec_AUC':{}, 'rec_AUC':{}}
        for class_ in true_classes_unique:
            aP_df[class_],prec,rec,prec_auc,rec_auc = _calc_aP_for_class_(probs_df,
                                                                          true_classes,
                                                                          class_)
            aP_plotting['precision'][class_] = prec
            aP_plotting['recall'][class_] = rec
            aP_plotting['prec_AUC'][class_] = prec_auc
            aP_plotting['rec_AUC'][class_] = rec_auc
        
        MaP = aP_df.mean()    
        return MaP, aP_df, aP_plotting
    
    
def calc_accuracy(probs_df: pd.DataFrame, 
                  true_classes: list):
    """Calculates both the overall accuracy of the model (picking the class with highest
        probability), and the class specific accuracies. 

    Args:
        probs_df (pd.DataFrame): probability distributions for all samples
        true_classes (list): list of true classes

    Returns:
        acc, acc_mean_class, acc_df: The accuracy of the model (acc), the mean across class
                                        accuracies (acc_mean_class), 
                                        and class specific accuracies (acc_df)
    """
    true_classes_unique = pd.unique(true_classes)
    acc_df_top1 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
    acc_df_top2 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
    acc_df_top3 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
    acc_df_top5 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
    pred_classes = []
    [pred_classes.append(list(probs_df.iloc[:,i].sort_values()[::-1].index[:5])) for i in 
                            range(len(probs_df.columns))]

    for i, classes in enumerate(pred_classes):
        t_class = true_classes[i]
        acc_df_top1[t_class] += t_class in classes[:1]
        acc_df_top2[t_class] += t_class in classes[:2]
        acc_df_top3[t_class] += t_class in classes[:3]
        acc_df_top5[t_class] += t_class in classes[:5]
        
    # Get overall metrics
    n = len(pred_classes)
    acc_top1 = acc_df_top1.sum()/n
    acc_top2 = acc_df_top2.sum()/n
    acc_top3 = acc_df_top3.sum()/n
    acc_top5 = acc_df_top5.sum()/n
    
    # Get class specific metrics    
    value_c = pd.Series(true_classes).value_counts()
    acc_df_top1 = acc_df_top1/value_c
    acc_df_top2 = acc_df_top2/value_c
    acc_df_top3 = acc_df_top3/value_c
    acc_df_top5 = acc_df_top5/value_c
    
    acc_df = pd.DataFrame({'acc_top1': acc_df_top1,
                            'acc_top2': acc_df_top2,
                            'acc_top3': acc_df_top3,
                            'acc_top5': acc_df_top5})
    
    return acc_top1, acc_top2, acc_top3, acc_top5, acc_df
   
def calc_ACE(probs_df: pd.DataFrame, 
             true_classes: list,
             num_b: int = 14):
    true_classes = np.array(true_classes)
    true_classes_unique = pd.unique(true_classes).tolist()
    
    # Init dataframes for storing accuracies of bins and number of elements in each bin
    cali_plot_df_acc  = pd.DataFrame(index = np.arange(0,num_b), columns = true_classes_unique)
    cali_plot_df_conf = pd.DataFrame(index = np.arange(0,num_b), columns = true_classes_unique)
    num_each_bin_df = pd.DataFrame(index = np.arange(0,num_b), columns = true_classes_unique)
     
    # unique true classes
    true_classes_unique = pd.unique(true_classes).tolist()
    for class_ in true_classes_unique:
        class_probs = probs_df.loc[class_,:]
        num_samples_class = len(class_probs)
        num_per_bin = int(np.ceil(num_samples_class/num_b))
        class_probs_sorted_idx = np.argsort(class_probs).values
        for i in range(num_b):
            min_idx = i*num_per_bin
            max_idx = np.min([(i+1)*num_per_bin,num_samples_class])
            idx_bin = class_probs_sorted_idx[min_idx:max_idx]
            class_probs_in_bin = class_probs[idx_bin]
            class_probs_in_bin_true = true_classes[idx_bin]==class_
            
            if class_probs_in_bin_true.size != 0:
                cali_plot_df_acc.loc[i,class_] = np.mean(class_probs_in_bin_true)
                cali_plot_df_conf.loc[i,class_] = np.mean(class_probs_in_bin)
                num_each_bin_df.loc[i,class_] = len(idx_bin)
            
    # Calculate ACE    
    ACE, ACE_df = _calc_CECE_(cali_plot_df_acc.iloc[:,:],
                                cali_plot_df_conf.iloc[:,:],
                                num_each_bin_df.iloc[:,:])
    
    return ACE, ACE_df, cali_plot_df_acc, cali_plot_df_conf, num_each_bin_df
    
     
def calc_ECE(probs_df: pd.DataFrame, 
             true_classes: list):
    true_classes = np.array(true_classes)
    true_classes_unique = pd.unique(true_classes).tolist()
    true_classes_unique.append('All (WECE)')
    bins = np.array([0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1])
    bins_mid = [(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]
    
    # Init dataframes for storing accuracies of bins and number of elements in each bin
    cali_plot_df_acc  = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
    cali_plot_df_conf = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
    num_each_bin_df = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
    
    cali_plot_df_acc['All (WECE)']  = np.zeros((len(bins_mid),))
    cali_plot_df_conf['All (WECE)'] = np.zeros((len(bins_mid),))
    num_each_bin_df['All (WECE)']   = np.zeros((len(bins_mid),))
    
    # reset true classes
    true_classes_unique = pd.unique(true_classes).tolist()
    
    # for numeric stability
    bins[0] += -1.e-1 
    bins[-1] += 1.e-1
    
    for class_ in true_classes_unique:
        class_probs = probs_df.loc[class_,:]
        for i in range(len(bins_mid)):
            class_probs_in_bin = (class_probs <= bins[i+1]) & (class_probs > bins[i])
            class_probs_in_bin_true = true_classes[class_probs_in_bin]==class_
            if class_probs_in_bin_true.size != 0:
                cali_plot_df_acc.loc[bins_mid[i],class_] = np.mean(class_probs_in_bin_true)
                cali_plot_df_acc.loc[bins_mid[i],'All (WECE)'] += np.sum(class_probs_in_bin_true)
                cali_plot_df_conf.loc[bins_mid[i],class_] = np.mean(class_probs[class_probs_in_bin])
                cali_plot_df_conf.loc[bins_mid[i],'All (WECE)'] += np.sum(class_probs[class_probs_in_bin])
                num_each_bin_df.loc[bins_mid[i],class_] = np.sum(class_probs_in_bin)
                num_each_bin_df.loc[bins_mid[i],'All (WECE)'] += np.sum(class_probs_in_bin)
    
    # Calc accuracy and conf across all images (WECE)
    cali_plot_df_acc['All (WECE)'] = cali_plot_df_acc['All (WECE)']/num_each_bin_df['All (WECE)']
    cali_plot_df_conf['All (WECE)'] = cali_plot_df_conf['All (WECE)']/num_each_bin_df['All (WECE)']
    
    # Calculate non-weighted mean across accuracies (AECE)
    cali_plot_df_acc['Class_mean (AECE)'] = cali_plot_df_acc.iloc[:,:-1].mean(axis=1)
    cali_plot_df_conf['Class_mean (AECE)'] = cali_plot_df_conf.iloc[:,:-1].mean(axis=1)
    num_each_bin_df['Class_mean (AECE)'] = num_each_bin_df['All (WECE)']
    
    # Calaculate ECE across weighted and non-weighted accuracies
    # SCE:
    # https://medium.com/codex/metrics-to-measuring-calibration-in-deep-learning-36b0b11fe816
    WECE = _calc_ECE_(cali_plot_df_acc['All (WECE)'], 
                      cali_plot_df_conf['All (WECE)'], 
                      num_each_bin_df['All (WECE)'])
    AECE = _calc_ECE_(cali_plot_df_acc['Class_mean (AECE)'],
                      cali_plot_df_conf['Class_mean (AECE)'],
                      num_each_bin_df['All (WECE)'])
    CECE, CECE_df = _calc_CECE_(cali_plot_df_acc.iloc[:,:-2],
                                cali_plot_df_conf.iloc[:,:-2],
                                num_each_bin_df.iloc[:,:-2])
    
    return (cali_plot_df_acc, cali_plot_df_conf, WECE, 
            AECE, CECE, CECE_df, num_each_bin_df)    
    
    
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
                          true_classes: list,
                          num_classes: int):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    
    # Get uncertainties
    uncertainty, uncertainty_unique = _get_uncertainties_(probs_df, num_classes)
    
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
                              true_classes: list,
                              num_classes: int):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    
    # Get uncertainties
    uncertainty, _ = _get_uncertainties_(probs_df, num_classes)
    idx_inacc = pred_classes != true_classes
    uncertainty = uncertainty[idx_inacc]
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


def get_UCE(probs_df: pd.DataFrame,
            true_classes: list,
            num_classes: int):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    
    # Get uncertainties
    uncertainty, _ = _get_uncertainties_(probs_df, num_classes)
    
    # Create expected bins
    uncert_targets = [0,0.0025,0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1]
    uncert_bins = [(uncert_targets[i+1]+uncert_targets[i])/2 for i in range(len(uncert_targets)-1)]
    uncert_targets[0] -= 0.01
    uncert_targets[-1] += 0.01
    
    inacc_targets = np.array(uncert_bins)/((num_classes-1)/float(num_classes))
    # Init series
    uncert_cali_plot_df = pd.DataFrame(index = inacc_targets, columns=['True error','Count'])
    
    # Calc expected vs. actual:
    for i in range(len(uncert_bins)):
        idx_in_bin = (uncertainty>=uncert_targets[i]) & (uncertainty<uncert_targets[i+1])
        preds = pred_classes[idx_in_bin]
        trues = true_classes[idx_in_bin]
        frac_inaccurate = np.mean(preds != trues)
        
        uncert_cali_plot_df['True error'].iloc[i] = frac_inaccurate
        uncert_cali_plot_df['Count'].iloc[i] = np.sum(idx_in_bin)
        
    # Calculate UCE
    n = uncert_cali_plot_df['Count'].sum()
    UCE = np.sum(uncert_cali_plot_df['Count']/n*
                 np.abs(uncert_cali_plot_df['True error']-uncert_cali_plot_df.index))
    
    uncert_cali_plot_df = uncert_cali_plot_df.reset_index().rename(columns={'index':'Uncertainty'})
    uncert_cali_plot_df['Optimal'] = uncert_cali_plot_df['Uncertainty']*(num_classes-1)/(num_classes)
    
    # Get UCE plots
    fig = UCE_plots(uncert_cali_plot_df)
    
    return UCE, fig, uncert_cali_plot_df, uncertainty


def get_AvU(probs_df: pd.DataFrame,
            true_classes: list,
            num_classes: int):
    true_classes = np.array(true_classes)
    pred_classes = probs_df.idxmax(0)
    
    # Get uncertainties
    uncertainty, uncertainty_unique = _get_uncertainties_(probs_df, num_classes)
    
    # Get index of accurate and inaccurate predicitons
    idx_acc = pred_classes == true_classes
    idx_inacc = pred_classes != true_classes
    
    # Find mean uncertainty for accurate and iunaccurate predictions
    acc_uncert = np.mean(uncertainty[idx_acc])
    inacc_uncert = np.mean(uncertainty[idx_inacc])
    
    # Find threshold
    simple_thresh = (acc_uncert + inacc_uncert)/2
    
    # Calculate measures
    nAU = np.sum(idx_acc & (uncertainty > simple_thresh))
    nAC = np.sum(idx_acc & (uncertainty <= simple_thresh))
    nIU = np.sum(idx_inacc & (uncertainty > simple_thresh))
    nIC = np.sum(idx_inacc & (uncertainty <= simple_thresh))
    
    # Calc AvU score
    AvU_simple_thresh = (nAC+nIU)/(nAC+nAU+nIC+nIU)
    
    AvU_simple_df = pd.DataFrame(index=['Accurate','Inaccurate'], columns=['Certain','Uncertain'])
    AvU_simple_df.loc['Accurate','Certain'] = nAC
    AvU_simple_df.loc['Inaccurate','Certain'] = nIC
    AvU_simple_df.loc['Accurate','Uncertain'] = nAU
    AvU_simple_df.loc['Inaccurate','Uncertain'] = nIU
    
    # Calc AUC for AvU
    ACr_list = [0]
    ICr_list = [0]
    
    # Find best AvU score
    AvU_best_thresh = 0
    best_thresh = 0
    
    # Best df
    AvU_best_df = pd.DataFrame(index=['Accurate','Inaccurate'], columns=['Certain','Uncertain'])
    for u_thresh in uncertainty_unique:
        nAU = np.sum(idx_acc & (uncertainty > u_thresh))
        nAC = np.sum(idx_acc & (uncertainty <= u_thresh))
        nIU = np.sum(idx_inacc & (uncertainty > u_thresh))
        nIC = np.sum(idx_inacc & (uncertainty <= u_thresh))
        
        # Accurate Certain rate (like TPR)
        ACr = nAC/(nAC+nAU)
        
        # Inaccurate Certain rate (Like FPR)
        ICr = nIC/(nIC+nIU)
        
        # Append to lists
        ACr_list.append(ACr)
        ICr_list.append(ICr)
        
        # Store best AvU
        if (nAC+nIU)/(nAC+nAU+nIC+nIU) > AvU_best_thresh:
            AvU_best_thresh = (nAC+nIU)/(nAC+nAU+nIC+nIU)
            best_thresh = u_thresh
            AvU_best_df.loc['Accurate','Certain'] = nAC
            AvU_best_df.loc['Inaccurate','Certain'] = nIC
            AvU_best_df.loc['Accurate','Uncertain'] = nAU
            AvU_best_df.loc['Inaccurate','Uncertain'] = nIU
    
    AUC_AvU = _get_AUC_AvU_(np.array(ACr_list), np.array(ICr_list))
    

    return (AvU_simple_thresh, simple_thresh, AvU_simple_df, 
            AvU_best_thresh, best_thresh, AvU_best_df,
            AUC_AvU, ACr_list, ICr_list)
    
    
    