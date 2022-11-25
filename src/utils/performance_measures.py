# Imports
import pandas as pd
import numpy as np

# Own Imports
from src.utils.performance_measures_helper_functions import (_get_entropy_,
                                                             _get_uncertainties_,
                                                             _get_AUC_AvU_)
from src.visualization.visualize import UCE_plots




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
    
    return UCE, fig, uncert_cali_plot_df


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
    
    
    