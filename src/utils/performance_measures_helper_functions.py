# Imports
import numpy as np
import pandas as pd


def _get_entropy_(probs,num_classes):
    ui = np.abs(-np.sum(probs*np.log(probs))/np.log(num_classes))
    return ui

def _get_uncertainties_(probs_df: pd.DataFrame,
                        num_classes: int):
    uncertainty = []
    for i in range(len(probs_df.columns)):
        ui = _get_entropy_(probs_df.iloc[:,i],num_classes)
        uncertainty.append(ui)
    
    uncertainty = np.array(uncertainty)
    uncertainty_unique = np.unique(uncertainty)
    
    return uncertainty, uncertainty_unique

def _get_AUC_AvU_(ACr: np.array,
                  ICr: np.array):
    AUC = 0
    for i in range(len(ICr)-1):
        AUC += (ACr[i]+ACr[i+1])/2*(ICr[i+1]-ICr[i])

    return AUC

def _calc_AUC_(precision: list, recall: list):
    AUC = 0
    for i in range(len(recall)-1):
        AUC += (precision[i+1])*(recall[i+1]-recall[i])
        
    return AUC
    
def _calc_aP_for_class_(probs_df: pd.DataFrame, 
                        true_classes: list, 
                        class_: str):
    """Calculates precision and recall for a class given the predicted classes

    Args:
        probs_df (pd.DataFrame): probability distributions for all samples
        true_classes (list): true classes
        class_ (str): class to calculate precision for

    Returns:
        aP, precision [list], recall [list] : average precision, precision and recall for class
    """
    true_classes = np.array(true_classes)==class_
    # get list of probs for all samples where true_class is class_ 
    probs_class = probs_df.loc[class_,:]
    probs_class_sorted = np.sort(probs_class.unique())-1e-7
    
    precision = []
    recall = []
    for i in range(len(probs_class_sorted)):
        pred_classes = probs_class >= probs_class_sorted[-(i+1)]
        TP = np.sum((pred_classes == True) & (true_classes == True))
        FP = np.sum((pred_classes == True) & (true_classes == False))
        FN = np.sum((pred_classes == False) & (true_classes == True))
    
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
    
    precision.insert(0,precision[0])
    recall.insert(0,0)
    
    prec_AUC = [np.max(precision)]
    rec_AUC = [recall[0]]
    prev_rec = recall[0]
    for i in range(len(precision)-1):
        if (precision[i+1] > np.max(precision[i+2:]+[-1])) & (prev_rec < recall[i+1]):
            prec_AUC.append(precision[i+1])
            rec_AUC.append(recall[i+1])
            prev_rec = recall[i+1]
            
    aP = _calc_AUC_(prec_AUC, rec_AUC)

    return aP, precision, recall, prec_AUC, rec_AUC

def _calc_ECE_(accuracies: pd.Series,
               confidences: pd.Series,
               num_samples_bins: pd.Series):
    n = num_samples_bins.sum()
    ECE = (num_samples_bins/n*np.abs(accuracies-confidences)).sum()
    
    return ECE
   
def _calc_CECE_(accuracies: pd.DataFrame,
                confidences: pd.DataFrame,
                num_samples_bins: pd.DataFrame):
    CECE_df = pd.Series(index=accuracies.columns)
    
    for i in range(len(accuracies.columns)):
        class_ = accuracies.columns[i]
        n = num_samples_bins[class_].sum()
        CECE_df[class_] = ((num_samples_bins[class_]/n*
                            np.abs(accuracies[class_]-confidences[class_])).sum())
        
    CECE = CECE_df.mean()
    return CECE, CECE_df 