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