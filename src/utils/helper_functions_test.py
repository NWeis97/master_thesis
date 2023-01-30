""" This script contains helper functions for the training_test/test_*.py scripts """

# Imports
import numpy as np
import torch
import pdb

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


def sort_idx_on_metric_per_class(metric, classes):
    classes_unique = np.sort(np.unique(classes))
    dict_vars = {class_:{} for class_ in classes_unique}
    for i, class_ in enumerate(classes):
        if type(metric) == torch.Tensor:
            dict_vars[class_][i]=torch.mean(metric[:,i]).item()
        else:
            dict_vars[class_][i]=metric[i]
    for key in dict_vars.keys():
        dict_vars[key] = dict(sorted(dict_vars[key].items(), key=lambda item: item[1]))
    
    return dict_vars