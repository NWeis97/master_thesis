# Imports
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
import pdb
from scipy.interpolate import interp1d





def main():
    # Define dataframe for storing data
    
    metrics_data = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                         'MaP', 'Accuracy top1', 'Accuracy top2', 'Accuracy top3', 
                                         'Accuracy top5', 'AECE', 'WECE', 'CECE', 'ACE', 'UCE', 
                                         'AvU_simple_thresh', 'AvU_best_thresh', 
                                         'Uncertainty_simple_thresh', 'Uncertainty_best_thresh', 
                                         'AUC_AvU'])
    
    class_data = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                             'index','Average precision',
                                             'Class Expected Calibration Error','acc_top1',
                                             'acc_top2','acc_top3','acc_top5'])
    
    acc_inacc_certain = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                              'Accurat_certain_ratio', 'Inaccurat_certain_ratio'])
    
    calibration_vs_acc = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                             'Confidence','Accuracy'])
    
    accurate_when_certain = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                                  'Uncertainty_thresh','Accuracy'])
    
    uncertain_when_inacc = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                             'Uncertainty_thresh','Recall_inacc'])
    
    class_cali_acc = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                            'aeroplane', 'person', 'tvmonitor', 'train', 'boat', 'dog', 'chair',
                                            'bird', 'bicycle', 'bottle', 'sheep', 'diningtable', 'motorbike',
                                            'sofa', 'cow', 'cat', 'bus', 'pottedplant', 'All (WECE)',
                                             'Class_mean (AECE)'])
    
    class_cali_conf = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                            'aeroplane', 'person', 'tvmonitor', 'train', 'boat', 'dog', 'chair',
                                            'bird', 'bicycle', 'bottle', 'sheep', 'diningtable', 'motorbike',
                                            'sofa', 'cow', 'cat', 'bus', 'pottedplant', 'All (WECE)',
                                             'Class_mean (AECE)'])
    
    class_cali_count = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                            'aeroplane', 'person', 'tvmonitor', 'train', 'boat', 'dog', 'chair',
                                            'bird', 'bicycle', 'bottle', 'sheep', 'diningtable', 'motorbike',
                                            'sofa', 'cow', 'cat', 'bus', 'pottedplant', 'All (WECE)',
                                             'Class_mean (AECE)'])
    
    uncert_cali = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Uncertainty','True error', 'Optimal', 'Count'])
    
    
    # Load pickle file and store in dataframe
    len_interp = 401
    files = [f for f in listdir("./reports/test_results/") if isfile(join("./reports/test_results/", f))]
    for file in files:
        with open("./reports/test_results/"+file, 'rb') as handle:
            b = pickle.load(handle)
            if file.split("_")[-2] == 'metrics':
                metrics_data = pd.concat([metrics_data,pd.DataFrame([b])],axis=0)
            
            else:
                # Class metrics
                b['Class Metrics']['Model Type'] = [b['Model Type']]*len(b['Class Metrics'])
                b['Class Metrics']['Method'] = [b['Method']]*len(b['Class Metrics'])
                b['Class Metrics']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Metrics'])
                b['Class Metrics']['Seed'] = [b['Seed']]*len(b['Class Metrics'])
                class_data = pd.concat([class_data,b['Class Metrics']],axis=0)
                
                # acc_inacc_certain
                f = interp1d(b['Inaccurat_certain_ratio'],b['Accurat_certain_ratio'],kind='linear')
                acc_inacc_certain_file = pd.DataFrame(columns=['Model Type', 'Method', 
                                                               'Calibration Method', 'Seed',
                                                               'Accurat_certain_ratio', 
                                                               'Inaccurat_certain_ratio'])
                acc_inacc_certain_file['Model Type'] = [b['Model Type']]*len_interp
                acc_inacc_certain_file['Method'] = [b['Method']]*len_interp
                acc_inacc_certain_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                acc_inacc_certain_file['Seed'] = [b['Seed']]*len_interp
                acc_inacc_certain_file['Inaccurat_certain_ratio'] = np.linspace(0,1,len_interp)
                acc_inacc_certain_file['Accurat_certain_ratio'] = f(np.linspace(0,1,len_interp))
                acc_inacc_certain = pd.concat([acc_inacc_certain,acc_inacc_certain_file],axis=0)
                
                # calibration vs accuracy
                calibration_vs_acc_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed',
                                             'Confidence','Accuracy'])
                f = interp1d(b['conf_vs_acc']['Confidence'],b['conf_vs_acc']['Accuracy'],kind='linear',fill_value="extrapolate")
                calibration_vs_acc_file['Model Type'] = [b['Model Type']]*len_interp
                calibration_vs_acc_file['Method'] = [b['Method']]*len_interp
                calibration_vs_acc_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                calibration_vs_acc_file['Seed'] = [b['Seed']]*len_interp
                calibration_vs_acc_file['Confidence'] = np.linspace(0,1,len_interp)
                calibration_vs_acc_file['Accuracy'] = f(np.linspace(0,1,len_interp))
                calibration_vs_acc = pd.concat([calibration_vs_acc,calibration_vs_acc_file],axis=0)
                
                # accurate when certain
                accurate_when_certain_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed',
                                             'Uncertainty_thresh','Accuracy'])
                uncert_norm = b['Accurate when certain_unc_thresh']['uncertainty']
                uncert_norm = (uncert_norm-np.min(uncert_norm))/(np.max(uncert_norm)-np.min(uncert_norm))
                f = interp1d(uncert_norm,b['Accurate when certain_unc_thresh']['Accuracy'],kind='linear')
                accurate_when_certain_file['Model Type'] = [b['Model Type']]*len_interp
                accurate_when_certain_file['Method'] = [b['Method']]*len_interp
                accurate_when_certain_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                accurate_when_certain_file['Seed'] = [b['Seed']]*len_interp
                accurate_when_certain_file['Uncertainty_thresh'] = np.linspace(0,1,len_interp)
                accurate_when_certain_file['Accuracy'] = f(np.linspace(0,1,len_interp))
                accurate_when_certain = pd.concat([accurate_when_certain,accurate_when_certain_file],axis=0)
                
                # uncertain when inaccurate
                uncertain_when_inacc_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed',
                                             'Uncertainty_thresh','Recall_inacc'])
                uncert_norm = b['Uncertain when inaccurate_thresh_unc']['uncertainty']
                uncert_norm = (uncert_norm-np.min(uncert_norm))/(np.max(uncert_norm)-np.min(uncert_norm))
                f = interp1d(uncert_norm,b['Uncertain when inaccurate_thresh_unc']['Frac_uncertain'],kind='linear')
                uncertain_when_inacc_file['Model Type'] = [b['Model Type']]*len_interp
                uncertain_when_inacc_file['Method'] = [b['Method']]*len_interp
                uncertain_when_inacc_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                uncertain_when_inacc_file['Seed'] = [b['Seed']]*len_interp
                uncertain_when_inacc_file['Uncertainty_thresh'] = np.linspace(0,1,len_interp)
                uncertain_when_inacc_file['Recall_inacc'] = f(np.linspace(0,1,len_interp))
                uncertain_when_inacc = pd.concat([uncertain_when_inacc,uncertain_when_inacc_file],axis=0)
                
                # Class calibrations
                b['Class Calibration Acc']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                
                b['Class Calibration Conf']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                
                b['Class Calibration count']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                
                class_cali_acc = pd.concat([class_cali_acc,b['Class Calibration Acc']],axis=0)
                class_cali_conf = pd.concat([class_cali_conf,b['Class Calibration Conf']],axis=0)
                class_cali_count = pd.concat([class_cali_count,b['Class Calibration count']],axis=0)
                
                # Uncertainty calibration
                b['Uncertainty Calibration']['Model Type'] = [b['Model Type']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Method'] = [b['Method']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Calibration Method'] = [b['Calibration Method']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Seed'] = [b['Seed']]*len(b['Uncertainty Calibration'])
                uncert_cali = pd.concat([uncert_cali,b['Uncertainty Calibration']],axis=0)
                



    # Plot map
    metrics = (['MaP', 'Accuracy top1', 'Accuracy top2', 'Accuracy top3', 'Accuracy top5',
                   'AECE', 'WECE', 'CECE', 'ACE', 'UCE', 'AUC_AvU'])
    for metric in metrics:
        
        fig, ax = plt.subplots(1,2, sharey=True,figsize=(12,5))
        for i,model_type in enumerate(metrics_data['Model Type'].unique()):  
            df_type = metrics_data[metrics_data['Model Type']==model_type] 
            if model_type == 'Classic':
                hue_order=['None','MCDropout','SWAG','TempScaling']
            else:
                hue_order=['none','MCDropout','SWAG']
            
            sns.boxplot(x='Method',y=metric,hue='Calibration Method',data=df_type, ax=ax.flatten()[i],
                        hue_order=hue_order, )
            
            if i != 0:
                ax.flatten()[i].set_ylabel(None)
            else:
                ax.flatten()[i].set_ylabel(metric,fontsize = 14)
                
            plt.subplots_adjust(left=0.1, right=0.82, top=0.9, bottom=0.1,wspace=0.1)
                
            if i != len(metrics_data['Model Type'].unique())-1:
                ax.flatten()[i].get_legend().remove()
            else:
                ax.flatten()[i].legend(bbox_to_anchor=(1.38, 1), borderaxespad=0)
            
            ax.flatten()[i].set_title(model_type,fontsize = 17)
            ax.flatten()[i].set_xlabel('Method',fontsize = 14)
            ax.flatten()[i].set_xticks(ax.flatten()[i].get_xticks(),fontsize=13, rotation=0)
            #ax.flatten()[i].set_yticks(plt.get_yticks(),fontsize=13, rotation=0)
            
        fig.savefig(f'reports/test/{metric}.png')


if __name__ == '__main__':
    main()
    