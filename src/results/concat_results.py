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
import matplotlib.ticker as ticker

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)





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
    
    entropies = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Entropy_dist','Entropy'])
    
    
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
                                                               'Calibration Method', 'Seed','With_OOD',
                                                               'Accurat_certain_ratio', 
                                                               'Inaccurat_certain_ratio'])
                acc_inacc_certain_file['Model Type'] = [b['Model Type']]*len_interp
                acc_inacc_certain_file['Method'] = [b['Method']]*len_interp
                acc_inacc_certain_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                acc_inacc_certain_file['Seed'] = [b['Seed']]*len_interp
                acc_inacc_certain_file['With_OOD'] = [b['With_OOD']]*len_interp
                acc_inacc_certain_file['Inaccurat_certain_ratio'] = np.linspace(0,1,len_interp)
                acc_inacc_certain_file['Accurat_certain_ratio'] = f(np.linspace(0,1,len_interp))
                acc_inacc_certain = pd.concat([acc_inacc_certain,acc_inacc_certain_file],axis=0)
                
                # calibration vs accuracy
                calibration_vs_acc_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed','With_OOD',
                                             'Confidence','Accuracy'])
                f = interp1d(b['conf_vs_acc']['Confidence'],b['conf_vs_acc']['Accuracy'],kind='linear',fill_value="extrapolate")
                calibration_vs_acc_file['Model Type'] = [b['Model Type']]*len_interp
                calibration_vs_acc_file['Method'] = [b['Method']]*len_interp
                calibration_vs_acc_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                calibration_vs_acc_file['Seed'] = [b['Seed']]*len_interp
                calibration_vs_acc_file['With_OOD'] = [b['With_OOD']]*len_interp
                calibration_vs_acc_file['Confidence'] = np.linspace(0,1,len_interp)
                calibration_vs_acc_file['Accuracy'] = f(np.linspace(0,1,len_interp))
                calibration_vs_acc = pd.concat([calibration_vs_acc,calibration_vs_acc_file],axis=0)
                
                # accurate when certain
                accurate_when_certain_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed','With_OOD',
                                             'Uncertainty_thresh','Accuracy'])
                uncert_norm = b['Accurate when certain_unc_thresh']['uncertainty']
                uncert_norm = (uncert_norm-np.min(uncert_norm))/(np.max(uncert_norm)-np.min(uncert_norm))
                f = interp1d(uncert_norm,b['Accurate when certain_unc_thresh']['Accuracy'],kind='linear')
                accurate_when_certain_file['Model Type'] = [b['Model Type']]*len_interp
                accurate_when_certain_file['Method'] = [b['Method']]*len_interp
                accurate_when_certain_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                accurate_when_certain_file['Seed'] = [b['Seed']]*len_interp
                accurate_when_certain_file['With_OOD'] = [b['With_OOD']]*len_interp
                accurate_when_certain_file['Uncertainty_thresh'] = np.linspace(0,1,len_interp)
                accurate_when_certain_file['Accuracy'] = f(np.linspace(0,1,len_interp))
                accurate_when_certain = pd.concat([accurate_when_certain,accurate_when_certain_file],axis=0)
                
                # uncertain when inaccurate
                uncertain_when_inacc_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'Seed','With_OOD',
                                             'Uncertainty_thresh','Recall_inacc'])
                uncert_norm = b['Uncertain when inaccurate_thresh_unc']['uncertainty']
                uncert_norm = (uncert_norm-np.min(uncert_norm))/(np.max(uncert_norm)-np.min(uncert_norm))
                f = interp1d(uncert_norm,b['Uncertain when inaccurate_thresh_unc']['Frac_uncertain'],kind='linear')
                uncertain_when_inacc_file['Model Type'] = [b['Model Type']]*len_interp
                uncertain_when_inacc_file['Method'] = [b['Method']]*len_interp
                uncertain_when_inacc_file['Calibration Method'] = [b['Calibration Method']]*len_interp
                uncertain_when_inacc_file['Seed'] = [b['Seed']]*len_interp
                uncertain_when_inacc_file['With_OOD'] = [b['With_OOD']]*len_interp
                uncertain_when_inacc_file['Uncertainty_thresh'] = np.linspace(0,1,len_interp)
                uncertain_when_inacc_file['Recall_inacc'] = f(np.linspace(0,1,len_interp))
                uncertain_when_inacc = pd.concat([uncertain_when_inacc,uncertain_when_inacc_file],axis=0)
                
                # Class calibrations
                b['Class Calibration Acc']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                b['Class Calibration Acc']['With_OOD'] = [b['With_OOD']]*len(b['Class Calibration Acc'])
                
                b['Class Calibration Conf']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                b['Class Calibration Conf']['With_OOD'] = [b['With_OOD']]*len(b['Class Calibration Acc'])
                
                b['Class Calibration count']['Model Type'] = [b['Model Type']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Method'] = [b['Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['Seed'] = [b['Seed']]*len(b['Class Calibration Acc'])
                b['Class Calibration count']['With_OOD'] = [b['With_OOD']]*len(b['Class Calibration Acc'])
                
                class_cali_acc = pd.concat([class_cali_acc,b['Class Calibration Acc']],axis=0)
                class_cali_conf = pd.concat([class_cali_conf,b['Class Calibration Conf']],axis=0)
                class_cali_count = pd.concat([class_cali_count,b['Class Calibration count']],axis=0)
                
                # Uncertainty calibration
                b['Uncertainty Calibration']['Model Type'] = [b['Model Type']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Method'] = [b['Method']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Calibration Method'] = [b['Calibration Method']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['Seed'] = [b['Seed']]*len(b['Uncertainty Calibration'])
                b['Uncertainty Calibration']['With_OOD'] = [b['With_OOD']]*len(b['Uncertainty Calibration'])
                uncert_cali = pd.concat([uncert_cali,b['Uncertainty Calibration']],axis=0)
                
                # Entropies
                entropies_id_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Entropy_dist','Entropy'])
                entropies_id_file['Model Type'] = [b['Model Type']]*len(b['Entropies_ID'])
                entropies_id_file['Method'] = [b['Method']]*len(b['Entropies_ID'])
                entropies_id_file['Calibration Method'] = [b['Calibration Method']]*len(b['Entropies_ID'])
                entropies_id_file['Seed'] = [b['Seed']]*len(b['Entropies_ID'])
                entropies_id_file['With_OOD'] = [b['With_OOD']]*len(b['Entropies_ID'])
                entropies_id_file['Entropy_dist'] = ['ID']*len(b['Entropies_ID'])
                entropies_id_file['Entropy'] = b['Entropies_ID']
                
                entropies_ood_file = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Entropy_dist','Entropy'])
                entropies_ood_file['Model Type'] = [b['Model Type']]*len(b['Entropies_OOD'])
                entropies_ood_file['Method'] = [b['Method']]*len(b['Entropies_OOD'])
                entropies_ood_file['Calibration Method'] = [b['Calibration Method']]*len(b['Entropies_OOD'])
                entropies_ood_file['Seed'] = [b['Seed']]*len(b['Entropies_OOD'])
                entropies_ood_file['With_OOD'] = [b['With_OOD']]*len(b['Entropies_OOD'])
                entropies_ood_file['Entropy_dist'] = ['OOD']*len(b['Entropies_OOD'])
                entropies_ood_file['Entropy'] = b['Entropies_OOD']
                
                entropies = pd.concat([entropies,entropies_id_file,entropies_ood_file],axis=0)
                
                
                
    pdb.set_trace()

    entropies['Model'] = entropies['']
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    sns.kdeplot(
    data=entropies, x="total_bill", hue="",
    fill=True, common_norm=False, palette="crest",
    alpha=.5, linewidth=0,
    )











    # ! --------------------------- METRICS GRAPHS ---------------------------------

    all_metrics_data_without_OOD = metrics_data[metrics_data['With_OOD']==False]
    all_metrics_data_without_OOD = all_metrics_data_without_OOD.sort_values(['Model Type'])
    
    btl_metrics_data_without_OOD = metrics_data[metrics_data['Method']!='Vanilla']
    btl_metrics_data_without_OOD = btl_metrics_data_without_OOD[btl_metrics_data_without_OOD['With_OOD']==False]
    btl_metrics_data_without_OOD = btl_metrics_data_without_OOD.sort_values(['Model Type'])
    
    no_knn_metrics_data_without_OOD = metrics_data[metrics_data['Method']!='kNN_gauss_kernel']
    no_knn_metrics_data_without_OOD = no_knn_metrics_data_without_OOD[no_knn_metrics_data_without_OOD['With_OOD']==False]
    no_knn_metrics_data_without_OOD = no_knn_metrics_data_without_OOD.sort_values(['Model Type'])
    
    metrics_iso_with_OOD = metrics_data[metrics_data['Method']!='kNN_gauss_kernel']
    metrics_iso_with_OOD['Model Type'][metrics_iso_with_OOD['With_OOD']==True] = metrics_iso_with_OOD['Model Type'][metrics_iso_with_OOD['With_OOD']==True] + '_OOD'
    metrics_iso_with_OOD = metrics_iso_with_OOD.sort_values(['Model Type'])

    
    if False:
        scale_fig = [1,1,1,1]
        figsize = [13.33,10,14,19]
        width_ratio = [[5,5,3.33],[5,5],[3,3,4],[3,3,3,3,4]]
        name = ['all','btl','no_knn','with_ood']
        data_sets = [all_metrics_data_without_OOD, btl_metrics_data_without_OOD,
                    no_knn_metrics_data_without_OOD,metrics_iso_with_OOD]
        order = [['kNN_gauss_kernel','min_dist_NN'],['kNN_gauss_kernel','min_dist_NN'],None,None]
        yticks = [[1.5,4,2.5,0.5,0.5,0.25,0.33,1.5],
                [1,4,2.5,0.5,0.5,0.25,0.33,1],
                [0.5,1,2.5,0.15,0.15,0.15,0.15,1],
                [1,1.5,2.5,0.15,0.15,0.15,0.15,1]]
        arrow_adjust_h = [0,-0.0095,-0.0085,-0.01]
        
        # * PLOT ALL WITHOUT OOD
        # Plot map
        sns.set()
        metrics = (['MaP', 'Accuracy top1', 'UCE', 'CECE', 'ACE', 'AECE', 'WECE', 'AUC_AvU'])
        arrows = ["->","->","<-","<-","<-","<-","<-","->"]
        arrow_adjust = [0,0,0.02,0.02,0.02,0.02,0.02,0]
        
        
        
        for k, data_set in enumerate(data_sets):
            num_types = len(data_set['Model Type'].unique())
            num_calibrators=len(data_set['Calibration Method'].unique())
            
            fig, ax = plt.subplots(len(metrics),num_types, sharey='row',
                                figsize=(figsize[k]*scale_fig[k],2*len(metrics)*scale_fig[k]),
                                width_ratios=width_ratio[k])
            
            for j,metric in enumerate(metrics):
                for i,model_type in enumerate(data_set['Model Type'].unique()):  
                    df_type = data_set[data_set['Model Type']==model_type]
                    df_type[metric] = df_type[metric]*100
                                                                    
                    if model_type == 'Classic':
                        hue_order=['None','MCDropout','SWAG','TempScaling']
                        sns.set_palette(sns.cubehelix_palette(start=.5, rot=-.5,n_colors=num_calibrators+1, as_cmap=False,))
                        sns.boxplot(x='Method',y=metric,hue='Calibration Method',data=df_type, ax=ax.flatten()[i+j*num_types],
                                hue_order=hue_order, linewidth=0.75,)
                    else:
                        hue_order=['None','MCDropout','SWAG']
                        colors = sns.cubehelix_palette(start=.5, rot=-.5,n_colors=num_calibrators+1, as_cmap=False,)
                        sns.set_palette([colors[i] for i in [0,1,2]])
                        sns.boxplot(x='Method',y=metric,hue='Calibration Method',data=df_type, ax=ax.flatten()[i+j*num_types],
                                hue_order=hue_order, linewidth=0.75,
                                order = order[k])
                    
                    
                    plt.subplots_adjust(left=0.1, right=0.82, top=0.95, bottom=0.05,wspace=0,hspace=0.05)
                    plt.tight_layout()
                    
                    
                    # Set ylabel
                    if (i != 0):
                        ax.flatten()[i+j*num_types].set_ylabel(None)

                    else:
                        # Adjust metric name
                        if metric == 'Accuracy top1':
                            ax.flatten()[i+j*num_types].set_ylabel('Accuracy (%)',fontsize = 14, weight='bold')
                        else:
                            ax.flatten()[i+j*num_types].set_ylabel(metric + ' (%)',fontsize = 14, weight='bold')
                        ax.flatten()[i+j*num_types].yaxis.set_label_coords(-.20, .5)
                        
                        ax.flatten()[i+j*num_types].yaxis.set_major_locator(ticker.MultipleLocator(yticks[k][j]))
                        ax.flatten()[i+j*num_types].yaxis.set_major_formatter(ticker.ScalarFormatter())
                        
                        ax.flatten()[i+j*num_types].annotate("", 
                                                            xy = (-0.375-arrow_adjust_h[k],0.61-arrow_adjust[j]),
                                                            xytext=(-0.375-arrow_adjust_h[k], 0.41-arrow_adjust[j]),xycoords='axes fraction',
                                                            arrowprops=dict(arrowstyle=arrows[j], color='k',
                                                                            lw=2))
                        ax.flatten()[i+j*num_types].annotate("( )", fontsize=20,
                                                            xy = (-0.42,0.45),
                                                            xytext=(-0.42, 0.45),xycoords='axes fraction')
                        
                        
                    # Adjust legends        
                    if (i != len(data_set['Model Type'].unique())-1) | (j > 0):
                        ax.flatten()[i+j*num_types].get_legend().remove()
                    else:
                        ax.flatten()[i+j*num_types].legend(bbox_to_anchor=(1.12, 1), borderaxespad=0,
                                            title="Calibration method")
                        
                    # Remove x-labels
                    ax.flatten()[i+j*num_types].set_xlabel('')
                    
                    # Set titles
                    if j == 0:
                        if model_type == 'Classic':
                            ax.flatten()[i+j*num_types].set_title('Vanilla',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_iso':
                            ax.flatten()[i+j*num_types].set_title('BTL_Iso',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_diag':
                            ax.flatten()[i+j*num_types].set_title('BTL_Diag',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_iso_OOD':
                            ax.flatten()[i+j*num_types].set_title('BTL_Iso_w.OOD',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_diag_OOD':
                            ax.flatten()[i+j*num_types].set_title('BTL_Diag_w.OOD',fontsize = 15,y=1.05, weight='bold')
                    
                    # Remove xticks 
                    if (j != len(metrics)-1):
                        ax.flatten()[i+j*num_types].set_xticklabels('')
                    # Adjust xticks    
                    ax.flatten()[i+j*num_types].set_xticks(ax.flatten()[i].get_xticks(),fontsize=13, rotation=0)
                    
                    
                    
                    
            fig.savefig(f'reports/test/{name[k]}.png')
    


if __name__ == '__main__':
    main()
    