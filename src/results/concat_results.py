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
from src.utils.performance_measures import Hellinger_distance

sns.set()



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
    uncert_cali_ood = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Uncertainty','True error', 'Optimal', 'Count'])
    
    entropies = pd.DataFrame(columns=['Model Type', 'Method', 'Calibration Method', 'With_OOD','Seed', 
                                        'Entropy_dist','Entropy'])
    
    
    # Load pickle file and store in dataframe
    #'./reports/test_results_iso_large_var_prior/'
    folder = './reports/test_results_final/'
    len_interp = 401
    files = [f for f in listdir(f"{folder}") if isfile(join(f"{folder}", f))]
    for file in files:
        with open(f"{folder}"+file, 'rb') as handle:
            b = pickle.load(handle)
            if file.split("_")[-2] == 'metrics':
                metrics_data = pd.concat([metrics_data,pd.DataFrame([b])],axis=0)
            
            else:
                # Class metrics
                b['Class Metrics']['Model Type'] = [b['Model Type']]*len(b['Class Metrics'])
                b['Class Metrics']['Method'] = [b['Method']]*len(b['Class Metrics'])
                b['Class Metrics']['Calibration Method'] = [b['Calibration Method']]*len(b['Class Metrics'])
                b['Class Metrics']['Seed'] = [b['Seed']]*len(b['Class Metrics'])
                b['Class Metrics']['With_OOD'] = [b['With_OOD']]*len(b['Class Metrics'])
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
                f1 = interp1d(b['Uncertainty Calibration']['Uncertainty'],b['Uncertainty Calibration']['True error'],kind='linear',fill_value=np.nan,bounds_error=False)
                f2 = interp1d(b['Uncertainty Calibration']['Uncertainty'],b['Uncertainty Calibration']['Count'],kind='linear',fill_value=0,bounds_error=False)
                f3 = interp1d(b['Uncertainty Calibration']['Uncertainty'],b['Uncertainty Calibration']['Optimal'],kind='linear',fill_value=np.nan,bounds_error=False)
                x_u = np.array([0,0.00025,0.0005,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.0125,
                                0.015,0.0175,0.02,0.025,0.030,0.035,0.040,0.050,0.060,
                                0.070,0.075,0.08,0.09,0.1,0.15,0.20,0.25,0.30,0.35,
                                0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1])
                uncert_new = pd.DataFrame({'Uncertainty':x_u,
                                           'True error':f1(x_u),
                                           'Count':f2(x_u),
                                           'Optimal': f3(x_u)})
                
                uncert_new['Model Type'] = [b['Model Type']]*len(uncert_new)
                uncert_new['Method'] = [b['Method']]*len(uncert_new)
                uncert_new['Calibration Method'] = [b['Calibration Method']]*len(uncert_new)
                uncert_new['Seed'] = [b['Seed']]*len(uncert_new)
                uncert_new['With_OOD'] = [b['With_OOD']]*len(uncert_new)
                
                uncert_cali = pd.concat([uncert_cali,uncert_new],axis=0)
                
                # Uncertainty calibration
                if 'Uncertainty Calibration (OOD)' in b.keys():
                    if b['Uncertainty Calibration (OOD)'] is not None:
                        f1_ood = interp1d(b['Uncertainty Calibration (OOD)']['Uncertainty'],b['Uncertainty Calibration (OOD)']['True error'],kind='linear',fill_value=np.nan,bounds_error=False)
                        f2_ood = interp1d(b['Uncertainty Calibration (OOD)']['Uncertainty'],b['Uncertainty Calibration (OOD)']['Count'],kind='linear',fill_value=0,bounds_error=False)
                        f3_ood = interp1d(b['Uncertainty Calibration (OOD)']['Uncertainty'],b['Uncertainty Calibration (OOD)']['Optimal'],kind='linear',fill_value=np.nan,bounds_error=False)
                        x_u = np.array([0,0.00025,0.0005,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.0125,
                                        0.015,0.0175,0.02,0.025,0.030,0.035,0.040,0.050,0.060,
                                        0.070,0.075,0.08,0.09,0.1,0.15,0.20,0.25,0.30,0.35,
                                        0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1])
                        uncert_new_ood = pd.DataFrame({'Uncertainty':x_u,
                                                'True error':f1_ood(x_u),
                                                'Count':f2_ood(x_u),
                                                'Optimal': f3_ood(x_u)})
                        
                        uncert_new_ood['Model Type'] = [b['Model Type']]*len(uncert_new_ood)
                        uncert_new_ood['Method'] = [b['Method']]*len(uncert_new_ood)
                        uncert_new_ood['Calibration Method'] = [b['Calibration Method']]*len(uncert_new_ood)
                        uncert_new_ood['Seed'] = [b['Seed']]*len(uncert_new_ood)
                        uncert_new_ood['With_OOD'] = [b['With_OOD']]*len(uncert_new_ood)
                        
                        uncert_cali_ood = pd.concat([uncert_cali_ood,uncert_new_ood],axis=0)
                
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
                
                
    print(metrics_data.groupby(['Model Type','Method','Calibration Method','With_OOD']).count())
   


    # ! --------------------------- METRICS MEANS AND STDS ---------------------------------
    dataset = metrics_data
    #dataset['Model Type short'] = dataset['Model Type'].replace({'BayesianTripletLoss':'BTL'},regex=True)
    #dataset['Model'] = dataset['Model Type short'] + '/' + dataset['Method'] + '/' + dataset['Calibration Method']
    
    stats = (dataset.groupby(["Model Type","Method","Calibration Method","With_OOD"]).agg(["mean","std"])).drop(columns=["Seed",'Uncertainty_simple_thresh',"Uncertainty_best_thresh","AvU_best_thresh","AvU_simple_thresh"]).reset_index().set_index(['Model Type','Method','Calibration Method'])
    stats[stats['With_OOD']==True]['MaP']
    pdb.set_trace()

    # ! --------------------------- CLASS-WISE METRICS ---------------------------------
    
    if False:
        metrics = ['Average precision','acc_top1','Class Expected Calibration Error']
        ylabel = ['Average Precision','Accuracy','Class-wise ECE']

        for j,ood in enumerate([True,False]):
            
            dataset = class_data
            if ood == False:
                dataset = dataset[class_data['With_OOD']==ood]
            dataset = dataset.sort_values(['index'])
            if ood == True:
                hue_order = ['BTL_iso/kNN_gauss_kernel/None','BTL_iso/kNN_gauss_kernel/None_wOOD','BTL_iso/min_dist_NN/SWAG','BTL_iso/min_dist_NN/SWAG_wOOD']
                palette = [sns.color_palette("mako_r", as_cmap=False,n_colors=6)[3]]+[sns.color_palette("mako_r", as_cmap=False,n_colors=6)[2]]+[sns.color_palette("flare", as_cmap=False,n_colors=6)[2]]+[sns.color_palette("flare", as_cmap=False,n_colors=6)[1]]
            else:
                hue_order = ['BTL_iso/kNN_gauss_kernel/None','BTL_iso/min_dist_NN/SWAG','Classic/Vanilla/TempScaling']
                palette = sns.color_palette("mako_r", as_cmap=False,n_colors=2) + [sns.color_palette("magma_r", as_cmap=False,n_colors=2)[0]]
                
            fig, axes = plt.subplots(3,1,figsize=(12,16),sharex=True)
            
            for indx,metric in enumerate(metrics):
                ax = axes.flatten()[indx]
                dataset = dataset[dataset['Model Type']!='BayesianTripletLoss_diag']
                #dataset = dataset[dataset['Method'] != 'kNN_gauss_kernel']
                dataset = dataset[True != ((dataset['Model Type']=='BayesianTripletLoss_iso')&((dataset['Calibration Method']=='MCDropout')|(dataset['Calibration Method']=='Ensemble')))]
                dataset = dataset[True != ((dataset['Model Type']=='Classic')&((dataset['Calibration Method']=='MCDropout')|(dataset['Calibration Method']=='Ensemble')|(dataset['Calibration Method']=='SWAG')|(dataset['Calibration Method']=='None')))]
                
                dataset['Model Type short'] = dataset['Model Type'].replace({'BayesianTripletLoss':'BTL'},regex=True)
                
                if ood == False:
                    dataset['Model'] = dataset['Model Type short'] + '/' + dataset['Method']+ '/' + dataset['Calibration Method']
                else:
                    with_ood = []
                    for i in range(len(dataset)):
                        if dataset['With_OOD'].iloc[i]==True:
                            with_ood.append('_wOOD')
                        else:
                            with_ood.append('')
                    dataset['Model'] = dataset['Model Type short'] +'/' + dataset['Method']+ '/' + dataset['Calibration Method'] + with_ood
                    
                plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1,wspace=0.15,hspace=0.05)
                sns.barplot(y=metric,x='index',data=dataset,ax=ax,hue_order=hue_order,hue='Model',palette=palette,errorbar=("se",2))
                
                
                if indx != 2:
                    ax.set_xlabel(None)
                    ax.get_legend().remove()
                    ax.tick_params(axis='both', which='major', labelsize=14)

                else:
                    ax.legend(fontsize=13,title='$\\bf{Model}$',title_fontsize=15)
                    #ax.get_legend().get_title().set_text('$\\bf{Model}$')
                    ax.set_xlabel('Class',fontsize=20,weight='bold')
                    ax.xaxis.set_label_coords(0.5, -0.32) 
                    xlabels = dataset['index'].unique()
                    ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor",fontsize=14,ha='right',weight='bold')

                    #ax.tick_params(axis='both', which='major', labelsize=13)
                    
                for patch in ax.patches:
                    clr = patch.get_facecolor()
                    clr = list(clr)
                    clr[3] = 0.0
                    clr = tuple(clr)
                    patch.set_edgecolor(clr)
                    patch.set_linewidth(2.5)
                        
                ax.set_ylabel(ylabel[indx],fontsize=20, weight='bold')
                ax.yaxis.set_label_coords(-.075, .5) 
                
                
                    
            fig.savefig(f'./reports/test/classwise_{ood}.png')
                



    # ! --------------------------- UCE ---------------------------------

    if True:
        
        
        datasets = [uncertain_when_inacc,accurate_when_certain,calibration_vs_acc,uncert_cali,
                    uncert_cali_ood,
                    acc_inacc_certain]
        names = ['uncertain_when_inacc','accurate_when_certain','calibration_vs_acc','uncert_cali',
                 'uncert_cali_ood','acc_inacc_certain']
        x_vals = ['Uncertainty_thresh','Uncertainty_thresh','Confidence','Uncertainty',
                  'Uncertainty','Inaccurat_certain_ratio']
        y_vals = ['Recall_inacc','Accuracy','Accuracy','True error','True error','Accurat_certain_ratio']
        xlabel = ['Uncertainty threshold','Uncertainty threshold',r'Confidence $\tau$','Uncertainty',
                  'Uncertainty','Inaccurate Certain Rate (ICR)']
        ylabel = ['p(uncertain|inaccurate)','p(accurate|certain)',
                  r'Accuracy on examples: $p(y|x)\geq \tau$','True error','True error',
                  'Accurate Certain Rate (ACR)']
        title = [r'Uncertain when inaccurate $(\uparrow)$',r'Accurate when certain $(\uparrow)$',
                 r'Confidence vs. Accuracy $(\uparrow)$', 'UCE plot','UCE plot (OOD samples)',
                 r'ROC (using certainty) $(\uparrow)$']
        
        index = [0,1]
        
        fig, axes = plt.subplots(1,2,figsize=(15,5),sharey=True)
        
        for i,dataset in enumerate(datasets):
            if (i < 3) | (i==5):
                continue;
            ax = axes.flatten()[index[i-3]]
            #dataset = dataset[((dataset['Model Type']=='BayesianTripletLoss_iso') & (dataset['Method']=='kNN_gauss_kernel') & (dataset['Calibration Method']=='None')) |
            #                  ((dataset['Model Type']=='BayesianTripletLoss_iso') & (dataset['Method']=='min_dist_NN') & (dataset['Calibration Method']=='SWAG')) | 
            #                  ((dataset['Model Type']=='BayesianTripletLoss_iso') & (dataset['Method']=='kNN_gauss_kernel') & (dataset['Calibration Method']=='TempScaling')) | 
            #                  ((dataset['Model Type']=='Classic') & ((dataset['Calibration Method']=='TempScaling') | (dataset['Calibration Method']=='None'))) ]
            #dataset = dataset[dataset['Model Type']!='BayesianTripletLoss_diag']
            dataset = dataset[dataset['Model Type']=='Classic']
            if i != 4:
                dataset = dataset[dataset['With_OOD'] != True]
            else:
                dataset = dataset[dataset['With_OOD'] == True]
            dataset = dataset.reset_index()
            
            dataset['Model Type short'] = dataset['Model Type'].replace({'BayesianTripletLoss':'BTL'},regex=True)
            dataset['Model'] = dataset['Model Type short'] + '-' + dataset['Calibration Method']
            
            hue_order = ['kNN_gauss_kernel','min_dist_NN','Vanilla']
            style_order = ['None','TempScaling','SWAG']
            palette = sns.color_palette("magma_r", as_cmap=False,n_colors=2) + sns.color_palette("mako_r", as_cmap=False,n_colors=1)
            palette = [(0.278,0.612,0.620,1),(0.286,0.314,0.518,1),(0.871,0.447,0.439,1)]

            
            if (i != 3) & (i != 4):
                sns.lineplot(x=x_vals[i], y=y_vals[i],errorbar=("se",1), hue="Model", data=dataset,ax=ax,palette=palette,hue_order=hue_order,style="Model Type")
            else:
                #dataset['error_var'] = dataset['True error']*(1-dataset['True error'])/dataset['Count']
                #dataset['error_var_times_count'] = dataset['error_var']*dataset['Count']
                #dataset['True error_times_count'] = dataset['True error']*dataset['Count']

                #dataset_errors = dataset.groupby(['Model','Uncertainty'])['error_var_times_count'].agg('sum')
                #dataset_count = dataset.groupby(['Model','Uncertainty'])['Count'].agg('sum')
                #dataset_errors_se = (dataset_errors/dataset_count)*(1/2)

                if i == 3:
                    #dataset_errors_se = dataset_errors_se.loc[hue_order]
                    pdb.set_trace()
                    #sns.lineplot(x=x_vals[i], y=y_vals[i],errorbar=("se",2), hue="Method", data=dataset,ax=ax,palette=palette,hue_order=hue_order,style="Calibration Method",style_order=style_order,linewidth=3)
                    palette = sns.color_palette('husl')
                    sns.lineplot(x=x_vals[i], y=y_vals[i],errorbar=("se",2), hue="Calibration Method", data=dataset,ax=ax,palette=palette,hue_order=['None','MCDropout','SWAG','TempScaling','Ensemble'],linewidth=3)
                    sns.lineplot(x=x_vals[i], y='Optimal',color='black',data=dataset,ax=ax,label='Optimal',linestyle=':',linewidth=3)
                    
                else:
                    hue_order = ['kNN_gauss_kernel','min_dist_NN']
                    #dataset_errors_se = dataset_errors_se.loc[hue_order]
                    sns.lineplot(x=x_vals[i], y=y_vals[i],errorbar=("se", 2), hue="Method", data=dataset,ax=ax,palette=palette,hue_order=hue_order,linestyle='--',linewidth=2,style_order=style_order,style="Calibration Method")
                    sns.lineplot(x=x_vals[i], y='Optimal',color='black',data=dataset,ax=ax,label='Optimal',linestyle=':',linewidth=3)
                
                """    
                dataset_errors_se = dataset_errors_se.reset_index().set_index('Model')    
                dataset_mean = dataset.groupby(['Model','Uncertainty']).agg({'True error_times_count': lambda x: x.sum(skipna=True)})
                dataset_mean['True error'] = dataset_mean.values.flatten()/dataset_count.astype(float).values.flatten()
                dataset_mean = dataset_mean.drop(columns="True error_times_count")
                dataset_mean = dataset_mean.reset_index().set_index('Model')

                for indx,model in enumerate(dataset_errors_se.index.unique()):
                    mean = dataset_mean.loc[model]['True error']
                    x = dataset_mean.loc[model]['Uncertainty']
                    #pdb.set_trace()
                    ymin = mean-2*dataset_errors_se.loc[model].iloc[:,-1].astype(float)
                    ymax = mean+2*dataset_errors_se.loc[model].iloc[:,-1].astype(float)
                    ymin = np.maximum(0,ymin)
                    ymax = np.minimum(1,ymax)
                    ax.fill_between(x,ymin,ymax, color=palette[indx],alpha=0.15)
                """
           
            plt.subplots_adjust(left=0.075, right=0.8, top=0.9, bottom=0.15,wspace=0.1,hspace=0.1)
            ax.set_xlabel(xlabel[i],fontsize=14, weight='bold')
            ax.set_ylabel(ylabel[i],fontsize=14, weight='bold')
            ax.yaxis.set_label_coords(-.1, .5)
            ax.set_title(title[i],fontsize=16,weight='bold',y=1.02)
            ax.set_ylim(-0.02,1)
            
            if index[i-3] == 0:
                ax.legend(bbox_to_anchor=(2.62, 1), borderaxespad=0)
                ax.get_legend().get_texts()[0].set_text('$\\bf{Probability}$ $\\bf{Method}$')
                ax.get_legend().get_texts()[4].set_text('\n$\\bf{Calibration}$ $\\bf{Method}$')
                ax.get_legend().get_texts()[8].set_text('\n$\\bf{Optimal}$\n')
                for legobj in ax.get_legend().legendHandles:
                    legobj.set_linewidth(3.0)
            else:
                ax.get_legend().remove()
        
        
        fig.savefig(f'./reports/test/graphs.png')
    



    # ! --------------------------- ENTROPY GRAPHS ---------------------------------

    if False:
        entropies_ID_vs_OOD = entropies[entropies['With_OOD']==False]
        entropies_ID_vs_OOD = entropies_ID_vs_OOD[entropies_ID_vs_OOD['Model Type']!='BayesianTripletLoss_diag'].sort_values(['Model Type'])
        #entropies_ID_vs_OOD = entropies_ID_vs_OOD[entropies_ID_vs_OOD['Method']!='kNN_gauss_kernel']
        
        fig, ax = plt.subplots(3,5,figsize=(15,10),sharex=True,sharey=True)
        for i,model_type in enumerate(entropies_ID_vs_OOD['Method'].unique()):
            for j, cali_meth in enumerate(['None','MCDropout','SWAG','TempScaling', 'Ensemble']):
                #pdb.set_trace()
                palette = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=False,n_colors=6)
                palette2 = sns.color_palette("dark:salmon_r", as_cmap=False,n_colors=2)
                palette = [palette[2],palette2[0]]
                
                sns.kdeplot(data=entropies_ID_vs_OOD[(entropies_ID_vs_OOD['Method'] == model_type) & 
                                                    (entropies_ID_vs_OOD['Calibration Method'] == cali_meth)],
                            x="Entropy", hue="Entropy_dist",fill="Entropy_dist", common_norm=False, 
                            palette=palette,alpha=.5, linewidth=1,ax=ax[i,j],
                            hue_order=['ID','OOD'])
                
                # calculate the Bhattacharyya distance
                ID_samples = entropies_ID_vs_OOD[(entropies_ID_vs_OOD['Method'] == model_type) & (entropies_ID_vs_OOD['Calibration Method'] == cali_meth) & (entropies_ID_vs_OOD['Entropy_dist'] == 'ID')]['Entropy']
                OOD_samples = entropies_ID_vs_OOD[(entropies_ID_vs_OOD['Method'] == model_type) & (entropies_ID_vs_OOD['Calibration Method'] == cali_meth) & (entropies_ID_vs_OOD['Entropy_dist'] == 'OOD')]['Entropy']
                dist = Hellinger_distance(ID_samples,OOD_samples)
                ax[i,j].text(0.7, 0.5, f"{dist:.3}",transform=ax[i,j].transAxes,ha="center", va="center", rotation=0, size=15,bbox=dict(boxstyle="round,pad=0.3", fc=None, ec="b", lw=2,alpha=0.25))
    
                # Set title
                if j == 0:
                    if model_type == 'Vanilla':
                        ax[i,j].set_ylabel('Vanilla',fontsize = 15, weight='bold')
                        ax[i,j].yaxis.set_label_coords(-.25, .5)
                    elif model_type == 'min_dist_NN':
                        ax[i,j].set_ylabel('BTL_Iso\n(min_dist_NN)',fontsize = 15, weight='bold')
                        ax[i,j].yaxis.set_label_coords(-.3, .5)
                    elif model_type == 'kNN_gauss_kernel':
                        ax[i,j].set_ylabel('BTL_Iso\n(kNN_gauss_kernel)',fontsize = 15, weight='bold')
                        ax[i,j].yaxis.set_label_coords(-.3, .5)
                    
                    
                
                # Set y-axis
                if i == 0:
                    ax[i,j].set_title(cali_meth,fontsize = 14, weight='bold',y=1.05)
                    
        
        fig.savefig('./reports/test/entropy.png')








    """
    no_knn_metrics_data_without_OOD = no_knn_metrics_data_without_OOD[(no_knn_metrics_data_without_OOD['Model Type']=='BayesianTripletLoss_iso') & (no_knn_metrics_data_without_OOD['Method']=='kNN_gauss_kernel') & (no_knn_metrics_data_without_OOD['Calibration Method']=='None') |
                                                                      (no_knn_metrics_data_without_OOD['Model Type']=='BayesianTripletLoss_iso') & (no_knn_metrics_data_without_OOD['Method']=='min_dist_NN') & (no_knn_metrics_data_without_OOD['Calibration Method']=='SWAG') | 
                                                                      (no_knn_metrics_data_without_OOD['Model Type']=='Classic')]
    """


    # ! --------------------------- METRICS GRAPHS ---------------------------------

    all_metrics_data_without_OOD = metrics_data[metrics_data['With_OOD']==False]
    all_metrics_data_without_OOD = all_metrics_data_without_OOD.sort_values(['Model Type'])
    
    btl_metrics_data_without_OOD = metrics_data[metrics_data['Method']!='Vanilla']
    btl_metrics_data_without_OOD = btl_metrics_data_without_OOD[btl_metrics_data_without_OOD['With_OOD']==False]
    btl_metrics_data_without_OOD = btl_metrics_data_without_OOD.sort_values(['Model Type'])
    
    no_knn_metrics_data_without_OOD = metrics_data[metrics_data['With_OOD']==False]
    no_knn_metrics_data_without_OOD = no_knn_metrics_data_without_OOD[(no_knn_metrics_data_without_OOD['Model Type']!='BayesianTripletLoss_diag')]
    no_knn_metrics_data_without_OOD = no_knn_metrics_data_without_OOD.sort_values(['Model Type'])
    
    metrics_iso_with_OOD = metrics_data[metrics_data['Method']!='Vanilla']
    metrics_iso_with_OOD = metrics_iso_with_OOD[(metrics_iso_with_OOD['Model Type']!='BayesianTripletLoss_diag')]
    metrics_iso_with_OOD['Model Type'][metrics_iso_with_OOD['With_OOD']==True] = metrics_iso_with_OOD['Model Type'][metrics_iso_with_OOD['With_OOD']==True] + '_OOD'
    metrics_iso_with_OOD = metrics_iso_with_OOD.sort_values(['Method','Model Type'])

    
    if False:
        scale_fig = [1,1,1,1]
        figsize = [13.33,10,10,10]
        width_ratio = [[5,5,3.33],[5,5],[5,5],[5,5]]
        name = ['all','btl','no_knn','with_ood']
        data_sets = [all_metrics_data_without_OOD, btl_metrics_data_without_OOD,
                    no_knn_metrics_data_without_OOD,metrics_iso_with_OOD]
        order = [['kNN_gauss_kernel','min_dist_NN'],['kNN_gauss_kernel','min_dist_NN'],None,None]
        yticks = [[1.5,4,2.5,1.5,0.5,0.25,0.33,0.5],
                [1,1,2,2.5,0.15,0.15,0.2,0.15],
                [0.5,1,2.5,1,0.15,0.15,0.15,0.15],
                [1,1.5,1,2.0,0.2,0.2,0.2,0.1]]
        arrow_adjust_h = [0,-0.0095,-0.015,-0.01]
        
        # * PLOT ALL WITHOUT OOD
        # Plot map
        metrics = (['MaP', 'Accuracy top1', 'UCE', 'AUC_AvU', 'CECE', 'ACE', 'AECE', 'WECE'])
        arrows = ["->","->","<-","->","<-","<-","<-","<-"]
        arrow_adjust = [0,0,0.02,0.02,0,0.02,0.02,0.02]
        
        
        
        for k, data_set in enumerate(data_sets):
            if name[k]!='with_ood':
               continue;
            
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
                        hue_order=['None','MCDropout','SWAG','TempScaling']
                        colors = sns.cubehelix_palette(start=.5, rot=-.5,n_colors=num_calibrators+1, as_cmap=False,)
                        sns.set_palette([colors[i] for i in [0,1,2,3]])
                        g=sns.boxplot(x='Method',y=metric,hue='Calibration Method',data=df_type, ax=ax.flatten()[i+j*num_types],
                                hue_order=hue_order, linewidth=0.75,
                                order = order[k])


                        # Select which box you want to change
                        for p in range(36,42):
                            ax.flatten()[i+j*num_types].lines[p].set_color('red')
                            
                        for p in range(0,6):
                            ax.flatten()[i+j*num_types].lines[p].set_color('green')

                        g.patches[1].set_edgecolor('g')
                        g.patches[10].set_edgecolor('r')
                        
                        
                    
                    
                    if 'Ensemble' in df_type['Calibration Method'].value_counts().keys():
                        if df_type['Calibration Method'].value_counts()['Ensemble'] == 1:
                            y = df_type[metric][df_type['Calibration Method']=='Ensemble'].item()
                            ax.flatten()[i+j*num_types].axhline(y, 0.04, 0.96,label='Ensemble',lw=1,linestyle='--',color=colors[-1])
                        else:
                            try:
                                y1 = df_type[metric][(df_type['Calibration Method']=='Ensemble') &
                                                    (df_type['Method']=='kNN_gauss_kernel')].item()
                                y2 = df_type[metric][(df_type['Calibration Method']=='Ensemble') &
                                                    (df_type['Method']=='min_dist_NN')].item()
                            except:
                                pdb.set_trace()
                            
                            ax.flatten()[i+j*num_types].axhline(y1, 0.05, 0.46,lw=1,linestyle='--',color=colors[-1])
                            ax.flatten()[i+j*num_types].axhline(y2, 0.55, 0.96,lw=1,linestyle='--',color=colors[-1])

                        
                        
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
                            ax.flatten()[i+j*num_types].set_title('BTL (Isotropic)',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_diag':
                            ax.flatten()[i+j*num_types].set_title('BTL (Diagonal)',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_iso_OOD':
                            ax.flatten()[i+j*num_types].set_title('BTL (Isotropic, w. OOD)',fontsize = 15,y=1.05, weight='bold')
                        elif model_type == 'BayesianTripletLoss_diag_OOD':
                            ax.flatten()[i+j*num_types].set_title('BTL (Diagonal, w. OOD)',fontsize = 15,y=1.05, weight='bold')
                    
                    # Remove xticks 
                    if (j != len(metrics)-1):
                        ax.flatten()[i+j*num_types].set_xticklabels('')
                    # Adjust xticks    
                    ax.flatten()[i+j*num_types].set_xticks(ax.flatten()[i].get_xticks(),fontsize=13, rotation=0)
                    
                    
                    
                    
            fig.savefig(f'reports/test/{name[k]}.png')
    




    # ! --------------------------- RADAR GRAPHS ---------------------------------
    
    if False:
        dataset = metrics_data
        dataset = dataset[dataset['Model Type']!='BayesianTripletLoss_diag']
        dataset = dataset[True != ((dataset['Model Type']=='BayesianTripletLoss_iso')& (dataset['Method']=='min_dist_NN')&((dataset['Calibration Method']=='MCDropout')|(dataset['Calibration Method']=='Ensemble')|(dataset['Calibration Method']=='None')|(dataset['Calibration Method']=='TempScaling')))]
        dataset = dataset[True != ((dataset['Model Type']=='BayesianTripletLoss_iso')& (dataset['Method']=='kNN_gauss_kernel')&((dataset['Calibration Method']=='MCDropout')|(dataset['Calibration Method']=='Ensemble')|(dataset['Calibration Method']=='SWAG')|(dataset['Calibration Method']=='TempScaling')))]
        dataset = dataset[True != ((dataset['Model Type']=='Classic')&((dataset['Calibration Method']=='MCDropout')|(dataset['Calibration Method']=='Ensemble')|(dataset['Calibration Method']=='SWAG')|(dataset['Calibration Method']=='None')))]
        dataset['Model Type short'] = dataset['Model Type'].replace({'BayesianTripletLoss':'BTL'},regex=True)
        dataset['Model Name'] = dataset['Model Type short'] + '/' + dataset['Method'] + '/' + dataset['Calibration Method']
        stats = (dataset.groupby(["Model Name","With_OOD"]).agg(["mean"])).drop(columns=["Seed",'Uncertainty_simple_thresh',"Uncertainty_best_thresh","AvU_best_thresh","AvU_simple_thresh","Accuracy top2","Accuracy top3","Accuracy top5","UCE_OOD"]).reset_index()
        stats.columns = ['Model Name','With_OOD','mAP','Accuracy','AECE','WECE','CECE','ACE','UCE','AUC_AvU']
        
        ood_list = [False,True]
        for ood in ood_list:
            if ood == False:
                df = stats[stats['With_OOD']==ood].drop(columns=['With_OOD']).reset_index().drop(columns=['index'])
            else:
                df = stats
                
            df['mAP'] = df['mAP']*100
            df['Accuracy'] = df['Accuracy']*100
            df['AECE'] = df['AECE']*10000
            df['WECE'] = df['WECE']*10000
            df['CECE'] = df['CECE']*10000
            df['ACE'] = df['ACE']*10000
            df['UCE'] = df['UCE']*1000
            df['AUC_AvU'] = df['AUC_AvU']*100
            
            
            
                
            # ------- PART 2: Add plots
            
            # Plot each individual = each line of the data
            # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
            
            if ood == False:
                # ------- PART 1: Create background
            
                # number of variable
                categories=list(df)[1:]
                N = len(categories)
                categories_list = [r'$\bf{mAP}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)',r'$\bf{Accuracy}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)',
                                r'$\bf{AECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',r'$\bf{WECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',
                                r'$\bf{CECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',r'$\bf{ACE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',
                                r'$\bf{UCE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-3)',r'$\bf{AUC\_AvU}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)']
                
                # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]
                
                # Initialise the spider plot
                plt.figure(figsize=(6, 7))
                ax = plt.subplot(111, polar=True)
                
                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([10,20,30,40,50,60,70,80,90,100], ["10","20","30","40","50","60","70","80","90","100"], color="black", size=9,zorder=-5)
                plt.ylim(0,100)
                
                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], categories)
                ax.set_xticklabels(categories_list, fontsize=14,zorder=5)
                ax.tick_params(axis='x', which='major', pad=17,zorder=5)
                ax.yaxis.set_zorder(-10)
                for xtick in ax.get_xticklabels():
                    xtick.set_zorder(18)
                ax.spines[:].set_visible(False)
            
                # Ind1
                values=df[df['Model Name']=='BTL_iso/kNN_gauss_kernel/None'].drop(columns=['Model Name']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(kNN_gauss_kernel)\n$\\it{None}$',color=(0.278,0.612,0.62,1),zorder=6)
                ax.fill(angles, values, color=(0.278,0.612,0.62,1), alpha=0.1,zorder=6)
                
                # Ind2
                values=(df[df['Model Name']=='BTL_iso/min_dist_NN/SWAG']).drop(columns=['Model Name']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(min_dist_NN)\n$\\it{SWAG}$',color=(0.286,0.314,0.518,1),zorder=5)
                ax.fill(angles, values, color=(0.286,0.314,0.518,1), alpha=0.1,zorder=5)
                
                # Ind3
                values=df[df['Model Name']=='Classic/Vanilla/TempScaling'].drop(columns=['Model Name']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{Classic}$'+'\n(Vanilla)\n$\\it{TempScaling}$',color=(0.871,0.447,0.439,1),zorder=4)
                ax.fill(angles, values, color=(0.871,0.447,0.439,1), alpha=0.1,zorder=4)
                
                # Add legend
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=3)
                
                plt.tight_layout()

                # Show the graph
                fig = ax.get_figure()
                fig.savefig('./reports/test/radar_plot_vanilla.png')
            
            else:
                # ------- PART 1: Create background
            
                # number of variable
                categories=list(df)[2:]
                N = len(categories)
                categories_list = [r'$\bf{mAP}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)',r'$\bf{Accuracy}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)',
                                r'$\bf{AECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',r'$\bf{WECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',
                                r'$\bf{CECE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',r'$\bf{ACE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-4)',
                                r'$\bf{UCE}$ '+r'$\bf{[\downarrow]}$'+'\n'+r'($\times$1e-3)',r'$\bf{AUC\_AvU}$ '+r'$\bf{[\uparrow]}$'+'\n'+'(%)']
                
                # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]
                
                # Initialise the spider plot
                plt.figure(figsize=(6, 7))
                ax = plt.subplot(111, polar=True)
                
                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([10,20,30,40,50,60,70,80,90,100], ["10","20","30","40","50","60","70","80","90","100"], color="black", size=9,zorder=-5)
                plt.ylim(0,100)
                
                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], categories)
                ax.set_xticklabels(categories_list, fontsize=14,zorder=5)
                ax.tick_params(axis='x', which='major', pad=17,zorder=5)
                ax.yaxis.set_zorder(-10)
                for xtick in ax.get_xticklabels():
                    xtick.set_zorder(18)
                ax.spines[:].set_visible(False)
                
                # Ind1
                values=(df[(df['Model Name']=='BTL_iso/kNN_gauss_kernel/None')&(df['With_OOD']==False)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(kNN_gauss_kernel)\n$\\it{None}$',color=(0.243,0.367,0.517,1),zorder=5)
                ax.fill(angles, values, color=(0.243,0.367,0.517,1), alpha=0.1,zorder=6)
                
                # Ind1_ ood 
                values=(df[(df['Model Name']=='BTL_iso/kNN_gauss_kernel/None')&(df['With_OOD']==True)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(kNN_gauss_kernel)\n$\\it{None}$\n[w.OOD]',color=(0.294,0.581,0.660,1),zorder=6)
                ax.fill(angles, values, color=(0.294,0.581,0.660,1), alpha=0.1,zorder=6)
                
                # Add legend
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=2)
                plt.tight_layout()

                # Show the graph
                fig = ax.get_figure()
                fig.savefig('./reports/test/radar_plot_fast_ood.png')
                
                
                
                
                
                
                # Initialise the spider plot
                plt.figure(figsize=(6, 7))
                ax = plt.subplot(111, polar=True)
                
                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([10,20,30,40,50,60,70,80,90,100], ["10","20","30","40","50","60","70","80","90","100"], color="black", size=9,zorder=-5)
                plt.ylim(0,100)
                
                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], categories)
                ax.set_xticklabels(categories_list, fontsize=14,zorder=5)
                ax.tick_params(axis='x', which='major', pad=17,zorder=5)
                ax.yaxis.set_zorder(-10)
                for xtick in ax.get_xticklabels():
                    xtick.set_zorder(18)
                ax.spines[:].set_visible(False)
                
                # Ind2
                values=(df[(df['Model Name']=='BTL_iso/min_dist_NN/SWAG')&(df['With_OOD']==False)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(min_dist_NN)\n$\\it{SWAG}$',color=(0.685,0.325,0.385,1),zorder=5)
                ax.fill(angles, values, color=(0.685,0.325,0.385,1), alpha=0.1,zorder=5)
                
                # Ind2_ ood
                values=(df[(df['Model Name']=='BTL_iso/min_dist_NN/SWAG')&(df['With_OOD']==True)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(min_dist_NN)\n$\\it{SWAG}$\n[w.OOD]',color=(0.906,0.509,0.470,1),zorder=6)
                ax.fill(angles, values, color=(0.906,0.509,0.470,1), alpha=0.1,zorder=6)
                
                # Add legend
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=2)
                
                plt.tight_layout()

                # Show the graph
                fig = ax.get_figure()
                fig.savefig('./reports/test/radar_plot_best_ood.png')
                
                
                
                
                
                
                
                
                # Initialise the spider plot
                plt.figure(figsize=(6, 7))
                ax = plt.subplot(111, polar=True)
                
                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([10,20,30,40,50,60,70,80,90,100], ["10","20","30","40","50","60","70","80","90","100"], color="black", size=9,zorder=-5)
                plt.ylim(0,100)
                
                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], categories)
                ax.set_xticklabels(categories_list, fontsize=14,zorder=5)
                ax.tick_params(axis='x', which='major', pad=17,zorder=5)
                ax.yaxis.set_zorder(-10)
                for xtick in ax.get_xticklabels():
                    xtick.set_zorder(18)
                ax.spines[:].set_visible(False)
                
                # Ind1_ ood 
                values=(df[(df['Model Name']=='BTL_iso/kNN_gauss_kernel/None')&(df['With_OOD']==True)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(kNN_gauss_kernel)\n$\\it{None}$\n[w.OOD]',color=(0.294,0.581,0.660,1),zorder=6)
                ax.fill(angles, values, color=(0.294,0.581,0.660,1), alpha=0.1,zorder=6)
                
                # Ind2_ ood
                values=(df[(df['Model Name']=='BTL_iso/min_dist_NN/SWAG')&(df['With_OOD']==True)]).drop(columns=['Model Name','With_OOD']).values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=r'$\bf{BTL\_iso}$'+'\n(min_dist_NN)\n$\\it{SWAG}$\n[w.OOD]',color=(0.906,0.509,0.470,1),zorder=6)
                ax.fill(angles, values, color=(0.906,0.509,0.470,1), alpha=0.1,zorder=6)
                
                # Add legend
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=2)
                
                plt.tight_layout()

                # Show the graph
                fig = ax.get_figure()
                fig.savefig('./reports/test/radar_plot_best_vs_fast_ood.png')
            
            

if __name__ == '__main__':
    main()
    