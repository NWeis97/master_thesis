from src.models.image_classifier import ImageClassifier
import wandb
import matplotlib.pyplot as plt
import os
import pdb
import argparse
import numpy as np
import pandas as pd
from numpy import linalg as LA
from src.utils.helper_functions import print_PCA_tSNE_plot



def main(args):
    # Extract args
    classifier_model = args.model_name
    database_model = args.model_database
    balanced_classes = int(args.balanced_dataset)
    test_dataset = args.test_dataset
    num_NN = int(args.num_NN)
    num_MC = int(args.num_MC)
    
    # Init wandb
    wandb.init(
                project="master_thesis",
                entity="nweis97",
                config={"classifier_model": classifier_model, 
                        "database_model": database_model,
                        "balanced_classes": balanced_classes,
                        "test_dataset": test_dataset,
                        "num_NN": num_NN,
                        "num_MC": num_MC},
                job_type="Test"
        )
    
    # Extract classifier model
    classifier = ImageClassifier(classifier_model,database_model,balanced_classes)
    
    # Calculate probabilities
    probs_df, true_classes = classifier.get_probability_dist_dataset(test_dataset,num_NN,num_MC)
    
    # Get measures
    MaP, aP_df = classifier.calc_MaP(probs_df,true_classes)
    acc, acc_df = classifier.calc_accuracy(probs_df,true_classes)
    cali_plot_df, ECE, CECE, CECE_df = classifier.calc_ECE(probs_df,true_classes)
    
    # Log simple measures
    wandb.log({'MaP':MaP,
               'Accuracy': acc,
               'ECE': ECE,
               'CECE': CECE})
    
    wandb.Table(dataframe=pd.DataFrame(aP_df))
    wandb.Table(dataframe=pd.DataFrame(acc_df))
    wandb.Table(dataframe=pd.DataFrame(CECE_df))
    
    plot = visualize_embedding_space(classifier)



def visualize_embedding_space(classifier: ImageClassifier):
    # Extract embeddings for 30 objects from same class in embedding space
    num_classes_samples = 30  
    classes_unique = pd.sort(pd.unique(classifier.classes.numpy()))
    classes = []
    
    means_list = classifier.means.to('cpu').numpy()
    vars_list = classifier.vars.to('cpu').numpy()
    classes_list = classifier.classes.to('cpu').numpy()
    
    means = np.zeros(means_list.shape[0], num_classes_samples*len(classes_unique))
    vars = np.zeros(num_classes_samples*len(classes_unique),)
    count = 0
    for class_ in classes_unique:
        class_idxs = np.where(classes_list == class_)
        rand_idx_of_class = np.random.choice(class_idxs,num_classes_samples)
        for idx in rand_idx_of_class:
            means[:,count] = means_list[:,idx]
            vars[count] = vars_list[idx]
            classes.append(classes_list[idx])
            count += 1
    
    
    pdb.set_trace()
    # print embedding space
    fig_tsne_train, fig_pca_train = print_PCA_tSNE_plot(means,vars,classes,0,
                                                        classifier.params['var_prior'],'test')
    


if __name__ == '__main__':
    
    # Get configs
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--model-name", help="Name of model to test")
    args_parser.add_argument("--model-database", help="Name set of images to build model upon")
    args_parser.add_argument("--balanced-dataset", help="Balance the classes in model embedding space")
    args_parser.add_argument("--test-dataset", help="What dataset to test model on")
    args_parser.add_argument("--num-NN", help="Number of NN to base probs on")
    args_parser.add_argument("--num-MC", help="Number of MC samples to base probs on")
    args = args_parser.parse_args()
    
    
    main(args)

