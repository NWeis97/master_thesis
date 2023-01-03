# Standard libraries
import wandb
import json
import time
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import pdb
import ast
import os
import configparser
import argparse
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Own libraries
from src.loaders.tuple_loader import TuplesDataset
from src.models.image_classification_network import (ImageClassifierNet_BayesianTripletLoss, 
                                                     init_network)
from src.loss_functions.bayesian_triplet_loss import BayesianTripletLoss
from src.loaders.tuple_loader import TuplesDataset
from src.utils.helper_functions import (AverageMeter,  
                                        collate_tuples,
                                        check_config,
                                        get_logger)
from src.visualization.visualize import print_PCA_tSNE_plot
        
        
        
def train(train_loader: DataLoader, model: ImageClassifierNet_BayesianTripletLoss, 
          criterion: BayesianTripletLoss, optimizer: torch.optim, epoch: int, tuple_batch: int,
          update_every: int, print_freq: int, clip: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    backward_time = AverageMeter()
    losses = AverageMeter()
    nll_losses = AverageMeter()
    kl_losses = AverageMeter()

    # switch to train mode
    model.train()

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()

    for i, (input, target, classes) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        
        if i == 0:
            in_dim0 = input[0][0].shape[0]
            in_dim1 = input[0][0].shape[1]
            in_dim2 = input[0][0].shape[2]
            input_tensor = torch.zeros(nq*ni,in_dim0,in_dim1,in_dim2).cuda()
            output = torch.zeros(nq*ni,model.meta['outputdim']).cuda()
       
        for q in range(nq):
            input_tensor[q*ni:(q+1)*ni] = torch.stack(input[q])  
            
        output = model.forward_head(input_tensor)
        
        for q in range(nq):
            # accumulate gradients for one tuple at a time
            loss, nll_loss, kl_loss = criterion(output[q*ni:(q+1)*ni].T, target[q].cuda())
            losses.update(loss.item())
            nll_losses.update(nll_loss.item())
            kl_losses.update(kl_loss.item())
            
            # Retain graph until update
            end2 = time.time()
            if (model.fixed_backbone == True):
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            backward_time.update(time.time() - end2)

        if ((i + 1) % update_every == 0) | (i+1 == len(train_loader)):
            # Gradient clipping
            base_params = model.features.parameters()
            head_params = [i for i in model.parameters() if i not in base_params]
            torch.nn.utils.clip_grad_norm_(base_params , clip)
            torch.nn.utils.clip_grad_norm_(head_params , clip)
            
            # Gradient step
            optimizer.step()
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Detach loss
            loss.detach()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print update
        if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print(f'>> Train: [{epoch+1}][{tuple_batch+1}][{i+1}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)')

    return losses.avg, nll_losses.avg, kl_losses.avg



def validate(val_loader: DataLoader, model: ImageClassifierNet_BayesianTripletLoss, 
             criterion: BayesianTripletLoss, epoch: int, print_freq: int, with_swag: bool = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    nll_losses = AverageMeter()
    kl_losses = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, classes) in enumerate(val_loader):
            nq = len(input) # number of training tuples
            ni = len(input[0]) # number of images per tuple
            
            if i == 0:
                in_dim0 = input[0][0].shape[0]
                in_dim1 = input[0][0].shape[1]
                in_dim2 = input[0][0].shape[2]
                input_tensor = torch.zeros(nq*ni,in_dim0,in_dim1,in_dim2).cuda()
                output = torch.zeros(nq*ni,model.meta['outputdim']).cuda()
        
            for q in range(nq):
                input_tensor[q*ni:(q+1)*ni] = torch.stack(input[q])  
                
            # Get model output
            if with_swag == True:
                output = model.forward_head_with_swag(input_tensor)
            else:
                output = model.forward_head(input_tensor)
            
            for q in range(nq):
                # accumulate gradients for one tuple at a time
                loss, nll_loss, kl_loss = criterion(output[q*ni:(q+1)*ni].T, target[q].cuda())
                losses.update(loss.item())
                nll_losses.update(nll_loss.item())
                kl_losses.update(kl_loss.item())
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print update
            if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(val_loader):
                print(f'>> Validate: [{epoch+1}][{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)')

    return losses.avg, nll_losses.avg, kl_losses.avg
            
            
# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2022/03/29/discriminative-lr.html
def parameters_update_lr(net: nn.Module,
                         lr_optim: float):
    lr_mult = 0.5
    # save layer names
    layer_names = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        layer_names.append(name)
        
    layer_names.reverse()
    
    # placeholder
    parameters      = []
    prev_group_name = layer_names[0].split('.')[1]
    
    backbone_model = 0

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        
        # parameter group name
        backbone_or_heads = name.split('.')[0]
        cur_group_name = name.split('.')[1]
        
        # update learning rate
        if (cur_group_name != prev_group_name) & (backbone_or_heads == 'features'):
            if backbone_model == 0:
                lr_optim *= 0.01
                backbone_model = 1
            lr_optim *= lr_mult
        prev_group_name = cur_group_name
        
        # append layer parameters
        parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
                        'lr':     lr_optim}]
    
    return parameters





def main(config):
    # **********************************
    # ******** Initalize run ***********
    # **********************************
    
    # Check if configs has been run before
    #config_filename, config_run_no = check_config(config)
    
    # Get configs
    default_conf = config['default']
    print_freq = int(default_conf['print_freq'])
    save_model_freq = int(default_conf['save_model_freq'])
    hyper_conf = config['hyperparameters']
    model_conf = config['model']
    img_conf = config['image_transform']
    dataset_conf = config['dataset']
    
    # Init wandb
    wandb.init(
                project="master_thesis",
                entity="nweis97",
                config={"default": dict(default_conf), 
                        "hyperparameters": dict(hyper_conf),
                        "model": dict(model_conf),
                        "img_conf": dict(img_conf),
                        "dataset_conf": dict(dataset_conf)},
                job_type="Train"
        )
    
    # Get wandb name
    wandb_run_name = wandb.run.name
    print(wandb_run_name)
    print(wandb.run.dir[:-5])
    
    # get logger
    logger = get_logger(wandb_run_name)
    logger.info("STARTING NEW RUN")

    # **********************************
    # ******** Make datasets ***********
    # **********************************
    
    # Make image transformer
    img_size = int(img_conf['img_size'])
    normalize_mv = ast.literal_eval(img_conf['normalize'])
    norm_mean = normalize_mv['mean']
    norm_var = normalize_mv['var']
    transformer_valid = transforms.Compose([
                                            transforms.Resize((img_size,img_size)),
                                            transforms.PILToTensor(),
                                            transforms.ConvertImageDtype(torch.float),
                                            transforms.Normalize(torch.tensor(norm_mean),
                                                                torch.tensor(norm_var))
    ])
    if ast.literal_eval(img_conf['augmentation']):
        transformer_train = transforms.Compose([
                                                transforms.Resize((img_size+20,img_size+20)),
                                                transforms.RandomCrop(img_size),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomRotation((-5,5)),
                                                transforms.ColorJitter(brightness=0.2, 
                                                                       hue=0.1,
                                                                       contrast=0.1,
                                                                       saturation=0.05),
                                                transforms.GaussianBlur(kernel_size=(5, 9), 
                                                                        sigma=(0.1, 0.5)),
                                                transforms.PILToTensor(),
                                                transforms.ConvertImageDtype(torch.float),
                                                transforms.Normalize(torch.tensor(norm_mean),
                                                                    torch.tensor(norm_var))
        ])
    else:
        transformer_train = transformer_valid
    
    
    # Make datasets
    logger.info(">> Initilizing datasets")
    classes_not_trained_on = ast.literal_eval(dataset_conf['classes_not_trained_on'])
    num_classes = 20 - len(classes_not_trained_on)
    nnum = int(dataset_conf['nnum'])
    
    if default_conf['training_dataset'] == 'train':
        logger.info(">> Training model with Train data")
        mode_train = 'train'
        mode_val = 'val'
    elif default_conf['training_dataset'] == 'trainval':
        logger.info(">> Training model with Train amnd Validation data")
        mode_train = 'trainval'
        mode_val = 'test'
    else:
        raise ValueError("training_dataset unknown -> choose either train or trainval")
    
    ds_train = TuplesDataset(mode=mode_train, 
                            nnum=nnum, 
                            qsize_class=int(dataset_conf['qsize_class']),
                            poolsize_class=int(dataset_conf['poolsize_class']),
                            npoolsize=int(dataset_conf['npoolsize']),
                            transform=transformer_train,
                            keep_prev_tuples=ast.literal_eval(dataset_conf['keep_prev_tuples']),
                            classes_not_trained_on=classes_not_trained_on,
                            approx_similarity=ast.literal_eval(dataset_conf['approx_similarity']))
    ds_val = TuplesDataset(mode=mode_val, 
                              nnum=nnum, 
                              qsize_class=1000,
                              poolsize_class=int(dataset_conf['poolsize_class']),
                              npoolsize=int(dataset_conf['npoolsize'])*10,
                              transform=transformer_valid,
                              keep_prev_tuples=False,
                              classes_not_trained_on=classes_not_trained_on,
                              approx_similarity=ast.literal_eval(dataset_conf['approx_similarity']))
    
    
    # ************************************* 
    # ******** Initialize model ***********
    # *************************************
    logger.info(">> Initilizing model")
    params = {'model_type': 'BayesianTripletLoss',          
                        'seed':int(default_conf['seed']),
                        'architecture':ast.literal_eval(model_conf['architecture']),
                        'fixed_backbone': ast.literal_eval(model_conf['fixed_backbone']),
                        'const_eval_mode': ast.literal_eval(model_conf['const_eval_mode']),
                        'head_layers_dim': ast.literal_eval(model_conf['head_layers_dim']),
                        'activation_fn': ast.literal_eval(model_conf['activation_fn']),
                        'pooling': ast.literal_eval(model_conf['pooling']),
                        'dim_out': int(model_conf['dim_out']),
                        'dropout': float(model_conf['dropout']),
                        'classes_not_trained_on': classes_not_trained_on,
                        'img_size': int(img_conf['img_size']),
                        'normalize_mv' : ast.literal_eval(img_conf['normalize']),
                        'var_prior': float(hyper_conf['var_prior']),
                        'var_type': ast.literal_eval(model_conf['var_type']),
                        'with_swag': ast.literal_eval(model_conf['with_swag']),
                        'trained_on': mode_train}
    net = init_network('BayesianTripletLoss',params)
    net.cuda()
    mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"Total memory consumption for storing model "+
                f"on Cuda: {mem_cuda:.3f}GB")
    
    # Save parameters to json file
    with open(f"./models/params/{wandb_run_name}.json", "w") as outfile:
        json.dump(params, outfile)
    wandb.save(f'./models/params/{wandb_run_name}.json', policy="now")
    
    # ***************************************
    # ******** Initialize loaders ***********
    # ***************************************
    # Init tuples
    logger.info(">> Initilizing tuples")
    ds_train.update_backbone_repr_pool(net)
    ds_val.update_backbone_repr_pool(net)
    ds_train.create_epoch_tuples(net,nnum)
    ds_val.create_epoch_tuples(net,nnum)
    
    train_loader = DataLoader(
            ds_train, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            sampler=None, drop_last=True, collate_fn=collate_tuples
        )
    val_loader = DataLoader(
            ds_val, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            sampler=None, drop_last=False, collate_fn=collate_tuples
        )
    
    mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"Total memory consumption for storing model and data "+
                f"on Cuda: {mem_cuda:.3f}GB")
    
    
    # ********************************
    # ******** Train model ***********
    # ********************************
    
    # init values
    num_epochs = int(hyper_conf['epochs'])
    kl_scale_init = float(hyper_conf['kl_scale_init'])
    kl_scale_end = float(hyper_conf['kl_scale_end'])
    kl_warmup = int(np.ceil(float(hyper_conf['kl_warmup'])*float(num_epochs)))
    #kl_frac = (kl_scale_end/kl_scale_init)**(1/(kl_warmup-1))
    kl_div = (kl_scale_end-kl_scale_init)/(kl_warmup)
    lr_init = float(hyper_conf['lr_init'])
    lr_end = float(hyper_conf['lr_end'])
    #lr_diff = (lr_end-lr_init)*(1/(num_epochs-kl_warmup))
    lr_frac = (lr_end/lr_init)**(1/max(1,num_epochs-kl_warmup))
    margin_fixed = ast.literal_eval(hyper_conf['margin_fixed'])
    margin_val = float(hyper_conf['margin_val'])
    var_prior = float(hyper_conf['var_prior'])
    clip = float(hyper_conf['clip'])
    update_every = int(hyper_conf['update_every'])
    valid_margin_val = margin_val
    valid_kl_scale = kl_scale_end
    neg_class_dist = ast.literal_eval(dataset_conf['neg_class_dist'])
    var_type = ast.literal_eval(model_conf['var_type']) 
    update_pool_num = int(hyper_conf['update_pool_every'])
    skip_closest_neg_prob = float(hyper_conf['skip_closest_neg_prob'])
    
    update_pool_count = 0
    
    num_train_per_update = int(ds_train.average_class_size)//int(dataset_conf['qsize_class'])
    logger.info(f"Number of times new tuples created per epoch: {num_train_per_update}")
    
    for i in range(num_epochs):
        logger.info(f"\n\n########## Training {i+1}/{num_epochs} ##########")
        
        # *********************************
        # ******** Create Tuples **********
        # *********************************
        
        # create tuples for training
        if neg_class_dist == 'free':
            neg_class_max = nnum
        elif neg_class_dist == 'unif_to_free': 
            neg_class_max = int(np.floor(nnum/(np.max([1,num_epochs-kl_warmup]))*np.max([i+1-kl_warmup,0])) + 
                                 1 + np.floor(nnum/(num_classes-1)))
        else:
            neg_class_max = int(1 + np.floor(nnum/(num_classes-1)))
          
        logger.info(f"Maximum number of negatives per class: {neg_class_max}")
        # *****************************
        # ******** Training ***********
        # *****************************
               
        # update loss and optimizer
        if margin_fixed:
            margin = margin_val
            #kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            kl_scale_factor = kl_scale_init+kl_div*min([i,kl_warmup])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor,
                                            var_type=var_type)
        else:
            margin = avg_neg_distance_train*margin_val
            #kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            kl_scale_factor = kl_scale_init+kl_div*min([i,kl_warmup])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor,
                                            var_type=var_type)
        lr_optim = max(lr_init*lr_frac**max(0,i+1-kl_warmup),lr_end)
        #lr_optim = max(lr_init+(lr_diff*(max(0,i+1-kl_warmup))),lr_end)
        base_params = net.features.parameters()
        head_params = [i for i in net.parameters() if i not in base_params]

        logger.info(f'Updating margin ({margin:.2e}) and kl_scale_factor ({kl_scale_factor:.2e})')
        
        # train model
        train_loss = 0
        nll_loss_train = 0
        kl_loss_train = 0
        for j in range(num_train_per_update):
            
            # Avoid sensitive reaction to increase in kl_frac
            if i < kl_warmup:
                lr_start = np.sqrt(lr_optim*lr_end)
                lr_optim_sensitive = lr_start + (lr_optim-lr_start)/(num_train_per_update-1)*(j)
                optim = AdamW([
                            {'params': head_params},
                            {'params': base_params, 'lr': lr_optim_sensitive*0.01}, 
                        ],lr=lr_optim_sensitive, weight_decay=1e-1)
            else:
                optim = AdamW([
                            {'params': head_params},
                            {'params': base_params, 'lr': lr_optim*0.01}, 
                        ],lr=lr_optim, weight_decay=1e-1)
            
            # Make new tuples
            (avg_neg_distance_train,
                qvecs_train,
                qvars_train,
                classes_train) = train_loader.dataset.create_epoch_tuples(net,neg_class_max,
                                                                   skip_closest_neg_prob)
            # Train on tuples
            train_loss_b, nll_loss_train_b, kl_loss_train_b = train(train_loader,net,criterion,optim,
                                                            i,j, update_every, print_freq, 
                                                            clip)
            # append 
            train_loss += train_loss_b
            nll_loss_train += nll_loss_train_b
            kl_loss_train += kl_loss_train_b
        
        train_loss = train_loss/num_train_per_update
        nll_loss_train = nll_loss_train/num_train_per_update
        kl_loss_train = kl_loss_train/num_train_per_update
        
        # *******************************
        # ******** Validation ***********
        # *******************************
        # create tuples for validation
        (avg_neg_distance_val,
         qvecs_val,
         qvars_val,
         classes_val) = val_loader.dataset.create_epoch_tuples(net,
                                                        int(1 + np.floor(nnum/(num_classes-1))), #uniform
                                                        0.0)
        
        # validate model
        criterion = BayesianTripletLoss(margin=valid_margin_val,
                                        varPrior=var_prior,
                                        kl_scale_factor=valid_kl_scale,
                                        var_type=var_type)

        val_loss, nll_loss_val, kl_loss_val = validate(val_loader,net,criterion,i,print_freq)
        
        
        # **************************************************
        # ******** Create Embedding Space Figures **********
        # **************************************************
        if i % 1 == 0:
            train_means = qvecs_train.T.to('cpu').numpy()
            train_vars = np.mean(qvars_train.T.to('cpu').numpy(),axis=0)
            
            val_means = qvecs_val.T.to('cpu').numpy()
            val_vars = np.mean(qvars_val.T.to('cpu').numpy(),axis=0)
            
            # Select random embeddings from 
            rand_idx_train = []
            rand_idx_val = []
            classes_train_list = []
            classes_val_list = []
            for key in classes_train:
                # train 
                n_class_train = len(classes_train[key])
                rand_idx = np.random.permutation(n_class_train)[:min([20,n_class_train])]
                rand_idx_train.extend(classes_train[key][rand_idx])
                classes_train_list.extend([key]*min([20,n_class_train]))
                
                # test
                n_class_val = len(classes_val[key])
                rand_idx = np.random.permutation(n_class_val)[:min([20,n_class_val])]
                rand_idx_val.extend(classes_val[key][rand_idx])
                classes_val_list.extend([key]*min([20,n_class_val]))
            
              
            train_means = train_means[:,rand_idx_train]
            train_vars = train_vars[rand_idx_train]
            val_means = val_means[:,rand_idx_val]
            val_vars = val_vars[rand_idx_val]
            
            # print embedding space
            fig_tsne_train, fig_pca_train = print_PCA_tSNE_plot(train_means, train_vars,
                                            classes_train_list,i,var_prior,'train')
            
            # print embedding space
            fig_tsne_val, fig_pca_val = print_PCA_tSNE_plot(val_means, val_vars,
                                                    classes_val_list,i,var_prior,'val')
            
            # Log figures
            wandb.log(
                {
                    "tSNE_train": wandb.Image(fig_tsne_train),
                    "PCA_train": wandb.Image(fig_pca_train),
                    "tSNE_val": wandb.Image(fig_tsne_val),
                    "PCA_val": wandb.Image(fig_pca_val),
                }
            )
        
        
        # *******************************
        # ******** Log results **********
        # *******************************
        
        logger.info(f'\n\t***** Epoch: {i+1}\t\t Train loss: {train_loss*1000:.4f}\t\t'+
                    f'Val loss: {val_loss*1000:.4f} ******')
        
        # Log diagnostics
        wandb.log(
                {
                    "Training_loss": train_loss*1000,
                    "Validation_loss": val_loss*1000,
                    "Epoch": i+1,
                    "kl_scale_factor": kl_scale_factor,
                    "lr_optim": lr_optim,
                    "avg_dist_neg_train": avg_neg_distance_train,
                    "avg_dist_neg_val": avg_neg_distance_val,
                    "nll_loss_train": nll_loss_train*1000,
                    "nll_loss_val": nll_loss_val*1000,
                    "kl_loss_train": kl_loss_train*1000*kl_scale_factor,
                    "kl_loss_val": kl_loss_val*1000*valid_kl_scale,     
                }
            )
        
        # Save model on W&B and remove locally
        if ((i+1)%save_model_freq == 0) | (i == num_epochs-1):
            logger.info('Saving the model')
            torch.save(net.state_dict(),f'./models/state_dicts/{wandb_run_name}.pt')
            wandb.save(f'./models/tmp_models/{wandb_run_name}.pt', policy="now")
            
            
        # *********************************
        # ******** Update Pool ************
        # *********************************
        if (update_pool_count == update_pool_num) & (i != num_epochs-1):
            train_loader.dataset.update_backbone_repr_pool(net)
            
            update_pool_count = 0
        else:
            update_pool_count += 1
            
          
        # ********************************************
        # ******** Save SWAG Init weights ************
        # ********************************************
        if params['with_swag'] == True:
            if i+1 == int(np.ceil(num_epochs*1/2)):
                
                base_params = net.features.parameters()
                head_params = [i for i in net.parameters() if i not in base_params]
                swag_init_mean_param = []
                swag_init_var_param = []
                for idx, (name, param) in enumerate(net.named_parameters()):
                    if name.split('.')[0] != 'features':
                        if name.split('_')[0] == 'mean':
                            swag_init_mean_param.append(param.data.clone())
                        else:
                            swag_init_var_param.append(param.data.clone())

                lr_end_swag = lr_optim
            
            

    # *******************************
    # ******** With SWAG ************
    # *******************************
    if params['with_swag'] == True:
        head_params_swag_mean = []
        head_params_swag_var = []
        head_params_mean__mean = [] # Mean of mean head
        head_params_mean__var = [] # Mean of var head
        head_params_var__mean = [] # Var of mean head
        head_params_var__var = [] # Var of var head
        logger.info(f'>> Extracting param values for SWAG')
        
        #Reset lr
        lr_init_swag = (lr_init+lr_end_swag)/2
        lr_frac_swag = (lr_end_swag/lr_init_swag)**(1/(num_train_per_update*2-1))
        
        #Reset weights to swag init
        count_mean = 0
        count_var = 0
        for idx, (name, param) in enumerate(net.named_parameters()):
            if name.split('.')[0] != 'features':
                if name.split('_')[0] == 'mean':
                    param.data = swag_init_mean_param[count_mean]
                    count_mean += 1
                else:
                    param.data = swag_init_var_param[count_var]
                    count_var += 1
    
        # Extract SWAG weights
        for j in range(20):
            # Only save model weights every two epochs
            if ((j+1) % 2 == 0) | (j == 0):
                base_params = net.features.parameters()
                head_params = [i for i in net.parameters() if i not in base_params]
                tmp_mean_param = []
                tmp_var_param = []
                for idx, (name, param) in enumerate(net.named_parameters()):
                    if name.split('.')[0] != 'features':
                        if name.split('_')[0] == 'mean':
                            tmp_mean_param.append(param.data.clone())
                        else:
                            tmp_var_param.append(param.data.clone())
                head_params_swag_mean.append(tmp_mean_param)
                head_params_swag_var.append(tmp_var_param)
        
            # train model
            for k in range(num_train_per_update):
                lr_optim = lr_init_swag*(lr_frac_swag**(k+k*j%2))
                
                optim = AdamW([
                            {'params': head_params},
                            {'params': base_params, 'lr': lr_optim*0.01}, 
                        ],lr=lr_optim, weight_decay=1e-1,)
                
                
                # Make new tuples
                (avg_neg_distance_train,
                    qvecs_train,
                    qvars_train,
                    classes_train) = train_loader.dataset.create_epoch_tuples(net,neg_class_max,
                                                                    skip_closest_neg_prob)
                # Train on tuples
                train(train_loader,net,criterion,optim,j,k, update_every, print_freq,clip)
            
            
            logger.info(f'>>>>> {j+1}/{20}')
            if (j+1 != 20):
                train_loader.dataset.update_backbone_repr_pool(net)
                
                
                
                
        # Get mean of params (mean head)
        for j in range(len(head_params_swag_mean[0])):
            params__mean = head_params_swag_mean[0][j].clone()
            for k in range(len(head_params_swag_mean)-1):
                params__mean += head_params_swag_mean[k+1][j]
        
            head_params_mean__mean.append(params__mean / len(head_params_swag_mean))
            
        # Get mean of params (var head)
        for j in range(len(head_params_swag_var[0])):
            params__var = head_params_swag_var[0][j].clone()
            for k in range(len(head_params_swag_var)-1):
                params__var += head_params_swag_var[k+1][j]

            head_params_mean__var.append(params__var / len(head_params_swag_var))
            
        # Get var of params (mean head)
        for j in range(len(head_params_swag_mean[0])):
            var_params__mean = ((head_params_swag_mean[0][j]-head_params_mean__mean[j])**2).clone()
            for k in range(len(head_params_swag_mean)-1):
                var_params__mean += (head_params_swag_mean[k+1][j]-head_params_mean__mean[j])**2
            
            head_params_var__mean.append((var_params__mean / len(head_params_swag_mean))**(1/2))
            
        # Get var of params (var head)
        for j in range(len(head_params_swag_var[0])):
            var_params__var = ((head_params_swag_var[0][j]-head_params_mean__var[j])**2).clone()
            for k in range(len(head_params_swag_var)-1):  
                var_params__var += (head_params_swag_var[k+1][j]-head_params_mean__var[j])**2
            
            head_params_var__var.append((var_params__var / len(head_params_swag_var))**(1/2))
        
        # Save results
        net.head_mean__mean = head_params_mean__mean
        net.head_mean__var = head_params_mean__var
        net.head_std__mean = head_params_var__mean
        net.head_std__var = head_params_var__var
        
        logger.info('Saving SWAG headers')
        torch.save(net.head_mean__mean,f'./models/swag_headers/{wandb_run_name}_mean_swag__mean.pt')
        torch.save(net.head_mean__var,f'./models/swag_headers/{wandb_run_name}_mean_swag__var.pt')
        torch.save(net.head_std__mean,f'./models/swag_headers/{wandb_run_name}_std_swag__mean.pt')
        torch.save(net.head_std__var,f'./models/swag_headers/{wandb_run_name}_std_swag__var.pt')
    
    
    # ********************************
    # ******** Finish up run *********
    # ********************************    
    wandb.save(f'./logs/training_test/train_model/{wandb_run_name}.log') 
    
    # Remove local files
    wandb_local_dir = wandb.run.dir[:-5]
    wandb.finish()
    os.remove(f'./logs/training_test/train_model/{wandb_run_name}.log')
    shutil.rmtree(wandb_local_dir, ignore_errors=True)
    
    
        
if __name__ == '__main__':
    
    # Get configs
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config-filename", help="What config file to use")
    args = args_parser.parse_args()
    config = configparser.ConfigParser()
    config.read(f'configs/{args.config_filename}.ini')
    
    # Set random seed
    torch.random.manual_seed(int(config['default']['seed']))
    
    main(config)
