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
from src.loaders.tuple_loader4 import TuplesDataset_2class
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
          criterion: BayesianTripletLoss, optimizer: torch.optim, epoch: int, 
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

        for q in range(nq):
            output = torch.zeros(model.meta['outputdim'], ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                input[q][imi] = input[q][imi][None,:,:,:] #Expand dim
                output[:, imi] = model.forward_head(input[q][imi]).squeeze()
            
            # accumulate gradients for one tuple at a time
            loss, nll_loss, kl_loss = criterion(output, target[q].cuda())
            losses.update(loss.item())
            nll_losses.update(nll_loss.item())
            kl_losses.update(kl_loss.item())
            
            # Retain graph until update
            end2 = time.time()
            if (model.fixed_backbone == True):
                loss.backward()
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
            print(f'>> Train: [{epoch+1}][{i+1}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)')

    return losses.avg, nll_losses.avg, kl_losses.avg



def validate(val_loader: DataLoader, model: ImageClassifierNet_BayesianTripletLoss, 
             criterion: BayesianTripletLoss, epoch: int, print_freq: int):
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
            # measure data loading time
            data_time.update(time.time() - end)

            nq = len(input) # number of training tuples
            ni = len(input[0]) # number of images per tuple

            for q in range(nq):
                output = torch.zeros(model.meta['outputdim'], ni).cuda()
                for imi in range(ni):
                    # compute output vector for image imi
                    input[q][imi] = input[q][imi][None,:,:,:] #Expand dim
                    output[:, imi] = model.forward_head(input[q][imi].cuda()).squeeze()

                # accumulate gradients for one tuple at a time
                loss, nll_loss, kl_loss = criterion(output, target[q].cuda())
                #loss /= nq # get mean across batch
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
        pdb.set_trace()
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
    ds_train = TuplesDataset_2class(mode='train', 
                            nnum=nnum, 
                            qsize_class=int(dataset_conf['qsize_class']),
                            poolsize_class=int(dataset_conf['poolsize_class']),
                            transform=transformer_train,
                            keep_prev_tuples=ast.literal_eval(dataset_conf['keep_prev_tuples']),
                            classes_not_trained_on=classes_not_trained_on,
                            approx_similarity=ast.literal_eval(dataset_conf['approx_similarity']))
    ds_val = TuplesDataset_2class(mode='val', 
                              nnum=nnum, 
                              qsize_class=int(dataset_conf['qsize_class']),
                              poolsize_class=int(dataset_conf['poolsize_class']),
                              transform=transformer_valid,
                              keep_prev_tuples=False,
                              classes_not_trained_on=classes_not_trained_on,
                              approx_similarity=ast.literal_eval(dataset_conf['approx_similarity']))
    
    
    # ************************************* 
    # ******** Initialize model ***********
    # *************************************
    logger.info(">> Initilizing model")
    params = {'model_type': 'BayesianTripletLoss',          
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
                        'var_type': ast.literal_eval(model_conf['var_type'])}
    net = init_network('BayesianTripletLoss',params)
    net.cuda()
    
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
            sampler=None, drop_last=False, collate_fn=collate_tuples
        )
    val_loader = DataLoader(
            ds_val, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            sampler=None, drop_last=False, collate_fn=collate_tuples
        )
    
    # ********************************
    # ******** Train model ***********
    # ********************************
    
    # init values
    num_epochs = int(hyper_conf['epochs'])
    kl_scale_init = float(hyper_conf['kl_scale_init'])
    kl_scale_end = float(hyper_conf['kl_scale_end'])
    kl_warmup = int(float(hyper_conf['kl_warmup'])*float(num_epochs))
    kl_frac = (kl_scale_end/kl_scale_init)**(1/(kl_warmup-1))
    lr_init = float(hyper_conf['lr_init'])
    lr_end = float(hyper_conf['lr_end'])
    lr_diff = (lr_end-lr_init)*(1/(num_epochs-kl_warmup))
    #lr_diff = (lr_end-lr_init)*(1/(50-int(0.25*50.0)))
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
    
    mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"Total memory consumption on Cuda: {mem_cuda:.3f}GB")
    
    for i in range(num_epochs):
        logger.info(f"\n\n########## Training {i+1}/{num_epochs} ##########")
        
        # *********************************
        # ******** Create Tuples **********
        # *********************************
        
        # create tuples for training
        if neg_class_dist == 'free':
            neg_class_max = nnum
        elif neg_class_dist == 'unif_to_free': 
            neg_class_max = int(np.ceil(nnum/num_epochs*(i+1)) + np.floor(nnum/(num_classes-1)))
        else:
            neg_class_max = int(1 + np.floor(nnum/(num_classes-1)))
          
            
        (avg_neg_distance_train,
         qvecs_train,
         qvars_train,
         classes_train) = train_loader.dataset.create_epoch_tuples(net,neg_class_max,
                                                                   skip_closest_neg_prob)
        
        # *****************************
        # ******** Training ***********
        # *****************************
               
        # update loss and optimizer
        if margin_fixed:
            margin = margin_val
            kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor,
                                            var_type=var_type)
        else:
            margin = avg_neg_distance_train*margin_val
            kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor,
                                            var_type=var_type)
        lr_optim = max(lr_init+(lr_diff*(max(0,i+1-kl_warmup))),lr_end)
        base_params = net.features.parameters()
        head_params = [i for i in net.parameters() if i not in base_params]
        optim = AdamW([
                        {'params': head_params},
                        {'params': base_params, 'lr': lr_optim*0.01}, 
                      ],lr=lr_optim, weight_decay=1e-1)

        logger.info(f'Updating margin ({margin:.2e}) and kl_scale_factor ({kl_scale_factor:.2e})')
        
        # train model
        train_loss, nll_loss_train, kl_loss_train = train(train_loader,net,criterion,optim,
                                                          i, update_every, print_freq, 
                                                          clip)
        
        # *******************************
        # ******** Validation ***********
        # *******************************
        # create tuples for validation
        (avg_neg_distance_val,
         qvecs_val,
         qvars_val,
         classes_val) = val_loader.dataset.create_epoch_tuples(net,neg_class_max,0.0)
        
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
            train_means = qvecs_train.to('cpu').numpy()
            train_vars = np.mean(qvars_train.to('cpu').numpy(),axis=0)
            
            val_means = qvecs_val.to('cpu').numpy()
            val_vars = np.mean(qvars_val.to('cpu').numpy(),axis=0)
            
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
                    "kl_loss_train": kl_loss_train*1000*kl_frac,
                    "kl_loss_val": kl_loss_val*1000*kl_frac,     
                }
            )
        
        # Save model on W&B and remove locally
        if (i+1)%save_model_freq == 0:
            logger.info('Saving the model')
            torch.save(net.state_dict(),f'./models/tmp_models/{wandb_run_name}.pt')
            wandb.save(f'./models/tmp_models/{wandb_run_name}.pt', policy="now")
            
            
        # *********************************
        # ******** Update Pool ************
        # *********************************
        if (update_pool_count == update_pool_num):
            train_loader.dataset.update_backbone_repr_pool(net)
            
            update_pool_count = 0
        else:
            update_pool_count += 1

        
    wandb.save(f'./logs/training_test/train_model/{wandb_run_name}.log') 
    
    # Remove local files
    wandb_local_dir = wandb.run.dir[:-5]
    wandb.finish()
    os.remove(f'./logs/training_test/train_model/{wandb_run_name}.log')
    os.remove(f'./models/tmp_models/{wandb_run_name}.json')
    os.remove(f'./models/tmp_models/{wandb_run_name}.pt')
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
