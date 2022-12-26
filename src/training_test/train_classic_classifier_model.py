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
from src.loaders.classic_loader import Pooling_Dataset
from src.models.image_classification_network import (ImageClassifierNet_Classic, 
                                                     init_network)
from src.loss_functions.bayesian_triplet_loss import BayesianTripletLoss
from src.utils.helper_functions import (AverageMeter,  
                                        get_logger)
from src.visualization.visualize import print_PCA_tSNE_plot
        
        
        
def train(train_loader: DataLoader, model: ImageClassifierNet_Classic, 
          criterion: BayesianTripletLoss, optimizer: torch.optim, epoch: int, 
          update_every: int, print_freq: int, clip: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    backward_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode
    model.train()

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target, classes) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        output = model.forward_head(input)
        loss = criterion(output, target.cuda())
        losses.update(loss.item())
        
        # Get accuracy
        _, out_idx = output.max(dim=1)
        accuracy.update(torch.sum(out_idx.cpu() == target).item()/len(classes))
        
        end2 = time.time()
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print update
        if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print(f'>> Train: [{epoch+1}][{i+1}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)\t'
                  f'Accuracy {(accuracy.val)*100:.3f}% ({(accuracy.avg)*100:.3f}%)')

    return losses.avg, accuracy.avg



def validate(val_loader: DataLoader, model: ImageClassifierNet_Classic, 
             criterion: BayesianTripletLoss, epoch: int, print_freq: int,
             temp: float = None, with_swag = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, classes) in enumerate(val_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            # Get model output
            if with_swag == True:
                output = model.forward_head_with_swag(input)
            else:
                output = model.forward_head(input)
            
            # Adjust with temp    
            if temp is not None:
                loss = criterion(output/temp, target.cuda())
            else:
                loss = criterion(output, target.cuda())
            losses.update(loss.item())
            
            # Get accuracy
            _, out_idx = output.max(dim=1)
            accuracy.update(torch.sum(out_idx.cpu() == target).item()/len(classes))
        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print update
            if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(val_loader):
                print(f'>> Validate: [{epoch+1}][{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)\t'
                      f'Accuracy {(accuracy.val)*100:.3f}% ({(accuracy.avg)*100:.3f}%)')

    return losses.avg, accuracy.avg
  

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
                config={"Model type": 'Classic',
                        "default": dict(default_conf), 
                        "hyperparameters": dict(hyper_conf),
                        "model": dict(model_conf),
                        "img_conf": dict(img_conf),
                        "dataset_conf": dict(dataset_conf)},
                job_type="Train"
        )
    
    # Get wandb name
    wandb_run_name = wandb.run.name
    print(wandb_run_name)
    
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
    
    ds_train = Pooling_Dataset(mode=mode_train, 
                            poolsize_class=int(dataset_conf['poolsize_class']),
                            transform=transformer_train,
                            classes_not_trained_on=classes_not_trained_on)
    ds_val = Pooling_Dataset(mode=mode_val, 
                              poolsize_class=int(dataset_conf['poolsize_class']),
                              transform=transformer_valid,
                              classes_not_trained_on=classes_not_trained_on)
    
    
    # ************************************* 
    # ******** Initialize model ***********
    # *************************************
    logger.info(">> Initilizing model")
    params = {'model_type': 'Classic',  
                        'seed':int(default_conf['seed']),        
                        'architecture':ast.literal_eval(model_conf['architecture']),
                        'fixed_backbone': ast.literal_eval(model_conf['fixed_backbone']),
                        'const_eval_mode': ast.literal_eval(model_conf['const_eval_mode']),
                        'layers_dim': ast.literal_eval(model_conf['layers_dim']),
                        'activation_fn': ast.literal_eval(model_conf['activation_fn']),
                        'pooling': ast.literal_eval(model_conf['pooling']),
                        'num_classes': int(num_classes),
                        'dropout': float(model_conf['dropout']),
                        'classes_not_trained_on': classes_not_trained_on,
                        'img_size': int(img_conf['img_size']),
                        'normalize_mv' : ast.literal_eval(img_conf['normalize']),
                        'with_swag': ast.literal_eval(model_conf['with_swag']),
                        'trained_on': mode_train}
    net = init_network('Classic',params)
    net.cuda()
    
    # Save parameters to json file
    with open(f"./models/params/{wandb_run_name}.json", "w") as outfile:
        json.dump(params, outfile)
    wandb.save(f'./models/params/{wandb_run_name}.json', policy="now")
    
    # ***************************************
    # ******** Initialize loaders ***********
    # ***************************************
    # Init tuples
    logger.info(">> Initilizing backbone representations")
    ds_train.update_backbone_repr_pool(net)
    ds_val.update_backbone_repr_pool(net)
    
    train_loader = DataLoader(
            ds_train, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            sampler=None, drop_last=False
        )
    val_loader = DataLoader(
            ds_val, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            sampler=None, drop_last=False
        )
    
    # ********************************
    # ******** Train model ***********
    # ********************************
    
    # init values
    num_epochs = int(hyper_conf['epochs'])
    lr_init = float(hyper_conf['lr_init'])
    lr_end = float(hyper_conf['lr_end'])
    #lr_diff = (lr_end-lr_init)*(1/(num_epochs-num_epochs//4))
    lr_frac = (lr_end/lr_init)**(1/max(1,num_epochs))
    clip = float(hyper_conf['clip'])
    update_every = int(hyper_conf['update_every'])
    update_pool_num = int(hyper_conf['update_pool_every'])
    
    update_pool_count = 0
    
    mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"Total memory consumption on Cuda: {mem_cuda:.3f}GB")
    
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_epochs):
        logger.info(f"\n\n########## Training {i+1}/{num_epochs} ##########")
        # *****************************
        # ******** Training ***********
        # *****************************
        #lr_optim = max(lr_init+(lr_diff*(max(0,i+1-num_epochs//4))),lr_end)
        lr_optim = max(lr_init*lr_frac**i,lr_end)
        base_params = net.features.parameters()
        head_params = [i for i in net.parameters() if i not in base_params]
        optim = AdamW([
                        {'params': head_params},
                        {'params': base_params, 'lr': lr_optim*0.01}, 
                      ],lr=lr_optim, weight_decay=1e-1)
        
        # train model
        train_loss, train_acc = train(train_loader,net,criterion,optim,
                                                          i, update_every, print_freq, 
                                                          clip)
        
        # *******************************
        # ******** Validation ***********
        # *******************************
        val_loss, val_acc = validate(val_loader,net,criterion,i,print_freq)
        
        
        # *******************************
        # ******** Log results **********
        # *******************************
        
        logger.info(f'\n\t***** Epoch: {i+1} *****\n\t\t Train loss: {train_loss*1000:.4f}\t\t'+
                    f'Train acc: {train_acc*100:.4f}%\n\t\t Val loss: {val_loss*1000:.4f}\t\t'+
                    f'Val acc: {val_acc*100:.4f}%')
        
        # Log diagnostics
        wandb.log(
                {
                    "Training_loss_classic": train_loss,
                    "Validation_loss_classic": val_loss,
                    "Training_accuracy": train_acc,
                    "Validation_accuracy": val_acc,
                    "Epoch": i+1,
                    "lr_optim": lr_optim,
                }
            )
        
        # Save model on W&B and remove locally
        if ((i+1)%save_model_freq == 0) | (i == num_epochs-1):
            logger.info('Saving the model')
            torch.save(net.state_dict(),f'./models/state_dicts_classic/{wandb_run_name}.pt')
            wandb.save(f'./models/tmp_models/{wandb_run_name}.pt', policy="now")
            
            
        # *********************************
        # ******** Update Pool ************
        # *********************************
        if (update_pool_count == update_pool_num):
            train_loader.dataset.update_backbone_repr_pool(net)
            
            update_pool_count = 0
        else:
            update_pool_count += 1
            
            
        # ********************************************
        # ******** Save SWAG Init weights ************
        # ********************************************
        if params['with_swag'] == True:
            if i+1 == int(num_epochs*1/4):
                base_params = net.features.parameters()
                head_params = [i for i in net.parameters() if i not in base_params]
                swag_init_head = [i.data.clone() for i in head_params]



    # **********************************************
    # ******** Find Optimal Temperature ************
    # **********************************************
    temp_vals = torch.logspace(-3,3,500)
    CE_out = []
    for temp in temp_vals:
        val_loss, val_acc = validate(val_loader,net,criterion,i,print_freq,temp)
        CE_out.append(val_loss)
        
    temp_opt = temp_vals[np.array(CE_out).argmin()].item()
    params['temp_opt'] = temp_opt
    
    # Save new parameters to json file
    with open(f"./models/params/{wandb_run_name}.json", "w") as outfile:
        json.dump(params, outfile)
    wandb.save(f'./models/params/{wandb_run_name}.json', policy="now")
    
    
    # *******************************
    # ******** With SWAG ************
    # *******************************
    if params['with_swag'] == True:
        head_params_swag = []
        head_params_mean = []
        head_params_var = []
        logger.info(f'>> Extracting param values for SWAG')
        
        #Reset lr
        lr_init_swag = (lr_init+lr_end)/2
        lr_frac_swag = (lr_end/lr_init_swag)**(1/(4-1))
        
        #Reset weights to swag init
        count = 0
        for idx, (name, param) in enumerate(net.named_parameters()):
            if name.split('.')[0] != 'features':
                param.data = swag_init_head[count]
                count += 1
            
        # Extract SWAG weights
        for j in range(10):
            base_params = net.features.parameters()
            head_params = [i for i in net.parameters() if i not in base_params]
            head_params_ex = [i.data.clone() for i in head_params]
            head_params_swag.append(head_params_ex)
            
            optim = AdamW([
                            {'params': head_params},
                            {'params': base_params, 'lr': lr_optim*0.01}, 
                        ],lr=lr_optim, weight_decay=1e-1)
        
            # train model
            for k in range(4):
                lr_optim = lr_init_swag*(lr_frac_swag**k)
                train_loss, train_acc = train(train_loader,net,criterion,optim,
                                                                j, update_every, print_freq, 
                                                                clip)
            
                if (j+1 != 10):
                    train_loader.dataset.update_backbone_repr_pool(net)
                    
                logger.info(f"Within update {k+1}/4")
                    
            logger.info(f'>>>>> {j+1}/{10}')
 
        
        # Get mean of params
        for j in range(len(head_params)):
            mean_params = head_params_swag[0][j].clone()
            for k in range(len(head_params_swag)-1):
                mean_params += head_params_swag[k+1][j]
            mean_params /= len(head_params_swag)
            head_params_mean.append(mean_params)
            
        # Get var of params
        for j in range(len(head_params)):
            var_params = ((head_params_swag[0][j]-head_params_mean[j])**2).clone()
            for k in range(len(head_params_swag)-1):
                var_params += (head_params_swag[k+1][j]-head_params_mean[j])**2
            var_params /= len(head_params_swag)
            head_params_var.append(var_params**(1/2))
            
        net.head_mean = head_params_mean
        net.head_std = head_params_var
        
        logger.info('Saving SWAG headers')
        torch.save(net.head_mean,f'./models/swag_headers/{wandb_run_name}_mean_swag.pt')
        torch.save(net.head_std,f'./models/swag_headers/{wandb_run_name}_var_swag.pt')
        
        
        
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
