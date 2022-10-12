# Standard libraries
from sqlite3 import paramstyle
import wandb
import json
import time
from torchvision import transforms
import torch
import pdb
import ast
import os
import configparser
import argparse
import shutil
import glob

# Own libraries
from src.loaders.tuple_loader import TuplesDataset
from src.loaders.tuple_loader2 import TuplesDataset_2class
from src.models.image_classification_network import ImageClassifierNet, init_network
from src.loss_functions.bayesian_triplet_loss import BayesianTripletLoss
from src.loaders.tuple_loader import TuplesDataset
from src.utils.helper_functions import (AverageMeter, 
                                        print_PCA_tSNE_plot, 
                                        collate_tuples,
                                        check_config,
                                        get_logger)
        
def train(train_loader, model, criterion, optimizer, epoch, update_every, print_freq, clip):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.zeros(model.meta['outputdim'], ni).cuda()
            for imi in range(ni):

                # compute output vector for image imi
                input[q][imi] = input[q][imi][None,:,:,:] #Expand dim
                output[:, imi] = model(input[q][imi].cuda()).squeeze()

            # accumulate gradients for one tuple at a time
            loss = criterion(output, target[q].cuda())
            #loss /= nq # get mean across batch
            losses.update(loss.item())
            loss.backward()

        if ((i + 1) % update_every == 0) | (i+1 == len(train_loader)):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
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
                  f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)')

    return losses.avg



def validate(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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
                    output[:, imi] = model(input[q][imi].cuda()).squeeze()

                # accumulate gradients for one tuple at a time
                loss = criterion(output, target[q].cuda())
                #loss /= nq # get mean across batch
                losses.update(loss.item())
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print update
            if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(val_loader):
                print(f'>> Validate: [{epoch+1}][{i+1}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {(losses.val)*1000:.4f}e-3 ({(losses.avg)*1000:.4f}e-3)')

    return losses.avg
            


def main(config):
    # **********************************
    # ******** Initalize run ***********
    # **********************************
    
    # Check if configs has been run before
    config_filename, config_run_no = check_config(config)
    
    # Get configs
    default_conf = config['default']
    print_freq = int(default_conf['print_freq'])
    save_model_freq = int(default_conf['save_model_freq'])
    hyper_conf = config['hyperparameters']
    model_conf = config['model']
    img_conf = config['image_transform']
    dataset_conf = config['dataset']
    
    # get logger
    logger = get_logger(config_filename, config_run_no)
    logger.info("STARTING NEW RUN")
    
    # Init wandb
    wandb.init(
                project="master_thesis",
                entity="nweis97",
                config={"default": dict(default_conf), 
                        "hyperparameters": dict(hyper_conf),
                        "model": dict(model_conf),
                        "img_conf": dict(img_conf),
                        "dataset_conf": dict(dataset_conf),
                        "config_filename": config_filename,
                        "config_run_no": str(config_run_no)},
                job_type="Train"
        )

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
                                                transforms.RandomRotation((-10,10)),
                                                transforms.PILToTensor(),
                                                transforms.ConvertImageDtype(torch.float),
                                                transforms.Normalize(torch.tensor(norm_mean),
                                                                    torch.tensor(norm_var))
        ])
    else:
        transformer_train = transformer_valid
    
    
    # Make datasets
    logger.info(">> Initilizing datasets")
    ds_train = TuplesDataset_2class(mode='train', 
                                    nnum=int(dataset_conf['nnum']), 
                                    qsize_class=int(dataset_conf['qsize_class']),
                                    poolsize=int(dataset_conf['poolsize']),
                                    transform=transformer_train,
                                    keep_prev_tuples=ast.literal_eval(dataset_conf['keep_prev_tuples']),
                                    num_classes=int(dataset_conf['num_classes']))
    ds_val = TuplesDataset_2class(mode='val', 
                                  nnum=int(dataset_conf['nnum']), 
                                  qsize_class=int(dataset_conf['qsize_class']),
                                  poolsize=int(dataset_conf['poolsize']),
                                  transform=transformer_valid,
                                  keep_prev_tuples=False,
                                  num_classes=int(dataset_conf['num_classes']))
    
    
    # ************************************* 
    # ******** Initialize model ***********
    # *************************************
    
    logger.info(">> Initilizing model")
    params = {'architecture':ast.literal_eval(model_conf['architecture']),
                        'fixed_backbone': ast.literal_eval(model_conf['fixed_backbone']),
                        'const_eval_mode': ast.literal_eval(model_conf['const_eval_mode']),
                        'head_layers_dim': ast.literal_eval(model_conf['head_layers_dim']),
                        'activation_fn': ast.literal_eval(model_conf['activation_fn']),
                        'pooling': ast.literal_eval(model_conf['pooling']),
                        'dim_out': int(model_conf['dim_out']),
                        'dropout': float(model_conf['dropout']),
                        'num_classes': int(dataset_conf['num_classes']),
                        'img_size': int(img_conf['img_size']),
                        'normalize_mv' : ast.literal_eval(img_conf['normalize'])}
    net = init_network(params)
    net.cuda()
    
    # Save parameters to json file
    os.makedirs(f'./models/{config_filename}',exist_ok=True)  
    with open(f"./models/{config_filename}/run_{config_run_no}.json", "w") as outfile:
        json.dump(params, outfile)
    wandb.save(f'./models/{config_filename}/run_{config_run_no}.json', policy="now")
    
    # ***************************************
    # ******** Initialize loaders ***********
    # ***************************************
    # Init tuples
    logger.info(">> Initilizing tuples")
    ds_train.create_epoch_tuples(net)
    ds_val.create_epoch_tuples(net)
    
    train_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            num_workers=8, pin_memory=True, sampler=None,
            drop_last=False, collate_fn=collate_tuples
        )
    val_loader = torch.utils.data.DataLoader(
            ds_val, batch_size=int(hyper_conf['batch_size']), shuffle=True,
            num_workers=8, pin_memory=True, sampler=None,
            drop_last=False, collate_fn=collate_tuples
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
    margin_fixed = ast.literal_eval(hyper_conf['margin_fixed'])
    margin_val = float(hyper_conf['margin_val'])
    var_prior = float(hyper_conf['var_prior'])
    clip = float(hyper_conf['clip'])
    update_every = int(hyper_conf['update_every'])
    valid_margin_val = margin_val
    valid_kl_scale = kl_scale_end
    
    # train and val losses
    train_losses = []
    val_losses = []
    avg_neg_dist_train = []
    avg_neg_dist_val = []
    
    for i in range(num_epochs):
        logger.info(f"\n\n########## Training {i+1}/{num_epochs} ##########")
        
        # *****************************
        # ******** Training ***********
        # *****************************
        # create tuples for training
        avg_neg_distance,qvecs,qvars,classes = train_loader.dataset.create_epoch_tuples(net)
        avg_neg_dist_train.append(avg_neg_distance)

        # print embedding space
        fig_tsne_train, fig_pca_train = print_PCA_tSNE_plot(qvecs.to('cpu').numpy(),
                                                            qvars.to('cpu').numpy(),
                                                            classes,i,var_prior,'train')
        
        # update loss and optimizer
        if margin_fixed:
            margin = margin_val
            kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor)
        else:
            margin = avg_neg_distance*margin_val
            kl_scale_factor = kl_scale_init*kl_frac**min([i,kl_warmup-1])
            criterion = BayesianTripletLoss(margin=margin,
                                            varPrior=var_prior,
                                            kl_scale_factor=kl_scale_factor)
        lr_optim = lr_init+(lr_diff*(max(0,i+1-kl_warmup)))
        optim = torch.optim.AdamW(net.parameters(),lr=lr_optim,
                                  weight_decay=1e-1,)

        logger.info(f'Updating margin ({margin:.2e}) and kl_scale_factor ({kl_scale_factor:.2e})')
        
        # train model
        train_loss = train(train_loader,net,criterion,optim,i,
                                  update_every, print_freq, clip)
        train_losses.append(train_loss)
        
        
        # *******************************
        # ******** Validation ***********
        # *******************************
        # create tuples for validation
        avg_neg_distance,qvecs,qvars,classes = val_loader.dataset.create_epoch_tuples(net)
        avg_neg_dist_val.append(avg_neg_distance)
        
        # print embedding space
        fig_tsne_val, fig_pca_val = print_PCA_tSNE_plot(qvecs.to('cpu').numpy(),
                                                        qvars.to('cpu').numpy(),
                                                        classes,i+1,var_prior,'train')
        # validate model
        criterion = BayesianTripletLoss(margin=valid_margin_val,
                                        varPrior=var_prior,
                                        kl_scale_factor=valid_kl_scale)
        val_loss = validate(val_loader,net,criterion,i,print_freq)
        val_losses.append(val_loss)
        
        
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
                    "tSNE_train": wandb.Image(fig_tsne_train),
                    "PCA_train": wandb.Image(fig_pca_train),
                    "tSNE_val": wandb.Image(fig_tsne_val),
                    "PCA_val": wandb.Image(fig_pca_val),
                    "kl_scale_factor": kl_scale_factor,
                    "lr_optim": lr_optim
                }
            )
        
        # Save model on W&B and remove locally
        if (i+1)%save_model_freq == 0:
            logger.info('Saving the model')
            torch.save(net.state_dict(),f'./models/{config_filename}/run_{config_run_no}.pt')
            wandb.save(f'./models/{config_filename}/run_{config_run_no}.pt', policy="now")
            

        
    wandb.save(f'./logs/training_test/{config_filename}/run_{config_run_no}.log') 
    
    # Remove local files
    wandb_local_dir = wandb.run.dir[:-5]
    wandb.finish()
    os.remove(f'./logs/training_test/{config_filename}/run_{config_run_no}.log')
    os.remove(f"./models/{config_filename}/run_{config_run_no}.json")
    os.remove(f'./models/{config_filename}/run_{config_run_no}.pt')
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
