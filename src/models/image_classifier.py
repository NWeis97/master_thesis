from argparse import ArgumentTypeError
import torch
import logging
import json
import os
import numpy as np
import pandas as pd
from numpy.random import noncentral_chisquare
from torchvision import transforms
from src.models.image_classification_network import init_network
import ast
import pdb
from os.path import exists
from torch_scatter import scatter_mean

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader
from src.utils.helper_functions import get_logger_test

#logger = logging.getLogger('__main__')
logger = get_logger_test('test')

class ImageClassifier():
    
    def __init__(self, model: str, 
                 model_data: str, 
                 balanced_classes: int = 0,
                 with_OOD: bool= False):
        """Image classifier model. Has embeddings of training data and can classify and calculate
           probability disitrbution for unknown object over objects trained on.

        Args:
            model (str): model name (path to params and state_dict should be in ./models/params and 
                         ./models/state_dict respectively, with names {model}.pt and {model}.json)
            model_data (str): TRAIN or TRAINVAL (use either training data or training and val data)
            balanced_classes (str): Determines whether or not upsampling (with data augmentation)
                                    and downsampling should be used in order to get 
                                    'balanced_classes' number of objects for each class.
                                    Defaults to 0 (no balancing of data).
        """
        super(ImageClassifier, self).__init__()
        self.model_name = model
        self.model_data = model_data
        self.balanced_classes = balanced_classes
        self.with_OOD = with_OOD
        
        if with_OOD == True:
            logger.info('Creating classifier *with* OOD classes')
        else:
            logger.info('Creating classifier *without* OOD classes')
        # *****************************************
        # ******** Initialize model data **********
        # *****************************************
        # Set paths to data
        self.data_root = './data/processed/'
        self.ims_root = './data/raw/JPEGImages/'
        self.model_data = model_data
        db = self._create_db_for_model_data_()
        
        
        # ****************************************
        # ******** Load model and data ***********
        # ****************************************
        # Load model
        f = open(f'./models/params/{model}.json',)
        self.params = json.load(f)
        
        self.model = init_network(self.params)
        self.model.load_state_dict(torch.load(f'models/state_dicts/{model}.pt'))
        
        # Select subset of classes (depends on the classes the model was trained on)
        self.num_classes = self.params['num_classes']
        self.all_class_list = ['train','cow','tvmonitor','boat','cat','person','aeroplane','bird',
                               'dog','sheep','bicycle','bus','motorbike','bottle','chair',
                               'diningtable','pottedplant','sofa','horse','car']
        
        # Remove we would like to no include OOD classes
        if self.with_OOD == False:
            db = {self.all_class_list[i]:db[self.all_class_list[i]] for i in range(self.num_classes)}
            
        # Extract list of images
        obj_names = [[db[key][j]['img_name'] for j in range(len(db[key]))] for key in db.keys()]
        self.objects = []
        [self.objects.extend(image_list) for image_list in obj_names]

        # Extract list of bbox
        bbox_list = [[db[key][j]['bbox'] for j in range(len(db[key]))] for key in db.keys()]
        self.bbox = []
        [self.bbox.extend(bbox) for bbox in bbox_list]
    
        # Make image transformer
        img_size = int(self.params['img_size'])
        normalize_mv = self.params['normalize_mv']
        norm_mean = normalize_mv['mean']
        norm_var = normalize_mv['var']
        self.transformer = transforms.Compose([
                                        transforms.Resize((img_size,img_size)),
                                        transforms.PILToTensor(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize(torch.tensor(norm_mean),
                                                            torch.tensor(norm_var))
        ])
        self.transformer_aug = transforms.Compose([
                                                transforms.Resize((img_size+20,img_size+20)),
                                                transforms.RandomCrop(img_size),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomRotation((-10,10)),
                                                transforms.PILToTensor(),
                                                transforms.ConvertImageDtype(torch.float),
                                                transforms.Normalize(torch.tensor(norm_mean),
                                                                    torch.tensor(norm_var))
        ])
    
        # ************************************************************************
        # ******** Extract means, variance and classes for model data  ***********
        # ************************************************************************
        # Extract list of classes
        class_num = [[key]*len(db[key]) for key in db.keys()]
        self.classes = []
        [self.classes.extend(class_list) for class_list in class_num]
        
        # Mark objects not trained on if with_OOD is true
        if self.with_OOD == True:
            classes_not_trained_on = self.get_classes_not_trained_on()
            for i in range(len(self.classes)):
                if self.classes[i] in classes_not_trained_on:
                    self.classes[i] = self.classes[i] + '_OOD'
        
        # Find means and variances of model data
        self.model.cuda()
        self.model.eval()

        file_name = (f'./models/embeddings/{model}_{model_data}_'+
                                        f'{str(balanced_classes)}_'+
                                        f'withOOD{self.with_OOD}_')
        if exists(file_name+'means.pt') == False:
            self.means, self.vars, self.classes = self._extract_means_and_variances_()
            torch.save(self.means,file_name+'means.pt')
            torch.save(self.vars,file_name+'vars.pt')
            torch.save(self.classes,file_name+'classes.pt')
        else:
            logger.info('Loading already existing classifier model')
            self.means = torch.load(file_name+'means.pt')
            self.vars = torch.load(file_name+'vars.pt')
            self.classes = torch.load(file_name+'classes.pt')
        
        # Split classes into list with indices of positions
        if self.with_OOD == True:
            self.unique_classes = np.sort(np.unique(self.classes))
        else:
            self.unique_classes = self.get_classes_trained_on()
            
        self.classes_idxs = {self.unique_classes[i]:[] for i in range(len(self.unique_classes))}
        for i, class_ in enumerate(self.classes):
            self.classes_idxs[class_].append(i)
        self.num_samples_classes = ({class_: len(self.classes_idxs[class_]) for class_ in 
                                               self.classes_idxs.keys()})

    def __repr__(self):
        tmpstr = super(ImageClassifier, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr


    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'
        tmpstr += '     model: {}\n'.format(self.model)
        tmpstr += '     model_data: {}\n'.format(self.model_data)
        tmpstr += '     num_classes: {}\n'.format(self.num_classes)
        tmpstr += '     transformer: {}\n'.format(self.transformer)
        tmpstr = tmpstr + '  )\n'
        return tmpstr


    def _create_db_for_model_data_(self):
        if self.model_data == 'TRAIN':
            db_root = os.path.join(self.data_root, 'train.json')
            # Load database
            f = open(db_root)
            db = json.load(f)
            f.close()
            
        elif self.model_data == 'TRAINVAL':
            db_root_train = os.path.join(self.data_root, 'train.json')
            db_root_val = os.path.join(self.data_root, 'val.json')
            
            # Load database
            f = open(db_root_train)
            db = json.load(f)
            f.close()
            f = open(db_root_val)
            db_val = json.load(f)
            f.close()
            
            # Concat databases
            for k in db.keys():
                db[k].extend(db_val[k])
        
        else:
            logger.error('Unknown model_data input (use either TRAIN or TRAINVAL)')
            exit;

        return db
    
    
    def _extract_means_and_variances_(self): 
        OFL = ObjectsFromList(root=self.ims_root, 
                              obj_names=[obj for obj in self.objects], 
                              obj_bbox=[bbox for bbox in self.bbox],
                              transform=self.transformer)
        OFL_aug = ObjectsFromList(root=self.ims_root, 
                              obj_names=[obj for obj in self.objects], 
                              obj_bbox=[bbox for bbox in self.bbox],
                              transform=self.transformer_aug)    
        loader = torch.utils.data.DataLoader(
                                OFL,
                                batch_size=1, shuffle=False, num_workers=8, pin_memory=False
            )

        logger.info("Extract means and variance for model data")

        # extract means and variances 
        with torch.no_grad():
            if self.balanced_classes < 1:
                logger.info("No balancing of classes")
                qvecs = torch.zeros(self.params['dim_out']-1, len(self.objects)).cuda()
                qvars = torch.zeros(len(self.objects),).cuda()
            
                for i, input in enumerate(loader):
                    output = self.model(input.cuda()).data.squeeze()
                    qvecs[:, i] = output[:-1]
                    qvars[i] = output[-1]
                    if (i+1) % 1000 == 0 or (i+1) == len(self.objects):
                        logger.info('>>>> {}/{} done... '.format(i+1, len(self.objects)))
            
                classes = self.classes
            
            else:
                logger.info(f"Balancing of classes: Total number of samples from each class will be "+
                            f"{self.balanced_classes}")
                dist_of_classes = pd.value_counts(self.classes)
                num_samples = len(dist_of_classes)*self.balanced_classes
                qvecs = torch.zeros(self.params['dim_out']-1, num_samples).cuda()
                qvars = torch.zeros(num_samples,).cuda()
                
                # for track-keeping
                classes_count = {dist_of_classes.index[i]:0 for i in range(len(dist_of_classes))}
                classes = []
                count = 0
                
                while True:
                    r_ord = torch.randperm(len(self.objects))
                    for i in range(len(self.objects)):
                        sample_class = self.classes[r_ord[i]]
                        if classes_count[sample_class] < self.balanced_classes: # add object
                            
                            if dist_of_classes[sample_class] <= self.balanced_classes: # no aug
                                input = OFL.__getitem__(r_ord[i])
                            else:
                                input = OFL_aug.__getitem__(r_ord[i])
                            
                            # save output
                            input = input[None,:,:,:]
                            output = self.model(input.cuda()).data.squeeze()
                            qvecs[:, count] = output[:-1]
                            qvars[count] = output[-1]
                            
                            # add counts and class
                            classes.append(sample_class)
                            classes_count[sample_class] += 1
                            count += 1
                            if (count) % 1000 == 0 or (count) == num_samples:
                                logger.info('>>>> {}/{} done... '.format(count, num_samples))
                                
                    if count == num_samples:
                        break;
            
        
        return qvecs, qvars, classes

    def get_classes_trained_on(self):
        return np.sort(np.array(self.all_class_list)[:self.num_classes]).tolist()
    
    def get_classes_not_trained_on(self):
        trained_on = self.get_classes_trained_on()
        return [class_ for class_ in self.all_class_list if class_ not in trained_on]

    def get_embedding(self, img_name: str, bbox: list):
        """Get model embeddings of image

        Args:
            img_path (str): image name
            bbox (list): bbox of object

        Returns:
            mean_emb, var_emb: mean and variance embedding of image
        """
        img_path = self.ims_root + f'/{img_name}'
        img = image_object_loader(img_path,bbox,self.transformer)
        img = img[None,:,:,:] #Expand dim
        with torch.no_grad():
            output = self.model(img.cuda()).data.squeeze()
            mean_emb = output[:-1]
            var_emb = output[-1]
        
        return mean_emb, var_emb
    

    def get_probability_dist(self, img_name: str, 
                                   bbox: list, 
                                   num_NN: int = 100, 
                                   num_MC: int = 10000,
                                   method: str = 'min_dist_NN'):
        """Get probability distribution over model classes for an input image (defined by path
           and bounding box). The probability distribution will be based on num_NN nearest neighbors
           in the embedding space. For each nearest neighbour the probability that it is the
           closest object in the embedding space will be calculated (based on num_MC samples), and
           then accumulated on class level. 
           Note that the euclidian distance between gaussian disitrbuted r.v. with mean different
           from 0 and variance different from 1 follows a scaled non-centered chi-sq distribution.

        Args:
            img_path (str): image name
            bbox (list): bbox of object on image
            num_NN (int, optional): Number of nearest objects to base probs on. Defaults to 100.
            num_MC (int, optional): Number of Monte Carlo samples. Defaults to 10000.
            method (str. optional): See description under function for get_probability_dist_dataset
        """
        # Get embeddings
        mean_emb, var_emb = self.get_embedding(img_name,bbox)
        
        dist = (torch.pow(self.means-mean_emb[None,:].T+1e-6, 2).sum(dim=0).sqrt())
        _, ranks = torch.sort(dist, dim=0, descending=False)
        if method == 'min_dist_NN':    
            probs = self._min_dist_NN_(ranks, mean_emb, var_emb, num_NN, num_MC)
        elif method == 'avg_dist_rand':
            probs = self._avg_dist_rand_(mean_emb, var_emb, num_NN, num_MC)
        elif method == 'avg_dist_NN_pr_class':
            probs = self._avg_dist_NN_pr_class_(ranks, mean_emb, var_emb, num_NN, num_MC)
        elif method == 'min_dist_NN_pr_class':
            probs = self._min_dist_NN_pr_class_(ranks, mean_emb, var_emb, num_NN, num_MC)
        elif method =='avg_dist_NN_cap_class_NN':    
            probs = self._avg_dist_NN_cap_class_NN_(ranks, mean_emb, var_emb, num_NN, num_MC)
        elif method =='min_dist_NN_cap_class_NN':    
            probs = self._min_dist_NN_cap_class_NN_(ranks, mean_emb, var_emb, num_NN, num_MC)
        elif method =='min_dist_NN_with_min_dist_rand':
            probs = self._min_dist_NN_with_min_dist_rand_(ranks, mean_emb, var_emb, num_NN, num_MC)
        else:
            raise ValueError('Method does not exist')

        return probs, mean_emb, var_emb
        
        
    def get_probability_dist_dataset(self, test_dataset: str = 'test',
                                           num_NN: int = 100, 
                                           num_MC: int = 10000,
                                           method: str = 'min_dist_NN'):
        """Calculates the probability distributiuon over classes trained on for a set of images
           defined by 'test_dataset'. Name of 'test_dataset' should be present in data/processed/.

        Args:
            test_dataset (str, optional): Name of test dataset (json file). Defaults to 'test'.
            num_NN (int, optional): Number of nearest objects to base probs on. Defaults to 100.
            num_MC (int, optional): Number of Monte Carlo samples. Defaults to 10000.
            method (str, optional): Determines how the probabilities are calculated. Choose between
                                    min_dist_NN: Based on num_NN nearest neighbours, sample num_MC
                                                 distances from anchor to NN's and note the class
                                                 with smallest distance. Accumulate on class level.
                                                 Propertion of class noted approximates probability
                                    min_dist_rand: For each num_MC select at random one object from
                                                   each class, sample distance to anchor and note
                                                   the class with smallest distance. Accumulate on 
                                                   class level. Propertion of class noted 
                                                   approximates probability 
                                    score_based: Based on num_NN nearest neighbours from each class, 
                                                 (calculated as mean distance to anchor), sort 
                                                 objects from closest to furthest away. Score each 
                                                 object based on position. 

        Returns:
            probs_df (pd.DataFrame): Pandas dataframe containing the probability dist over classes
                                     for all samples in the test dataset
            classes (list):          List of true classes 
        """
        # Extract test dataset
        objects, bboxs, classes = self._extract_test_dataset_(test_dataset)
        
        # For storing probabilities
        file_name = (f'{self.model_name}_{self.model_data}_'+
                        f'{self.balanced_classes}_numNN{num_NN}_'+
                        f'numMC{num_MC}_{method}_{test_dataset}_'+
                        f'withOOD{self.with_OOD}')
        file_name_probs = f'./reports/probability_distributions/' + file_name + '.csv'
        file_name_embs = f'./reports/embeddings_test/' + file_name
        if exists(file_name_probs) == False:
            logger.info('Extracting probability distributions')
            probs_df = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                    columns = [i for i in range(len(classes))],
                                    index = self.unique_classes)
            
            # For storing mean_emb and var_emb
            mean_embs = torch.zeros(self.params['dim_out']-1, len(classes)).cuda()
            var_embs = torch.zeros(len(classes),).cuda()
            for i in range(len(classes)):
                probs, mean_emb, var_emb = self.get_probability_dist(objects[i],
                                                                     bboxs[i],
                                                                     num_NN,
                                                                     num_MC,
                                                                     method)
                probs_df[i] = np.round(np.array(list(probs.values())),int(np.log10(num_MC)+1))
                mean_embs[:,i] = mean_emb
                var_embs[i] = var_emb
                
                if (i+1) % 100 == 0:
                    logger.info('>>>> {}/{} done... '.format(i+1, len(classes)))
                    
            # Save dataframe to reports (for later use)
            probs_df.to_csv(file_name_probs)
            torch.save(mean_embs, file_name_embs + '_means.pt')
            torch.save(var_embs, file_name_embs + '_vars.pt')
        else:
            logger.info('Loading existing probability distribution')
            probs_df = pd.read_csv(file_name_probs,index_col=0)
        
        return probs_df, objects, bboxs, classes
    
    
    
    def get_embeddings_of_dataset(self, test_dataset: str = 'test',
                                           num_NN: int = 100, 
                                           num_MC: int = 10000,
                                           method: str = 'min_dist_NN'):
        file_name = (f'{self.model_name}_{self.model_data}_'+
                        f'{self.balanced_classes}_numNN{num_NN}_'+
                        f'numMC{num_MC}_{method}_{test_dataset}_'+
                        f'withOOD{self.with_OOD}')
        file_name_embs = f'./reports/embeddings_test/' + file_name
        if exists(file_name_embs + '_means.pt'):
            means = torch.load(file_name_embs + '_means.pt')
            vars = torch.load(file_name_embs + '_vars.pt')
        else:
            raise RuntimeError('You have to call "get_probability_dist_dataset" before you can get'+
                               ' embeddings!')
    
        return means, vars
    

    
    def _extract_test_dataset_(self, test_dataset: str = 'test'):
        # Extract test dataset
        db_root = os.path.join(self.data_root, f'{test_dataset}.json')
        f = open(db_root)
        db_test = json.load(f)
        f.close()
        
        if test_dataset == 'test':
            # Extract list of images
            objects_list = [[db_test[j]['img_name']]*len(db_test[j]['classes']) 
                                                                    for j in range(len(db_test))]
            objects = []
            [objects.extend(obj) for obj in objects_list]
            
            # Extract list of bbox
            bbox_list = [db_test[j]['bbox'] for j in range(len(db_test))]
            bboxs = []
            [bboxs.extend(bbox) for bbox in bbox_list]
            
            # Extract list of classes
            class_num = [db_test[j]['classes'] for j in range(len(db_test))]
            classes = []
            [classes.extend(class_list) for class_list in class_num]
        else:
            # Extract list of images
            obj_names = [[db_test[key][j]['img_name'] for j in range(len(db_test[key]))] for key in db_test.keys()]
            objects = []
            [objects.extend(image_list) for image_list in obj_names]

            # Extract list of bbox
            bbox_list = [[db_test[key][j]['bbox'] for j in range(len(db_test[key]))] for key in db_test.keys()]
            bboxs = []
            [bboxs.extend(bbox) for bbox in bbox_list]
            
            # Extract list of classes
            class_num = [[key]*len(db_test[key]) for key in db_test.keys()]
            classes = []
            [classes.extend(class_list) for class_list in class_num]
            
        # Mark objects not trained on if with_OOD is true else remove them
        classes_not_trained_on = self.get_classes_not_trained_on()
        if self.with_OOD == True:
            for i in range(len(classes)):
                if classes[i] in classes_not_trained_on:
                    classes[i] = classes[i] + '_OOD'
        else:
            idx_trained_on = [i for i in range(len(classes)) if classes[i] 
                                                                    not in classes_not_trained_on]
            objects = [objects[i] for i in idx_trained_on]
            bboxs = [bboxs[i] for i in idx_trained_on]
            classes = [classes[i] for i in idx_trained_on]
            
        return objects, bboxs, classes
    
    
    def calc_aP_for_class(self, probs_df: pd.DataFrame, 
                                true_classes: list, 
                                class_: str):
        """Calculates precision and recall for a class given the predicted classes

        Args:
            probs_df (pd.DataFrame): probability distributions for all samples
            true_classes (list): true classes
            class_ (str): class to calculate precision for

        Returns:
            aP, precision [list], recall [list] : average precision, precision and recall for class
        """
        true_classes = np.array(true_classes)==class_
        # get list of probs for all samples where true_class is class_ 
        probs_class = probs_df.loc[class_,:]
        probs_class_sorted = np.sort(probs_class.unique())-1e-7
        
        precision = []
        recall = []
        for i in range(len(probs_class_sorted)):
            pred_classes = probs_class >= probs_class_sorted[-(i+1)]
            TP = np.sum((pred_classes == True) & (true_classes == True))
            FP = np.sum((pred_classes == True) & (true_classes == False))
            FN = np.sum((pred_classes == False) & (true_classes == True))
        
            precision.append(TP/(TP+FP))
            recall.append(TP/(TP+FN))
        
        precision.insert(0,precision[0])
        recall.insert(0,0)
        
        prec_AUC = [np.max(precision)]
        rec_AUC = [recall[0]]
        prev_rec = recall[0]
        for i in range(len(precision)-1):
            if (precision[i+1] > np.max(precision[i+2:]+[-1])) & (prev_rec < recall[i+1]):
                prec_AUC.append(precision[i+1])
                rec_AUC.append(recall[i+1])
                prev_rec = recall[i+1]
                
        aP = self._calc_AUC_(prec_AUC, rec_AUC)

        return aP, precision, recall, prec_AUC, rec_AUC


    def _calc_AUC_(self, precision: list, recall: list):
        AUC = 0
        for i in range(len(recall)-1):
            AUC += (precision[i+1])*(recall[i+1]-recall[i])
            
        return AUC


    def calc_MaP(self, probs_df: pd.DataFrame, 
                       true_classes: list):
        """Calculates both the mean average precision of the model, and the class specific 
           average precisions (area under precision/recall curve)

        Args:
            probs_df (pd.DataFrame): probability distributions for all samples
            true_classes (list): true classes

        Returns:
            MaP, acc_df: The mean average precision (MaP), and class specific precisions (aP_df)
        """
        true_classes_unique = pd.unique(true_classes)
        aP_df = pd.Series(index=true_classes_unique)
        aP_plotting = {'precision':{}, 'recall':{}, 'prec_AUC':{}, 'rec_AUC':{}}
        for class_ in true_classes_unique:
            aP_df[class_],prec,rec,prec_auc,rec_auc = self.calc_aP_for_class(probs_df,
                                                                             true_classes,
                                                                             class_)
            aP_plotting['precision'][class_] = prec
            aP_plotting['recall'][class_] = rec
            aP_plotting['prec_AUC'][class_] = prec_auc
            aP_plotting['rec_AUC'][class_] = rec_auc
        
        MaP = aP_df.mean()    
        return MaP, aP_df, aP_plotting
        
        
    def calc_accuracy(self, probs_df: pd.DataFrame, 
                            true_classes: list):
        """Calculates both the overall accuracy of the model (picking the class with highest
           probability), and the class specific accuracies. 

        Args:
            probs_df (pd.DataFrame): probability distributions for all samples
            true_classes (list): list of true classes

        Returns:
            acc, acc_mean_class, acc_df: The accuracy of the model (acc), the mean across class
                                         accuracies (acc_mean_class), 
                                         and class specific accuracies (acc_df)
        """
        true_classes_unique = pd.unique(true_classes)
        acc_df_top1 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
        acc_df_top2 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
        acc_df_top3 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
        acc_df_top5 = pd.Series(np.zeros((len(true_classes_unique),)),index=true_classes_unique)
        pred_classes = []
        [pred_classes.append(list(probs_df.iloc[:,i].sort_values()[::-1].index[:5])) for i in 
                             range(len(probs_df.columns))]
 
        for i, classes in enumerate(pred_classes):
            t_class = true_classes[i]
            acc_df_top1[t_class] += t_class in classes[:1]
            acc_df_top2[t_class] += t_class in classes[:2]
            acc_df_top3[t_class] += t_class in classes[:3]
            acc_df_top5[t_class] += t_class in classes[:5]
            
        # Get overall metrics
        n = len(pred_classes)
        acc_top1 = acc_df_top1.sum()/n
        acc_top2 = acc_df_top2.sum()/n
        acc_top3 = acc_df_top3.sum()/n
        acc_top5 = acc_df_top5.sum()/n
        
        # Get class specific metrics    
        value_c = pd.Series(true_classes).value_counts()
        acc_df_top1 = acc_df_top1/value_c
        acc_df_top2 = acc_df_top2/value_c
        acc_df_top3 = acc_df_top3/value_c
        acc_df_top5 = acc_df_top5/value_c
        
        acc_df = pd.DataFrame({'acc_top1': acc_df_top1,
                               'acc_top2': acc_df_top2,
                               'acc_top3': acc_df_top3,
                               'acc_top5': acc_df_top5})
        
        return acc_top1, acc_top2, acc_top3, acc_top5, acc_df
    
    
    def calc_ECE(self, probs_df: pd.DataFrame, 
                       true_classes: list):
        true_classes = np.array(true_classes)
        true_classes_unique = pd.unique(true_classes).tolist()
        true_classes_unique.append('All')
        bins = np.array([0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1])
        bins_mid = [(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]
        
        # Init dataframes for storing accuracies of bins and number of elements in each bin
        cali_plot_df_acc  = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
        cali_plot_df_conf = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
        num_each_bin_df = pd.DataFrame(index = bins_mid, columns = true_classes_unique)
        
        cali_plot_df_acc['All']  = np.zeros((len(bins_mid),))
        cali_plot_df_conf['All'] = np.zeros((len(bins_mid),))
        num_each_bin_df['All']   = np.zeros((len(bins_mid),))
        
        # reset true classes
        true_classes_unique = pd.unique(true_classes).tolist()
        
        # for numeric stability
        bins[0] += -1.e-1 
        bins[-1] += 1.e-1
        
        for class_ in true_classes_unique:
            class_probs = probs_df.loc[class_,:]
            for i in range(len(bins_mid)):
                class_probs_in_bin = (class_probs <= bins[i+1]) & (class_probs > bins[i])
                class_probs_in_bin_true = true_classes[class_probs_in_bin]==class_
                if class_probs_in_bin_true.size != 0:
                    cali_plot_df_acc.loc[bins_mid[i],class_] = np.mean(class_probs_in_bin_true)
                    cali_plot_df_acc.loc[bins_mid[i],'All'] += np.sum(class_probs_in_bin_true)
                    cali_plot_df_conf.loc[bins_mid[i],class_] = np.mean(class_probs_in_bin_true)
                    cali_plot_df_conf.loc[bins_mid[i],'All'] += np.sum(class_probs_in_bin_true)
                    num_each_bin_df.loc[bins_mid[i],class_] = np.sum(class_probs_in_bin)
                    num_each_bin_df.loc[bins_mid[i],'All'] += np.sum(class_probs_in_bin)

        # Calc accuracy for all images
        cali_plot_df['All'] = cali_plot_df['All']/num_each_bin_df['All']
        
        # Calculate non-weighted mean across accuracies
        cali_plot_df['Class_mean'] = cali_plot_df.iloc[:,:-1].mean(axis=1)
        num_each_bin_df['Class_mean'] = num_each_bin_df['All']
        
        # Calaculate ECE across weighted and non-weighted accuracies
        # SCE:
        # https://medium.com/codex/metrics-to-measuring-calibration-in-deep-learning-36b0b11fe816
        pdb.set_trace()
        WECE = self._calc_ECE_(cali_plot_df['All'], num_each_bin_df['All'])
        AECE = self._calc_ECE_(cali_plot_df['Class_mean'], num_each_bin_df['All'])
        CECE, CECE_df = self._calc_CECE_(cali_plot_df.iloc[:,:-2],num_each_bin_df.iloc[:,:-2])
        
        return (cali_plot_df, WECE, AECE, CECE, CECE_df, num_each_bin_df)
    
    
    def _calc_ECE_(self, accuracies: pd.Series,
                         num_samples_bins: pd.Series):
        n = num_samples_bins.sum()
        ECE = (num_samples_bins/n*np.abs(accuracies-accuracies.index)).sum()
        
        return ECE

 
    def _calc_CECE_(self, accuracies: pd.DataFrame,
                          num_samples_bins: pd.DataFrame):
        CECE_df = pd.Series(index=accuracies.columns)
        for i in range(len(accuracies.columns)):
            
            class_ = accuracies.columns[i]
            n = num_samples_bins[class_].sum()
            CECE_df[class_] = ((num_samples_bins[class_]/n*
                                np.abs(accuracies[class_]-accuracies.index)).sum())
            
        CECE = CECE_df.mean()
        return CECE, CECE_df
        
        
    def _min_dist_NN_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]

        # Extract means, variance, and classes
        means_NN = self.means[:,ranks].cuda()
        vars_NN = self.vars[ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts.cpu()/num_MC
        probs = {key:0 for key in np.unique(self.classes)}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()
        
        return probs
    
    
    
    
    
    def _min_dist_NN_test_time_sampling_(self, ranks, mean_emb, var_emb):
        import time
        # extract num_NN nearest neighbours and use 1000 MC samples
        num_MC = 200000
        D = 15
    
        # Extract means, variance, and classes
        means_NN = torch.randn((D,1)).cuda()
        vars_NN = torch.ones((1,)).cuda()
        mean_emb = torch.randn((D,1)).cuda()
        var_emb = 2*torch.ones((1,)).cuda()
    
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        df = D
        
        # TEST SAMPLING FROM KNOWN DISTRIBUTION
        t_dist_0 = time.time()
        # Sample dist
        scaling.cpu().numpy()*noncentral_chisquare(df,nonc,num_MC)
        t_dist_1 = time.time()

    
        # TEST SAMPLING FROM GAUSSIAN DISTRIBUTION THEN CALC DISTANCES
        t_gaus_0 = time.time()
        emb_samples = torch.distributions.Normal(mean_emb,torch.Tensor.repeat(var_emb,mean_emb.shape[0])).rsample(torch.Size((num_MC,)))
        rank_samples = torch.distributions.Normal(means_NN.flatten(),torch.Tensor.repeat(vars_NN,mean_emb.shape[0])).rsample(torch.Size((num_MC,)))
        (emb_samples - rank_samples).pow(2).sum(1).sqrt()
        
        t_gaus_1 = time.time()
        
        return t_dist_1-t_dist_0, t_gaus_1-t_gaus_0
    
    
    