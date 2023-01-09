import torch
import json
import os
import numpy as np
import pandas as pd
from numpy.random import noncentral_chisquare
from torchvision import transforms
from torch_scatter import scatter_sum
from src.models.image_classification_network import init_network
import pdb
from os.path import exists
import multiprocessing

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader
from src.utils.helper_functions import get_logger_test

logger = get_logger_test('__main__')



class ImageClassifier():
    
    def __init__(self, model_name: str,
                 model_type: str,
                 params: dict,
                 model_data: str,
                 with_OOD: bool = False,
                 calibration_method: str = 'None'):
        
        # *****************************************
        # ******** Initialize model data **********
        # *****************************************
        super().__init__()
        
        # Set paths to data
        self.data_root = './data/processed/'
        self.ims_root = './data/raw/JPEGImages/'
        self.with_OOD = with_OOD
        self.model_data = model_data
        self.model_name = model_name
        self.calibration_method = calibration_method
        db = self._create_db_for_model_data_()
        
        # List of all classes
        self.all_class_list = [key for key in db.keys()]
        
        # ****************************************
        # ******** Load model and data ***********
        # ****************************************
        
        # Load model
        self.model_type = model_type
        self.params = params
        print(f'** Setting random seed to model training seed: {params["seed"]} **')
        torch.manual_seed(self.params['seed'])
        self.model = init_network(self.model_type, self.params)
        if self.model_type == 'BayesianTripletLoss':
            if self.calibration_method == 'SWAG':
                self.model.load_state_dict(torch.load(f'models/state_dicts_swag/{self.model_name}.pt'))
            else:
                self.model.load_state_dict(torch.load(f'models/state_dicts/{self.model_name}.pt'))
        else:
            self.model.load_state_dict(torch.load(f'models/state_dicts_classic/{self.model_name}.pt'))
        self.model.cuda()
        self.model.eval()
        
        # Select subset of classes (depends on the classes the model was trained on)
        self.classes_not_trained_on = np.sort(self.params.get('classes_not_trained_on',[])).tolist()
        self.classes_trained_on = ([class_ for class_ in self.all_class_list 
                                            if class_ not in self.classes_not_trained_on])
        self.classes_trained_on = np.sort(self.classes_trained_on).tolist()
        self.num_classes = len(self.classes_trained_on)
        
        # Create db for classes trained on (with OOD if wanted)
        if self.with_OOD == True:
            if self.model_type != 'BayesianTripletLoss':
                Warning("Model is not compatible with OOD classes - OOD not included")
                db = {key:db[key] for key in self.classes_trained_on}
        else:
            db = {key:db[key] for key in self.classes_trained_on}
                
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
        
        # Extract list of classes
        class_num = [[key]*len(db[key]) for key in db.keys()]
        self.classes = []
        [self.classes.extend(class_list) for class_list in class_num]
        
        # Mark objects not trained on if with_OOD is true
        if (self.with_OOD == True) & (self.model_type == 'BayesianTripletLoss'):
            for i in range(len(self.classes)):
                if self.classes[i] in self.classes_not_trained_on:
                    self.classes[i] = self.classes[i] + '_OOD'
               
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
            
        # Mark objects not trained on with _OOD 
        for i in range(len(classes)):
            if classes[i] in self.classes_not_trained_on:
                classes[i] = classes[i] + '_OOD'
        #if self.with_OOD == True:
            
        #else:
        #    idx_trained_on = [i for i in range(len(classes)) if classes[i] 
        #                                                    not in self.classes_not_trained_on]
        #    objects = [objects[i] for i in idx_trained_on]
        #    bboxs = [bboxs[i] for i in idx_trained_on]
        #    classes = [classes[i] for i in idx_trained_on]
            
        return objects, bboxs, classes
    
    
    def _create_dict_of_class_idxs_(self):
        # Split classes into list with indices of positions
        if self.with_OOD == True & (self.model_type == 'BayesianTripletLoss'):
            self.unique_classes = np.sort(np.unique(self.classes))
        else:
            self.unique_classes = self.classes_trained_on
            
        self.classes_idxs = {class_:[] for class_ in self.unique_classes}
        for i, class_ in enumerate(self.classes):
            self.classes_idxs[class_].append(i)
        self.num_samples_classes = ({class_: len(self.classes_idxs[class_]) for class_ in 
                                               self.classes_idxs.keys()})
    
    def get_probability_dist_dataset(self):
        raise NotImplementedError


class ImageClassifier_BayesianTripletLoss(ImageClassifier):
    
    def __init__(self, model_name: str,
                 model_type: str,
                 params: dict,
                 model_data: str,
                 with_OOD: bool = False,
                 balanced_classes: int = 0,
                 calibration_method: str = 'None'):
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
        super().__init__(model_name, model_type, params, model_data, with_OOD,calibration_method)
        self.balanced_classes = balanced_classes
        
        # ************************************************************************
        # ******** Extract means, variance and classes for model data  ***********
        # ************************************************************************
        # Find means and variances of model data
        file_name = './models/embeddings/' + self._get_embedding_file_storing_name_()
        
        if exists(file_name+'means.pt') == False:
            self.means, self.vars, self.backbone_repr, self.classes = self._extract_means_and_variances_()
            torch.save(self.means,file_name+'means.pt')
            torch.save(self.vars,file_name+'vars.pt')
            torch.save(self.classes,file_name+'classes.pt')
        else:
            if ((self.calibration_method == 'SWAG') | (self.calibration_method == 'MCDropout')):
                logger.info('SWAG and MCDropout calibration need backbone representations... Loading them first')
                self.means, self.vars, self.backbone_repr, self.classes = self._extract_means_and_variances_()
            
            else:
                logger.info('Loading already existing classifier model')
                self.means = torch.load(file_name+'means.pt')
                self.vars = torch.load(file_name+'vars.pt')
                self.classes = torch.load(file_name+'classes.pt')
                self.backbone_repr = None
            
                
   
        # Init dict of class idx
        self._create_dict_of_class_idxs_()
        
        if self.calibration_method == 'SWAG':
            if self.model.with_swag == False:
                raise ValueError("Model can not be evaulated with SWAG - does not exist")
            else:
                logger.info('Loading SWAG headers')
                self.model.head_mean__mean = torch.load(f'./models/swag_headers/{model_name}_mean_swag__mean.pt')
                self.model.head_mean__var = torch.load(f'./models/swag_headers/{model_name}_mean_swag__var.pt')
                self.model.head_std__mean = torch.load(f'./models/swag_headers/{model_name}_std_swag__mean.pt')
                self.model.head_std__var = torch.load(f'./models/swag_headers/{model_name}_std_swag__var.pt')
                
        self.out_rand = None
        self.out_rand_init = False
   
    # --------------------------------------------------------------------------------------------
    # ----------------------------------- Public functions ---------------------------------------
    # --------------------------------------------------------------------------------------------  
    def get_probability_dist_dataset(self, test_dataset: str = 'test',
                                           num_NN: int = 100, 
                                           num_MC: int = 10000,
                                           method: str = 'min_dist_NN',
                                           dist_classes: str = 'unif'):
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
                                    kNN-gauss-kernel: ---- 

        Returns:
            probs_df (pd.DataFrame): Pandas dataframe containing the probability dist over classes
                                     for all samples in the test dataset
            classes (list):          List of true classes 
        """
        # Extract test dataset
        objects, bboxs, classes = self._extract_test_dataset_(test_dataset)
        
        # Get file names
        file_name = self._get_test_file_storing_name_(num_NN,num_MC,method,
                                                      test_dataset,dist_classes)
        file_name_probs = f'./reports/probability_distributions/' + file_name + '.csv'
        
        file_name_embs = self._get_embedding_file_storing_name_()
        file_name_embs = f'./reports/embeddings_test/' + file_name_embs + f'_{test_dataset}'
        
        # If probs have not already been calculated, extract them
        if exists(file_name_probs) == False:
            logger.info('Extracting probability distributions')
            if method == 'mixed':
                probs_df_min = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                        columns = [i for i in range(len(classes))],
                                        index = self.unique_classes)
                probs_df_gauss = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                        columns = [i for i in range(len(classes))],
                                        index = self.unique_classes)
                probs_df_mixed = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                        columns = [i for i in range(len(classes))],
                                        index = self.unique_classes)
            else:
                probs_df = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                        columns = [i for i in range(len(classes))],
                                        index = self.unique_classes)
            
            # For storing mean_emb and var_emb
            if self.params['var_type'] == 'iso':
                mean_dim = self.params['dim_out']-1
                var_dim = 1
            else:
                mean_dim = int(self.params['dim_out']/2)
                var_dim = int(self.params['dim_out']/2)
                
            mean_embs = torch.zeros(mean_dim, len(classes)).cuda()
            var_embs = torch.zeros(var_dim,len(classes)).cuda()
            
            torch.manual_seed(self.params['seed'])
            self.rnd_states = torch.randperm(100000000)[:100]
            for i in range(len(classes)):
                probs, mean_emb, var_emb = self._get_probability_dist_(objects[i],
                                                                       bboxs[i],
                                                                       num_NN,
                                                                       num_MC,
                                                                       method,
                                                                       dist_classes)
                if method == 'mixed':
                    probs_df_min[i] = np.round(np.array(list(probs[0].values())),int(np.log10(num_MC)+1))
                    probs_df_gauss[i] = np.round(np.array(list(probs[1].values())),int(np.log10(num_MC)+1))
                    probs_df_mixed[i] = np.round(np.array(list(probs[2].values())),int(np.log10(num_MC)+1))
                else:
                    probs_df[i] = np.round(np.array(list(probs.values())),int(np.log10(num_MC)+1))
                    
                mean_embs[:,i] = mean_emb
                var_embs[:,i] = var_emb
                
                if (i+1) % 100 == 0:
                    logger.info('>>>> {}/{} done... '.format(i+1, len(classes)))
                    
            # Save dataframe to reports (for later use)
            if method == 'mixed':
                file_name = self._get_test_file_storing_name_(num_NN,num_MC,'min_dist_NN',
                                                      test_dataset,dist_classes)
                file_name_probs = f'./reports/probability_distributions/' + file_name + '.csv'
                probs_df_min.to_csv(file_name_probs)
                
                file_name = self._get_test_file_storing_name_(num_NN*10,num_MC,'kNN_gauss_kernel',
                                                      test_dataset,dist_classes)
                file_name_probs = f'./reports/probability_distributions/' + file_name + '.csv'
                probs_df_gauss.to_csv(file_name_probs)
                
                file_name = self._get_test_file_storing_name_(num_NN,num_MC,'mixed',
                                                      test_dataset,dist_classes)
                file_name_probs = f'./reports/probability_distributions/' + file_name + '.csv'
                probs_df_mixed.to_csv(file_name_probs)
                
                probs_df = probs_df_mixed
            else:
                probs_df.to_csv(file_name_probs)
                
            torch.save(mean_embs, file_name_embs + '_means.pt')
            torch.save(var_embs, file_name_embs + '_vars.pt')
        
        else:
            logger.info('Loading existing probability distribution')
            probs_df = pd.read_csv(file_name_probs,index_col=0)
        
        return probs_df, objects, bboxs, classes  
    
    def get_embeddings_of_test_dataset(self, test_dataset):
        
        file_name = self._get_embedding_file_storing_name_()
        file_name_embs = f'./reports/embeddings_test/' + file_name + f'_{test_dataset}'
        if exists(file_name_embs + '_means.pt'):
            means = torch.load(file_name_embs + '_means.pt')
            vars = torch.load(file_name_embs + '_vars.pt')
        else:
            raise RuntimeError('You have to call "get_probability_dist_dataset" before you can get'+
                               ' embeddings!')
    
        return means, vars
 
 
    # --------------------------------------------------------------------------------------------
    # ----------------------------------- Privat functions ---------------------------------------
    # -------------------------------------------------------------------------------------------- 
    def _get_embedding_(self, img_name: str, bbox: list):
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
        if self.model.var_type == 'iso':
            var_dim = 1
        else:
            var_dim = int(self.params['dim_out']/2)
            
        with torch.no_grad():
            backbone_emb = self.model.forward_backbone(img.cuda())
            output = self.model.forward_head(backbone_emb).data.squeeze()
            mean_emb = output[:-var_dim]
            var_emb = output[-var_dim:]
        
        return mean_emb, var_emb, backbone_emb

    def _get_probability_dist_(self, img_name: str, 
                                   bbox: list, 
                                   num_NN: int = 100, 
                                   num_MC: int = 10000,
                                   method: str = 'min_dist_NN',
                                   dist_classes: str = 'unif'):
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
        mean_emb, var_emb, backbone_emb = self._get_embedding_(img_name,bbox)
        
        if self.calibration_method == 'None':
            with torch.no_grad():
                dist = (torch.pow(self.means-mean_emb[None,:].T+1e-6, 2).sum(dim=0).sqrt())
                _, ranks = torch.sort(dist, dim=0, descending=False)
                
                if method == 'min_dist_NN':    
                    probs = self._min_dist_NN_(ranks, mean_emb, var_emb, num_NN, num_MC)
                elif method == 'kNN_gauss_kernel':
                    probs = self._kNN_gauss_kernel_(ranks, mean_emb, var_emb, num_NN, dist_classes)
                elif method == 'mixed':
                    probs_min = self._min_dist_NN_(ranks, mean_emb, var_emb, num_NN, num_MC)
                    probs_gauss = self._kNN_gauss_kernel_(ranks, mean_emb, var_emb, num_NN*10, dist_classes)
                    
                    probs_mixed = {key:0 for key in probs_min.keys()}
                    for key in probs_mixed:
                        probs_mixed[key] += (probs_min[key]+probs_gauss[key])/2
                        
                    probs = (probs_min,probs_gauss,probs_mixed)
        
                else:
                    raise ValueError('Method does not exist')
        
        
        elif (self.calibration_method == 'SWAG') | (self.calibration_method == 'MCDropout'):
            
            num_samples = 20
            
            if self.calibration_method == 'MCDropout':
                self.model.eval_with_dropout()
            
            if method == 'mixed':
                probs_swag_min = {key:0 for key in np.unique(self.classes)}
                probs_swag_gauss = {key:0 for key in np.unique(self.classes)}
                probs_swag_mixed = {key:0 for key in np.unique(self.classes)}
            else:
                probs_swag = {key:0 for key in np.unique(self.classes)}

            for j in range(num_samples):
                with torch.no_grad():
                    if self.calibration_method == 'SWAG':
                        if (self.out_rand_init is False):
                            out = self.model.forward_head_with_swag(self.backbone_repr,self.rnd_states[j])
                        out_emb = self.model.forward_head_with_swag(backbone_emb,self.rnd_states[j])
                    else:
                        if (self.out_rand_init is False):
                            out = self.model.forward_head(self.backbone_repr,self.rnd_states[j])
                        out_emb = self.model.forward_head(backbone_emb,self.rnd_states[j])
                    
                    # Store random init
                    if self.out_rand is None: #Init random embedding database
                        self.out_rand = torch.zeros(num_samples,out.shape[0],out.shape[1]).cuda()
                    if (self.out_rand_init is False):
                        self.out_rand[j] = out
                    
                    if (j == num_samples-1): #When initialized, set marker to True
                        self.out_rand_init = True
                    
                    means_NN = self.out_rand[j][:,:-self.model.var_dim].T
                    vars_NN = self.out_rand[j][:,-self.model.var_dim:].T
                    mean_emb = out_emb[0,:-self.model.var_dim]
                    var_emb = out_emb[0,-self.model.var_dim:]
                    
                    dist = (torch.pow(self.means-mean_emb[None,:].T+1e-6, 2).sum(dim=0).sqrt())
                    _, ranks = torch.sort(dist, dim=0, descending=False)
            
                if method == 'min_dist_NN':    
                    probs = self._min_dist_NN_(ranks, mean_emb, var_emb, num_NN, num_MC, 
                                               means_NN[:,ranks[:num_NN]], 
                                               vars_NN[:,ranks[:num_NN]])
                elif method == 'kNN_gauss_kernel':
                    probs = self._kNN_gauss_kernel_(ranks, mean_emb, var_emb, num_NN, dist_classes, 
                                                means_NN, 
                                                vars_NN)
                elif method == 'mixed':
                    probs_min = self._min_dist_NN_(ranks, mean_emb, var_emb, num_NN, num_MC, 
                                               means_NN[:,ranks[:num_NN]], 
                                               vars_NN[:,ranks[:num_NN]])
                    probs_gauss = self._kNN_gauss_kernel_(ranks, mean_emb, var_emb, num_NN*10, dist_classes, 
                                                means_NN, 
                                                vars_NN)
                    
                    probs_mixed = {key:0 for key in probs_min.keys()}
                    for key in probs_mixed:
                        probs_mixed[key] += (probs_min[key]+probs_gauss[key])/2
                
                    
                else:
                    raise ValueError('Method does not exist')
                
                
                if method == 'mixed':
                    for key in probs_swag:
                        probs_swag_min[key] += probs_min[key]/num_samples
                        probs_swag_gauss[key] += probs_gauss[key]/num_samples
                        probs_swag_mixed[key] += ((probs_swag_min[key]+probs_gauss[key])/2)/num_samples
                else:
                    for key in probs_swag:
                        probs_swag[key] += probs[key]/num_samples
             
            if method == 'mixed':
                probs = (probs_swag_min,probs_swag_gauss,probs_swag_mixed)
            else:  
                probs = probs_swag
        else:
            raise ValueError(f"{self.calibration_method} calibration method is not implemented for this model type")
        
        # reset seed to original
        torch.manual_seed(self.params['seed'])
        
        # reset evaluation method to original
        if self.calibration_method == 'MCDropout':
            self.model.eval()
            
        return probs, mean_emb, var_emb
    
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
                                batch_size=1, shuffle=False, 
                                num_workers=np.min([8,multiprocessing.cpu_count()]), 
                                pin_memory=False
            )

        logger.info("Extract means and variance for model data")

        # extract means and variances 
        if self.model.var_type == 'iso':
            var_dim = 1
        else:
            var_dim = int(self.params['dim_out']/2)
            
        backbone_repr= None
        
        # ********************
        # *** No balancing ***
        # ********************
        with torch.no_grad():
            if self.balanced_classes == 0:
                logger.info("No balancing of classes")
                qvecs = torch.zeros(self.params['dim_out']-var_dim, len(self.objects)).cuda()
                qvars = torch.zeros(var_dim,len(self.objects)).cuda()
            
                for i, input in enumerate(loader):
                    if ((self.calibration_method == 'SWAG') | 
                        (self.calibration_method == 'MCDropout')):
                        o = self.model.forward_backbone(input.cuda())
                        if i == 0: #init backbone representation tensor
                            try:
                                backbone_repr = torch.zeros(len(loader),o.shape[1],
                                                                        o.shape[2],
                                                                        o.shape[3]).cuda()
                                mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                                logger.info(f"Total memory consumption on Cuda: {mem_cuda:.3f}GB")
                            except:
                                raise MemoryError("Not enough memory on GPU for storing "+
                                                    "backbone representations. Either turn" +
                                                    "of SWAG calibration or lower the number"+
                                                    " of images in model database (i.e. by"+ 
                                                    " balancing the classes)")
                        backbone_repr[i] = o[0]
                        output = self.model.forward_head(o).data.squeeze()
                    else:
                        output = self.model(input.cuda()).data.squeeze()
                    qvecs[:, i] = output[:-var_dim]
                    qvars[:, i] = output[-var_dim:]
                    if (i+1) % 1000 == 0 or (i+1) == len(self.objects):
                        logger.info('>>>> {}/{} done... '.format(i+1, len(self.objects)))
            
                classes = self.classes
        
            # *****************
            # *** Balancing ***
            # *****************
            elif self.balanced_classes > 0:
                logger.info(f"Balancing of classes: Total number of samples from each class will be "+
                            f"{self.balanced_classes}")
                dist_of_classes = pd.value_counts(self.classes)
                num_samples = len(dist_of_classes)*self.balanced_classes
                qvecs = torch.zeros(self.params['dim_out']-var_dim, num_samples).cuda()
                qvars = torch.zeros(var_dim,num_samples).cuda()
                
                # for track-keeping
                classes_count = {dist_of_classes.index[i]:0 for i in range(len(dist_of_classes))}
                classes = []
                count = 0
                
                while True:
                    r_ord = torch.randperm(len(self.objects))
                    for i in range(len(self.objects)):
                        sample_class = self.classes[r_ord[i]]
                        if classes_count[sample_class] < self.balanced_classes: # add object
                            
                            if dist_of_classes[sample_class] >= classes_count[sample_class]: # no aug
                                input = OFL.__getitem__(r_ord[i])
                            else:
                                input = OFL_aug.__getitem__(r_ord[i])
                            
                            # save output
                            input = input[None,:,:,:]
                            if ((self.calibration_method == 'SWAG') | 
                                (self.calibration_method == 'MCDropout')):
                                o = self.model.forward_backbone(input.cuda())
                                if count == 0: #init backbone representation tensor
                                    try:
                                        backbone_repr = torch.zeros(num_samples,o.shape[1],
                                                                                o.shape[2],
                                                                                o.shape[3]).cuda()
                                        mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                                        logger.info(f"Total memory consumption on Cuda: {mem_cuda:.3f}GB")
                                    except:
                                        raise MemoryError("Not enough memory on GPU for storing "+
                                                          "backbone representations. Either turn" +
                                                          "of SWAG calibration or lower the number"+
                                                          " of images in model database (i.e. by"+ 
                                                          " balancing the classes)")
                                backbone_repr[count] = o[0]
                                output = self.model.forward_head(o).data.squeeze()
                            else:
                                output = self.model(input.cuda()).data.squeeze()
                            qvecs[:, count] = output[:-var_dim]
                            qvars[:,count] = output[-var_dim:]
                            
                            # add counts and class
                            classes.append(sample_class)
                            classes_count[sample_class] += 1
                            count += 1
                            if (count) % 1000 == 0 or (count) == num_samples:
                                logger.info('>>>> {}/{} done... '.format(count, num_samples))
                                
                        if count == num_samples:
                            break;
                        
                    if count == num_samples:
                        break;
            
            # *****************************
            # *** Upsampling to minimum ***
            # *****************************
            else:
                upsampling_size = np.abs(self.balanced_classes)
                logger.info(f"Upsampling of classes: The least number of samples from each class will be "+
                        f"{upsampling_size}")
                dist_of_classes = pd.value_counts(self.classes)
                num_samples = sum([max(upsampling_size,dist_of_classes[i]) for i in dist_of_classes.index])
                qvecs = torch.zeros(self.params['dim_out']-var_dim, num_samples).cuda()
                qvars = torch.zeros(var_dim,num_samples).cuda()   
                
                # Add original
                logger.info("Adding original images")
                for i, input in enumerate(loader):
                    if ((self.calibration_method == 'SWAG') | 
                        (self.calibration_method == 'MCDropout')):
                        o = self.model.forward_backbone(input.cuda())
                        if i == 0: #init backbone representation tensor
                            try:
                                backbone_repr = torch.zeros(num_samples,o.shape[1],
                                                                        o.shape[2],
                                                                        o.shape[3]).cuda()
                                mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                                logger.info(f"Total memory consumption on Cuda: {mem_cuda:.3f}GB")
                            except:
                                raise MemoryError("Not enough memory on GPU for storing "+
                                                    "backbone representations. Either turn" +
                                                    "of SWAG calibration or lower the number"+
                                                    " of images in model database (i.e. by"+ 
                                                    " balancing the classes)")
                                
                        backbone_repr[i] = o[0]
                        output = self.model.forward_head(o).data.squeeze()
                    else:
                        output = self.model(input.cuda()).data.squeeze()
                    qvecs[:, i] = output[:-var_dim]
                    qvars[:, i] = output[-var_dim:]
                    if (i+1) % 1000 == 0 or (i+1) == len(self.objects):
                        logger.info('>>>> {}/{} done... '.format(i+1, len(self.objects)))
            
                classes = self.classes
                
                count=sum(dist_of_classes)
                
                # add augmentations
                logger.info("Upsampling images")
                while True:
                    r_ord = torch.randperm(len(self.objects))
                    for i in range(len(self.objects)):
                        sample_class = self.classes[r_ord[i]]
                        if dist_of_classes[sample_class] < upsampling_size: # add object
                            
                            # Get augmentation
                            input = OFL_aug.__getitem__(r_ord[i])
                            
                            # save output
                            input = input[None,:,:,:]
                            if ((self.calibration_method == 'SWAG') | 
                                (self.calibration_method == 'MCDropout')):
                                o = self.model.forward_backbone(input.cuda())
                                backbone_repr[count] = o[0]
                                output = self.model.forward_head(o).data.squeeze()
                            else:
                                output = self.model(input.cuda()).data.squeeze()
                            qvecs[:, count] = output[:-var_dim]
                            qvars[:,count] = output[-var_dim:]
                            
                            # add counts and class
                            classes.append(sample_class)
                            dist_of_classes[sample_class] += 1
                            count += 1
                            if (count) % 1000 == 0 or (count) == num_samples:
                                logger.info('>>>> {}/{} done... '.format(count, num_samples))
                                
                        if count == num_samples:
                            break;
                    if count == num_samples:
                        break;
        
        return qvecs, qvars, backbone_repr, classes
    
    def _get_test_file_storing_name_(self,num_NN,
                                     num_MC,
                                     method,
                                     test_dataset,
                                     dist_classes):
        
        if method == 'min_dist_NN':
            file_name = (f'{self.model_name}_{self.model_data}_'+
                            f'{self.balanced_classes}_numNN{num_NN}_'+
                            f'numMC{num_MC}_{method}_{test_dataset}_'+
                            f'withOOD{self.with_OOD}_'+
                            f'varType-{self.model.var_type}_'+
                            f'calibrationMethod-{self.calibration_method}')
        elif method == 'kNN_gauss_kernel':
            if dist_classes == 'all':
                file_name = (f'{self.model_name}_{self.model_data}_'+
                                f'{self.balanced_classes}_'+
                                f'{method}_{dist_classes}_{test_dataset}_'+
                                f'withOOD{self.with_OOD}_'+
                                f'varType-{self.model.var_type}_'+
                                f'calibrationMethod-{self.calibration_method}')
            elif dist_classes == 'unif':
                file_name = (f'{self.model_name}_{self.model_data}_'+
                                f'{self.balanced_classes}_'+
                                f'{method}_{dist_classes}_{test_dataset}_'+
                                f'withOOD{self.with_OOD}_'+
                                f'varType-{self.model.var_type}_'+
                                f'calibrationMethod-{self.calibration_method}')
            else: 
                file_name = (f'{self.model_name}_{self.model_data}_'+
                                f'{self.balanced_classes}_numNN{num_NN}_'+
                                f'{method}_{dist_classes}_{test_dataset}_'+
                                f'withOOD{self.with_OOD}_'+
                                f'varType-{self.model.var_type}_'+
                                f'calibrationMethod-{self.calibration_method}')
                
        elif (method == 'mixed'):
            file_name = (f'{self.model_name}_{self.model_data}_'+
                            f'{self.balanced_classes}_numNN{num_NN}_numMC{num_MC}_'+
                            f'{dist_classes}_'+
                            f'{method}_{test_dataset}_'+
                            f'withOOD{self.with_OOD}_'+
                            f'varType-{self.model.var_type}_'+
                            f'calibrationMethod-{self.calibration_method}')
        
        return file_name
    
    def _get_embedding_file_storing_name_(self):
        file_name = (f'{self.model_name}_{self.model_data}_'+
                     f'{str(self.balanced_classes)}_'+
                     f'withOOD{self.with_OOD}_'+
                     f'varType-{self.model.var_type}_')
        return file_name
    
    def _min_dist_NN_(self, ranks, mean_emb, var_emb, num_NN, num_MC, means_NN=None, vars_NN=None):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]
        # Extract means, variance, and classes
        if means_NN is None:
            means_NN = self.means[:,ranks].cuda()
            vars_NN = self.vars[:,ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
        
        if self.model.var_type == 'iso':
            vars_NN = (torch.Tensor.repeat(vars_NN.flatten(),mean_emb.shape[0])
                                   .reshape(mean_emb.shape[0],-1))
            var_emb = (torch.Tensor.repeat(var_emb.flatten(),mean_emb.shape[0])
                                   .reshape(mean_emb.shape[0],))
        
        emb_samples = torch.distributions.Normal(mean_emb,var_emb).rsample(torch.Size((num_MC,)))
        rank_samples = (torch.distributions.Normal(means_NN.T.flatten(),vars_NN.T.flatten()).rsample(torch.Size((num_MC,)))
                                                  .reshape(num_MC,mean_emb.shape[0],-1))
        
        rank_samples = rank_samples.permute(2, 0, 1)
        dist_to_NN = (torch.sub(rank_samples,emb_samples)).pow(2).sum(2)
    
        """ OLD
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        """
        
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts.cpu()/num_MC
        probs = {key:0 for key in np.unique(self.classes)}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()
        
        return probs 
       
    def _kNN_gauss_kernel_(self, ranks, mean_emb, var_emb, num_NN, dist_classes = 'all', means_NN=None, vars_NN=None):
        if means_NN is None:
            # Extract database images of interest
            if dist_classes=='unif':
                # Extract as many objects from each class as the one with the lowest count
                min_samples_class = min(self.num_samples_classes.values())
                ranks_new = np.zeros((self.num_classes*min_samples_class,))
                ranks_dict = {}
                for i, class_ in enumerate(self.num_samples_classes.keys()):
                    class_selections = np.random.choice(self.classes_idxs[class_],
                                                            min_samples_class, False)
                    ranks_new[i*min_samples_class:(i+1)*min_samples_class] = class_selections
                    ranks_dict[class_] = class_selections
            elif dist_classes=='nn':
                # extract num_NN nearest neighbours
                ranks_new = ranks[:num_NN].cpu().numpy()
            elif dist_classes=='all':
                # extract all 
                ranks_new = ranks.cpu().numpy()
            else:
                raise ValueError("dist_classes not known - choose between [unif,nn,all]")
        
            # Extract means, variance, and classes
            ranks_new = ranks_new.astype(int)
            
            means_NN = self.means[:,ranks_new].cuda()
            vars_NN = self.vars[:,ranks_new].cuda()
            classes_NN = [self.classes[i] for i in ranks_new]
        
        else:
            ranks_new = ranks.cpu().numpy()
            ranks_new = ranks_new.astype(int)
            
            if dist_classes=='all':
                classes_NN = [self.classes[i] for i in ranks_new]
            elif dist_classes=='nn':
                classes_NN = [self.classes[i] for i in ranks_new[:num_NN]]
                means_NN = means_NN[:,ranks_new[:num_NN]]
                vars_NN = vars_NN[:,ranks_new[:num_NN]]
            else:
                raise ValueError("Method not valid for calibrated run")
        
        # Extract the expected distance dimension wise (since isotropic gaussians) and sum up 
        means_per_dim_sub = torch.pow(torch.sub(means_NN.T,mean_emb),2)
        vars_per_dim = vars_NN.T+var_emb
        expected_dist_per_dim = vars_per_dim*(1+torch.pow(vars_per_dim,-1)*means_per_dim_sub)
        expected_dist = expected_dist_per_dim.sum(axis=1)
        
        # Calculate all Gaussian Kernels
        D = self.means.shape[0]
        if self.means.shape[0] > self.vars.shape[0]:
            tr_var = var_emb
        else:
            tr_var = torch.mean(var_emb)
        f_Bayes = torch.exp(-expected_dist/(2*tr_var))    
        
        # Accumulate on class level per sample
        classes, idx = np.unique(classes_NN, return_inverse=True)
        idx = torch.Tensor(idx).type(torch.int64).cuda()
        f_Bayes_classwise = torch.zeros((len(classes),)).cuda()
        scatter_sum(src=f_Bayes,index=idx,out=f_Bayes_classwise,dim=0)
        
        # Calculate total Bayes
        total_f_Bayes = torch.sum(f_Bayes_classwise)
        
        # Get class-wise probs
        probs_Bayes = f_Bayes_classwise/total_f_Bayes
    
        probs = {key:0 for key in self.unique_classes}
        for i, class_ in enumerate(classes):
            if np.isnan(probs_Bayes[i].item())==False:
                probs[class_]=probs_Bayes[i].item()
            else:
                probs[class_]=1/self.num_classes
            
        return probs
    
    
            
        
    # --------------------------------------------------------------------------------------------
    # ----------------------------------- Unused functions ---------------------------------------
    # -------------------------------------------------------------------------------------------- 
    def _min_dist_NN_test_time_sampling_(self, ranks, mean_emb, var_emb):
        #! DO NOT USE (Only for testing implementation time)
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
    
    
class ImageClassifier_Classic(ImageClassifier):
    
    def __init__(self, model_name: str,
                 model_type: str,
                 params: dict,
                 model_data: str,
                 calibration_method: str):
        """Image classifier model. Classic 

        Args:
            model (str): model name (path to params and state_dict should be in ./models/params and 
                         ./models/state_dict respectively, with names {model}.pt and {model}.json)
            model_data (str): TRAIN or TRAINVAL (use either training data or training and val data).
        """
        super().__init__(model_name, model_type, params, model_data,calibration_method)
        self._create_dict_of_class_idxs_()
        all_classes = ['train','cow','tvmonitor','boat','cat','person','aeroplane','bird',
                       'dog','sheep','bicycle','bus','motorbike','bottle','chair',
                       'diningtable','pottedplant','sofa','horse','car']
        output_classes = ([class_ for class_ in all_classes if class_ 
                                    not in self.classes_not_trained_on]) 
        self.output_class_idx = [output_classes.index(class_) for class_ in self.unique_classes]
        
        if self.calibration_method == 'SWAG':
            if self.model.with_swag == False:
                raise ValueError("Model can not be evaulated with SWAG - does not exist")
            else:
                self.model.head_mean = torch.load(f'./models/swag_headers/{model_name}_mean_swag.pt')
                self.model.head_std = torch.load(f'./models/swag_headers/{model_name}_var_swag.pt')
            
        

        
    def get_probability_dist_dataset(self, test_dataset: str):
        """Calculates the probability distributiuon over classes trained on for a set of images
           defined by 'test_dataset'. Name of 'test_dataset' should be present in data/processed/.

        Args:
            test_dataset (str, optional): Name of test dataset (json file). Defaults to 'test'.

        Returns:
            probs_df (pd.DataFrame): Pandas dataframe containing the probability dist over classes
                                     for all samples in the test dataset
            classes (list):          List of true classes 
        """
        # Extract test dataset
        objects, bboxs, classes = self._extract_test_dataset_(test_dataset)
        
        # Get file names
        file_name = self._get_test_file_storing_name_()
        file_name_probs = f'./reports/probability_distributions_classic/' + file_name + '.csv'
        
        # If probs habe not already been calculated, extract them
        if exists(file_name_probs) == False:
            logger.info('Extracting probability distributions')
            probs_df = pd.DataFrame(np.zeros((len(self.unique_classes),len(classes))),
                                    columns = [i for i in range(len(classes))],
                                    index = self.unique_classes)

            # Make fixed random states
            torch.manual_seed(self.params['seed'])
            self.rnd_states = torch.randperm(100000000)[:100]
            for i in range(len(classes)):
                # Get and save probs
                obj = objects[i]
                bbox = bboxs[i]
                probs = self._get_probability_dist_(obj,bbox)
                probs_df[i] = probs[self.output_class_idx]

                if (i+1) % 100 == 0:
                    logger.info('>>>> {}/{} done... '.format(i+1, len(classes)))
                    
            # Save dataframe to reports (for later use)
            probs_df.to_csv(file_name_probs)
        else:
            logger.info('Loading existing probability distribution')
            probs_df = pd.read_csv(file_name_probs,index_col=0)
        
        return probs_df, objects, bboxs, classes  
        
        
    def _get_test_file_storing_name_(self):
        file_name = (f'{self.model_name}_{self.calibration_method}')
        
        return file_name
        
        
    def _get_probability_dist_(self, obj:str, bbox: list):
        # Set image path
        img_path = self.ims_root + f'/{obj}'
        
        if self.calibration_method == 'None':
            with torch.no_grad():
                # Get image
                img = image_object_loader(img_path,bbox,self.transformer)
                img = img[None,:,:,:] #Expand dim
                
                # Get model output
                probs = torch.softmax(self.model(img.cuda()).squeeze(),-1).cpu().numpy()
        
        elif self.calibration_method == 'MCDropout':
            with torch.no_grad():
                # Get image
                img = image_object_loader(img_path,bbox,self.transformer)
                img = img[None,:,:,:] #Expand dim
                
                # Get model output
                backbone_repr = self.model.forward_backbone(img.cuda())
                self.model.eval_with_dropout()
                probs = torch.softmax(self.model.forward_head(backbone_repr,self.rnd_states[0])
                                      .squeeze(),-1).cpu().numpy()
                for i in range(24):
                    probs += torch.softmax(self.model.forward_head(backbone_repr,self.rnd_states[i])
                                           .squeeze(),-1).cpu().numpy()
                    
                probs = probs/25
                self.model.eval()
                
        elif self.calibration_method == 'TempScaling':
            with torch.no_grad():
                # Get image
                img = image_object_loader(img_path,bbox,self.transformer)
                img = img[None,:,:,:] #Expand dim
                
                # Get model output
                logits = self.model(img.cuda()).squeeze()
                logits = logits/self.params['temp_opt']
                probs = torch.softmax(logits,-1).cpu().numpy()

        elif self.calibration_method == 'SWAG':
            with torch.no_grad():
                # Get image
                img = image_object_loader(img_path,bbox,self.transformer)
                img = img[None,:,:,:] #Expand dim
                
                # Get model output
                o = self.model.forward_backbone(img.cuda())
                logits = self.model.forward_head_with_swag(o,self.rnd_states).squeeze()
                probs = torch.softmax(logits,-1).cpu().numpy()
        else: 
            raise ValueError("Unknown calibration method")
        
        return probs
            

    
def init_classifier_model(model: str,
                          model_data: str,
                          with_OOD: bool = False,
                          balanced_classes: int = 0,
                          calibration_method = 'None'):
    # Load model params
    f = open(f'./models/params/{model}.json',)
    params = json.load(f)
    model_type = params.get('model_type', None)
    print(f'** Loading model of type {model_type} **')
    
    if model_type == 'BayesianTripletLoss':
        classifier = ImageClassifier_BayesianTripletLoss(model,
                                                         model_type,
                                                         params,
                                                         model_data,
                                                         with_OOD,
                                                         balanced_classes,
                                                         calibration_method)
    elif model_type == 'Classic':
        classifier = ImageClassifier_Classic(model,
                                             model_type,
                                             params,
                                             model_data,
                                             calibration_method)
    else:
        raise ValueError("Model Type in params is unknown")
    
    return classifier