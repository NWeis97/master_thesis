# Standard
import os
import pickle
import pdb
import json
from PIL import Image
import logging
import numpy as np

# Torch
import torch
import torch.utils.data as data
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader
from src.models.image_classification_network import ImageClassifierNet_BayesianTripletLoss

# Init logger
logger = logging.getLogger('__main__')


class TuplesDataset_2class(data.Dataset):
    """Data loader that loads training and validation tuples
        (based on 'cnnimageretrieval-pytorch/cirtorch/datasets/traindataset.py' by 'filipradenovic')
    
    Args:
        mode (string): 'train' or 'val' for training and validation parts of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        nnum (int, Default:10): Number of negatives for a query image in a training tuple
        poolsize_class (int, Default:200): Number of images per class in pool
        keep_prev_tuples (bool, Default:True): Should we keep tuples from previous generation?
        classes_not_trained_on (list, Default:[]): What classes should we not train on
        approx_similarity (bool, Default:True): Should the similarity measure be approximative (dot
                                                product) or exact (l2-norm)
     
     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qidxs (list): List of qsize_class*#classes query image indexes to be processed in an epoch
        pidxs (list): List of qsize_class*#classes positive image indexes, each corresponding to 
                      query at the same position in qidxs
        nidxs (list): List of qsize_class*#classes tuples of negative images
                      Each nidxs tuple contains nnum images corresponding to query image at the 
                      same position in qidxs
        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, mode: str, nnum: int=10,
                 poolsize_class:int = 200, transform: transforms=None, keep_prev_tuples: bool=True, 
                 classes_not_trained_on: list=[], approx_similarity: bool=True):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        # Define attributes
        self.mode = mode
        self.loader = image_object_loader
        self.keep_prev_tuples = keep_prev_tuples
        self.approx_similarity = approx_similarity
        self.classes_not_trained_on = classes_not_trained_on
        self.transform = transform
        self.print_freq = 500
        self.require_grad = True
        if self.mode == 'val':
            self.require_grad = False
        
        # Set paths to data
        data_root = './data/processed/'
        db_root = os.path.join(data_root, self.mode+'.json')
        self.ims_root = './data/raw/JPEGImages/'

        # Load database
        f = open(db_root)
        db = json.load(f)
        f.close()
        
        # Select subset of classes
        class_list = ['train','cow','tvmonitor','boat','cat','person','aeroplane','bird',
                      'dog','sheep','bicycle','bus','motorbike','bottle','chair',
                      'diningtable','pottedplant','sofa','horse','car']
        db = ({class_:db[class_] for class_ in class_list if 
                                     class_ not in self.classes_not_trained_on})
        
        # Extract list of classes
        class_num = [[key]*len(db[key]) for key in db.keys()]
        self.classes = []
        [self.classes.extend(class_list) for class_list in class_num]
        
        # Get index hash for each class
        self.class_hash = {}
        count = 0
        for key in db.keys():
            self.class_hash[key] = (count,count+len(db[key]))
            count += len(db[key])

        # Extract list of images
        obj_names = [[db[key][j]['img_name'] for j in range(len(db[key]))] for key in db.keys()]
        self.objects = []
        [self.objects.extend(image_list) for image_list in obj_names]

        # Extract list of bbox
        bbox_list = [[db[key][j]['bbox'] for j in range(len(db[key]))] for key in db.keys()]
        self.bbox = []
        [self.bbox.extend(bbox) for bbox in bbox_list]

        # size of training subset for an epoch
        self.nnum = nnum
        self.poolsize_class = poolsize_class
        self.poolsize = self.poolsize_class*len(db.keys())
        
        # Init idxs list
        self.qidxs = [] # Query idxs (from pool)
        self.pidxs = [] # Positives idxs (from pool)
        self.nidxs = [] # Negatives idxs (from pool)

        # Init tensors for storing backbone output
        self.backbone_repr = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            objects tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` "+
                               "method to create subset for train/val!"))

        output = []

        # query object
        output.append(self.backbone_repr[self.qidxs[index]])
        # positive object
        output.append(self.backbone_repr[self.pidxs[index]])
        # negative objects
        for i in range(self.nnum):
            output.append(self.backbone_repr[self.nidxs[index][i]])

        target = torch.Tensor([-1, 1] + [0]*self.nnum)

        return output, target, self.classes[self.qidxs[index]]

    def __len__(self):
        if self.qidxs is not None:
            return len(self.qidxs)
        else:
            return 0

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Mode: {} {}\n'.format(self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.objects))
        fmt_str += '    Poolsize: {}\n'.format(self.poolsize)
        fmt_str += '    Number of training tuples (for each class): {}\n'.format(self.poolsize_class)
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + 
                                     ' ' * len(tmp)))
        return fmt_str

    def update_backbone_repr_pool(self, net: ImageClassifierNet_BayesianTripletLoss):
        logger.info('\n\n §§§§ Updating pool for *{}* dataset... §§§§\n'.format(self.mode))
        
        #prepare net
        net.cuda()
        if self.mode == 'train':
            net.eval()#net.train()
        else:
            net.eval()
        
        self.backbone_repr = []
        
        # draw pool_size random images from each class (constitues balanced pool)
        idxs2pool = {class_:torch.IntTensor() for class_ in self.class_hash.keys()}
        pool2class = {class_:torch.IntTensor() for class_ in self.class_hash.keys()}
        count = 0
        for key in self.class_hash:
            min = self.class_hash[key][0]
            max = self.class_hash[key][1]
            pool = min+torch.randperm(max-min)
            num_class_to_pool = np.min([self.poolsize_class,len(pool)])
            idxs2pool[key] = pool[:num_class_to_pool].tolist()
            pool2class[key] = torch.arange(count,count+num_class_to_pool)
            count += num_class_to_pool

        idxs2pool = idxs2pool.values()
        idxs2pool_list = []
        ([idxs2pool_list.extend(i) for i in idxs2pool])
        self.pool_idx_to_class = pool2class
        
        # prepare loader
        loader = torch.utils.data.DataLoader(
            ObjectsFromList(root=self.ims_root, 
                            obj_names=[self.objects[i] for i in idxs2pool_list], 
                            obj_bbox=[self.bbox[i] for i in idxs2pool_list],
                            transform=self.transform),
                            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        
        # extract positive vectors
        for i, input in enumerate(loader):
            if self.require_grad:
                o = net.forward_backbone(input.cuda())
            else:
                with torch.no_grad():
                    o = net.forward_backbone(input.cuda())
            self.backbone_repr.append(o)
            if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2pool_list):
                logger.info('>>>> {}/{} done... '.format(i+1, len(idxs2pool_list)))
        self.backbone_repr = torch.concat(self.backbone_repr,dim=0)
        
        self.pool_classes_list = []
        ([self.pool_classes_list.extend([i]*len(self.pool_idx_to_class[i])) for 
                                            i in self.pool_idx_to_class.keys()])
        
    def create_epoch_tuples(self, net: ImageClassifierNet_BayesianTripletLoss, num_classes_per_neg):

        logger.info('\n>> Creating tuples for *{}* dataset...'.format(self.mode))
        #if self.keep_prev_tuples:
        #    logger.info('>> Keeping tuples from previous generation')        

        # prepare network
        net.cuda()
        if self.mode == 'train':
            net.train()
        else:
            net.eval()
    
        # **********************************
        # **** SELECTING POSITIVE PAIRS ****
        # **********************************
        idxs2qpool = torch.IntTensor()
        idxs2ppool = torch.IntTensor()

        for class_ in self.pool_idx_to_class.keys():
            class_idxs = self.pool_idx_to_class[class_]
            idxs2qpool = torch.cat((idxs2qpool,class_idxs))
            perm_idxs = torch.randperm(len(class_idxs))
            while torch.sum(class_idxs[perm_idxs] == self.pool_idx_to_class[class_]) != 0:
                perm_idxs = torch.randperm(len(class_idxs))
                
            idxs2ppool = torch.cat((idxs2ppool,class_idxs[perm_idxs]))

        self.qidxs = idxs2qpool.tolist()
        self.pidxs = idxs2ppool.tolist()
        
        # **********************************
        # **** SELECTING NEGATIVE PAIRS ****
        # **********************************
        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0
        
        # Init lists for query, positive and negative means and variances
        q_means = []
        q_vars = []
        n_means = []
        n_vars = []
        
        # **************************************************
        # ***** Query and Positive images extraction *******
        # **************************************************
        # extract query vectors
        logger.info('>> Extracing pool images...')
        for i in self.qidxs:
            if self.require_grad:
                output = net.forward_head(self.backbone_repr[[i]])
            else:
                with torch.no_grad():
                    output = net.forward_head(self.backbone_repr[[i]])
            q_means.append(output[:,:-net.var_dim].T)
            q_vars.append(output[:,-net.var_dim:].T)
            if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                logger.info('>>>> {}/{} done... '.format(i+1, len(self.qidxs)))
        q_means = torch.concat(q_means,dim=1)
        q_vars = torch.concat(q_vars,dim=1)

        # ****************************************
        # ***** Negative images extraction *******
        # ****************************************   
        logger.info('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        if self.approx_similarity:
            scores = torch.mm(q_means.t(), q_means)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
        else:
            scores = torch.cdist(q_means.t(),q_means.t(),2)
            scores, ranks = torch.sort(scores, dim=0, descending=False)
        
        avg_ndist = torch.tensor(0).float().cuda()  # for statistics
        n_ndist = torch.tensor(0).float().cuda()  # for statistics
       
        # ****************************************
        # ******** Hard Negative Mining **********
        # **************************************** 
        # selection of negative examples
        for q in range(len(self.qidxs)):
            # do not use query class,
            # those images are potentially positive
            qclass = self.pool_classes_list[self.qidxs[q]]
            num_class = {class_:0 for class_ in self.pool_idx_to_class.keys()}
            num_class[qclass] += num_classes_per_neg
            nidxs = []
            r = 0
            count = 0
            while len(nidxs) < self.nnum:
                potential = ranks[r, q]
                if num_class[self.pool_classes_list[potential]] < num_classes_per_neg:
                    nidxs.append(potential.item())
                    n_means.append(q_means[:,[r]])
                    n_vars.append(q_vars[:,[r]])
                    num_class[self.pool_classes_list[potential]] += 1
                    avg_ndist += (torch.pow(q_means[:,q]-q_means[:,potential]+1e-6, 2)
                                        .sum(dim=0).sqrt())
                    n_ndist += 1
                r += 1
            count += 1
            self.nidxs.append(nidxs)
             
            if q % 1000 == 0:
                logger.info(f">>>>>> {q}/{len(self.qidxs)}")
     
        n_means = torch.concat(n_means,dim=1).reshape(-1,len(self.qidxs),self.nnum)
        n_vars = torch.concat(n_vars,dim=1).reshape(-1,len(self.qidxs),self.nnum)
        logger.info('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
        logger.info('>>>> Done\n')


        # **************************
        # **** SAVE EXTRACTIONS **** not in use atm
        # **************************
        """
        if (self.keep_prev_tuples):    
            if len(self.q_out.shape) == 1: # Not initialized yet
                self.q_out = q_out
                self.p_out = p_out
                self.n_out = n_out
            else:    
                self.q_out = torch.concat((self.qp_out,q_out),dim=1)
                self.p_out = torch.concat((self.pp_out,p_out),dim=1)
                self.n_out = torch.concat((self.np_out,n_out),dim=1)
        """
    
                    
        # return average negative l2-distance, query vectors, variances, class (for plotting)
        return (avg_ndist/n_ndist).item(), q_means.detach(), q_vars.detach(), self.pool_idx_to_class 




