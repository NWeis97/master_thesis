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
        qsize_class (int, Default:20): Number of query images, ie number of (q,p,n1,...nN) tuples, 
                                         to be processed in one epoch (for each class)
        poolsize (int, Default:3000): Pool size for negative images re-mining
        keep_prev_tuples (bool, Default:True): Should we keep tuples from previous generation?
        num_classes (int, Default:6): How many classes should be trained on?
        approx_similarity (bool, Default:True): Should the similarity measure be approximative (dot
                                                product) or exact (l2-norm)
     
     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same 
                      position in qpool
        qidxs (list): List of qsize_class*#classes query image indexes to be processed in an epoch
        pidxs (list): List of qsize_class*#classes positive image indexes, each corresponding to 
                      query at the same position in qidxs
        nidxs (list): List of qsize_class*#classes tuples of negative images
                      Each nidxs tuple contains nnum images corresponding to query image at the 
                      same position in qidxs
        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, mode: str, nnum: int=10, qsize_class: int=20, poolsize:int =3000, 
                 transform: transforms=None, keep_prev_tuples: bool=True, num_classes: int=6,
                 approx_similarity: bool=True):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        # Define mode and loader
        self.mode = mode
        self.loader = image_object_loader
        
        # Define distance measure
        self.approx_similarity = approx_similarity

        # Set paths to data
        data_root = './data/processed/'
        db_root = os.path.join(data_root, self.mode+'.json')
        self.ims_root = './data/raw/JPEGImages/'

        # Load database
        f = open(db_root)
        db = json.load(f)
        f.close()
        
        # Select subset of classes
        self.num_classes = num_classes
        class_list = ['train','cow','tvmonitor','boat','cat','person','aeroplane','bird',
                      'dog','sheep','bicycle','bus','motorbike','bottle','chair',
                      'diningtable','pottedplant','sofa','horse','car']
        db = {class_list[i]:db[class_list[i]] for i in range(self.num_classes)}
        
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
        self.qsize_class = qsize_class
        self.qsize = qsize_class*len(db.keys())
        self.poolsize = min(poolsize, len(self.objects))
        self.qidxs = None # Query 
        self.pidxs = None # Positives 
        self.nidxs = None # Negatives
        self.qcidxs = [] # Query (current)
        self.pcidxs = [] # Positives (current)
        self.ncidxs = [] # Negatives (current)
        
        self.keep_prev_tuples = keep_prev_tuples
        if self.keep_prev_tuples:
            self.qpidxs = None # Query (previous)
            self.ppidxs = None# Positives (previous)
            self.npidxs = None # Negatives (previous)

        # Init transformer
        self.transform = transform
        self.print_freq = 1000

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
        output.append(self.loader(os.path.join(self.ims_root,self.objects[self.qidxs[index]]), 
                                  self.bbox[self.qidxs[index]],
                                  self.transform))
        # positive object
        output.append(self.loader(os.path.join(self.ims_root,self.objects[self.pidxs[index]]), 
                                  self.bbox[self.pidxs[index]],
                                  self.transform))
        # negative objects
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(os.path.join(self.ims_root,
                                                   self.objects[self.nidxs[index][i]]), 
                                  self.bbox[self.nidxs[index][i]],
                                  self.transform))

        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

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
        fmt_str += '    Number of training tuples: {}\n'.format(self.qsize)
        fmt_str += '    Number of training tuples (for each class): {}\n'.format(self.qsize_class)
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + 
                                     ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, net, num_classes_per_neg):

        logger.info('\n>> Creating tuples for an epoch of *{}*...'.format(self.mode))
        #if self.keep_prev_tuples:
        #    logger.info('>> Keeping tuples from previous generation')
        
        ## ------------------------
        ## SAVE PREVIOUS TUPLES
        ## ------------------------
        if (self.keep_prev_tuples):
            self.qpidxs = self.qcidxs.copy()
            self.ppidxs = self.pcidxs.copy()
            self.npidxs = self.ncidxs.copy()
            


        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
        #logger.info(f'>> Selecting {self.qsize_class} imgs for each class with a pos pair')
        # draw qsize random queries for tuples
        # draw randomly the same amount from each class
        idxs2qpool = torch.IntTensor()
        idxs2ppool = torch.IntTensor()
        classes_list = []
        for key in self.class_hash:
            min = self.class_hash[key][0]
            max = self.class_hash[key][1]
            pool = min+torch.randperm(max-min)
            idxs2qpool = torch.cat((idxs2qpool,pool[:self.qsize_class]))
            idxs2ppool = torch.cat((idxs2ppool,pool[self.qsize_class:self.qsize_class*2]))
            classes_list.extend([key]*self.qsize_class)

        self.qcidxs = idxs2qpool.tolist()
        self.pcidxs = idxs2ppool.tolist()

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.ncidxs = [[] for _ in range(len(self.qcidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        idxs2objects = torch.randperm(len(self.objects))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            logger.info('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ObjectsFromList(root=self.ims_root, 
                                obj_names=[self.objects[i] for i in self.qcidxs], 
                                obj_bbox=[self.bbox[i] for i in self.qcidxs],
                                transform=self.transform),
                                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            
            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim']-1, len(self.qcidxs)).cuda()
            qvars = torch.zeros(len(self.qcidxs),)
            for i, input in enumerate(loader):
                output = net(input.cuda()).data.squeeze()
                qvecs[:, i] = output[:-1]
                qvars[i] = output[-1]
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qcidxs):
                    logger.info('>>>> {}/{} done... '.format(i+1, len(self.qcidxs)))

            logger.info('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            loader = torch.utils.data.DataLoader(
                ObjectsFromList(root=self.ims_root, 
                                obj_names=[self.objects[i] for i in idxs2objects], 
                                obj_bbox=[self.bbox[i] for i in idxs2objects],
                                transform=self.transform),
                                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            
            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim']-1, len(idxs2objects)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()[:-1]
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2objects):
                    logger.info('>>>> {}/{} done... '.format(i+1, len(idxs2objects)))

            logger.info('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            
            if self.approx_similarity:
                scores = torch.mm(poolvecs.t(), qvecs)
                scores, ranks = torch.sort(scores, dim=0, descending=True)
            else:
                scores = torch.cdist(poolvecs.t(),qvecs.t(),2)
                scores, ranks = torch.sort(scores, dim=0, descending=False)
            
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.ncidxs = []
            for q in range(len(self.qcidxs)):
                # do not use query class,
                # those images are potentially positive
                qclass = self.classes[self.qcidxs[q]]
                classes = [qclass]*num_classes_per_neg
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2objects[ranks[r, q]]
                    num_class = []
                    [num_class.append(self.classes[potential]==classes[i]) for i in range(len(classes))]
                    if sum(num_class) < num_classes_per_neg:
                        nidxs.append(potential.item())
                        classes.append(self.classes[potential])
                        avg_ndist += (torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2)
                                           .sum(dim=0).sqrt())
                        n_ndist += 1
                    r += 1
                self.ncidxs.append(nidxs)
            logger.info('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            logger.info('>>>> Done\n')

            # Add previous and current tuples
            self.qidxs = self.qcidxs.copy()
            self.pidxs = self.pcidxs.copy()
            self.nidxs = self.ncidxs.copy()
            if (self.keep_prev_tuples):
                self.qidxs.extend(self.qpidxs)
                self.pidxs.extend(self.ppidxs)
                self.nidxs.extend(self.npidxs)

        # return average negative l2-distance, query vectors, variances, class (for plotting)
        return (avg_ndist/n_ndist).item(), qvecs, qvars, classes_list 




