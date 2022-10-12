# Standard
import os
import pickle
import pdb
import json
from PIL import Image
import logging

# Torch
import torch
import torch.utils.data as data
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader

import numpy as np
from PIL import Image

# Init logger
logger = logging.getLogger('__main__')


class TuplesDataset(data.Dataset):
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
     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same 
                      position in qpool
        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same 
                      position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the 
                        same position in qidxs
        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, mode, nnum=10, qsize_class=20, poolsize=3000, transform=None):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        # Define mode and loader
        self.mode = mode
        self.loader = image_object_loader

        # Set paths to data
        data_root = './data/processed/'
        db_root = os.path.join(data_root, self.mode+'.json')
        self.ims_root = './data/raw/JPEGImages/'

        # Load database
        f = open(db_root)
        db = json.load(f)
        f.close()

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
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        # remember previous hard negatives for a query
        self.prev_nidxs = None

        # Init transformer
        self.transform = transform
        self.print_freq = 1000

        # mean of training images: tensor([0.4108, 0.3773, 0.3510])
        # std of training images: tensor([0.2102, 0.2016, 0.1987])

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
            output.append(self.loader(os.path.join(self.ims_root,self.objects[self.nidxs[index][i]]), 
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

    def create_epoch_tuples(self, net):

        logger.info('>> Creating tuples for an epoch of {}...'.format(self.mode))
        #logger.info(">>>> used network: ")
        #logger.info(net.meta_repr())

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        # draw ranomdly the same amount from each class
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

        self.qidxs = idxs2qpool.tolist()
        self.pidxs = idxs2ppool.tolist()

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        idxs2objects = torch.randperm(len(self.objects))[:self.poolsize]

        # prepare network
        net.cuda()
        net.train()
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            logger.info('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ObjectsFromList(root=self.ims_root, obj_names=[self.objects[i] for i in self.qidxs], 
                                obj_bbox=[self.bbox[i] for i in self.qidxs],
                                transform=self.transform),
                                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            
            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim']-1, len(self.qidxs)).cuda()
            qvars = torch.zeros(len(self.qidxs),)
            for i, input in enumerate(loader):
                output = net(input.cuda()).data.squeeze()
                qvecs[:, i] = output[:-1]
                qvars[i] = output[-1]
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                    logger.info('\r>>>> {}/{} done... '.format(i+1, len(self.qidxs)))
            logger.info('')

            logger.info('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            loader = torch.utils.data.DataLoader(
                ObjectsFromList(root=self.ims_root, obj_names=[self.objects[i] for i in idxs2objects], 
                                obj_bbox=[self.bbox[i] for i in idxs2objects],
                                transform=self.transform),
                                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            
            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim']-1, len(idxs2objects)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()[:-1]
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2objects):
                    logger.info('\r>>>> {}/{} done... '.format(i+1, len(idxs2objects)))
            logger.info('')

            logger.info('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query class,
                # those images are potentially positive
                qclass = self.classes[self.qidxs[q]]
                classes = [qclass]
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2objects[ranks[r, q]]
                    # take at most one image from the same cluster
                    if not self.classes[potential] in classes:
                        nidxs.append(potential.item())
                        classes.append(self.classes[potential])
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
            logger.info('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            logger.info('>>>> Done')

        return (avg_ndist/n_ndist).item(), qvecs, qvars, classes_list # return average negative l2-distance




