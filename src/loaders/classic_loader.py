# Standard
import os
import pdb
import json
import logging
import numpy as np
import multiprocessing

# Torch
import torch
import torch.utils.data as data
from torchvision import transforms

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader
from src.models.image_classification_network import ImageClassifierNet_Classic

# Init logger
logger = logging.getLogger('__main__')


class Pooling_Dataset(data.Dataset):
    """Dataeset that loads training and validation images for the classic softmax classifer model
    
    Args:
        mode (string): 'train' or 'val' for training and validation parts of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        poolsize_class (int, Default:200): Number of images per class in pool
        classes_not_trained_on (list, Default:[]): What classes should we not train on
     
     Attributes:
        update_backbone_repr_pool (func): Update backbone representation of training images
    """

    def __init__(self, mode: str, poolsize_class:int = 200, transform: transforms=None,
                 classes_not_trained_on: list=[]):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        # Define attributes
        self.mode = mode
        self.loader = image_object_loader
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
        self.classes_trained_on = ([class_ for class_ in class_list if class_ 
                                    not in self.classes_not_trained_on]) 
        # Extract list of classes
        class_num = [[key]*len(db[key]) for key in db.keys()]
        self.classes = []
        [self.classes.extend(class_list) for class_list in class_num]

        # Extract list of images
        obj_names = [[db[key][j]['img_name'] for j in range(len(db[key]))] for key in db.keys()]
        self.objects = []
        [self.objects.extend(image_list) for image_list in obj_names]

        # Extract list of bbox
        bbox_list = [[db[key][j]['bbox'] for j in range(len(db[key]))] for key in db.keys()]
        self.bbox = []
        [self.bbox.extend(bbox) for bbox in bbox_list]
        
        # Get index hash for each class
        self.class_hash = {}
        count = 0
        for key in db.keys():
            self.class_hash[key] = (count,count+len(db[key]))
            count += len(db[key])

        # size of pool when training
        self.poolsize_class = poolsize_class
        self.poolsize = self.poolsize_class*len(db.keys())
        
        # Init tensors for storing backbone output
        self.backbone_repr = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object (backbone representation), target (one-hot-encoding tensor)
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` "+
                               "method to create subset for train/val!"))


        # query object
        output = self.backbone_repr[index]
        
        # Target
        class_ = self.pool_classes_list[index]
        class_idx = self.classes_trained_on.index(class_)

        target = torch.zeros((len(self.classes_trained_on)))
        target[class_idx] = 1

        return output, target, class_


    def __len__(self):
        if self.backbone_repr is not None:
            return self.backbone_repr.shape[0]
        else:
            return 0

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Mode: {} {}\n'.format(self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.objects))
        fmt_str += '    Poolsize: {}\n'.format(self.poolsize)
        fmt_str += '    Number of images from each class in pool {}\n'.format(self.poolsize_class)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + 
                                     ' ' * len(tmp)))
        return fmt_str



    def update_backbone_repr_pool(self, net: ImageClassifierNet_Classic):
        logger.info('\n\n §§§§ Updating pool for *{}* dataset... §§§§\n'.format(self.mode))
        
        #prepare net
        net.cuda()
        if self.mode == 'train':
            net.eval() #net.train() 
        else:
            net.eval()
        
        self.backbone_repr = []
        
        # draw pool_size random images from each class (constitues balanced pool)
        idxs2pool = {class_:torch.IntTensor() for class_ in self.classes_trained_on}
        pool2class = {class_:torch.IntTensor() for class_ in self.classes_trained_on}
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
                            batch_size=1, shuffle=False, 
                            num_workers=np.min([8,multiprocessing.cpu_count()]), 
                            pin_memory=False
        )

        # extract backbone feature embeddings
        for i, input in enumerate(loader):
            if self.require_grad & (net.fixed_backbone == False):
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

