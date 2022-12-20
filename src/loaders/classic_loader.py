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

        if not (mode == 'train' or mode == 'val' or mode == 'test' or mode == 'trainval'):
            raise(RuntimeError("MODE should be either train, val, test, trainval"))

        # Define attributes
        self.mode = mode
        self.loader = image_object_loader
        self.classes_not_trained_on = classes_not_trained_on
        self.transform = transform
        self.print_freq = 500
        self.require_grad = True
        if (self.mode == 'val') | (self.mode == 'test'):
            self.require_grad = False
        
        # Select subset of classes
        class_list = ['train','cow','tvmonitor','boat','cat','person','aeroplane','bird',
                      'dog','sheep','bicycle','bus','motorbike','bottle','chair',
                      'diningtable','pottedplant','sofa','horse','car']
        self.classes_trained_on = ([class_ for class_ in class_list if class_ 
                                    not in self.classes_not_trained_on]) 
        
        
        # Extract data
        self.data_root = './data/processed/'
        self.ims_root = './data/raw/JPEGImages/'
        (self.objects, 
         self.bbox, 
         self.classes, 
         self.class_hash) = self._extract_dataset_()   

        # size of pool when training
        self.poolsize_class = poolsize_class
        self.poolsize = self.poolsize_class*len(self.classes_trained_on)
        
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
        target = self.classes_trained_on.index(class_)

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


    def _extract_dataset_(self):
        dataset = self.mode
        # Extract database
        if dataset != 'trainval':
            db_root = os.path.join(self.data_root, f'{dataset}.json')
            f = open(db_root)
            db = json.load(f)
            f.close()  
            
        else:  
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
        
        #Extract objects from database
        if dataset == 'test':
            # Extract list of images
            objects_list = [[db[j]['img_name']]*len(db[j]['classes']) for j in range(len(db))]
            objects = []
            [objects.extend(obj) for obj in objects_list]
            
            # Extract list of bbox
            bbox_list = [db[j]['bbox'] for j in range(len(db))]
            bboxs = []
            [bboxs.extend(bbox) for bbox in bbox_list]
            
            # Extract list of classes
            class_num = [db[j]['classes'] for j in range(len(db))]
            classes = []
            [classes.extend(class_list) for class_list in class_num]
            
            # Sort on classes
            class_sort_idxs = np.argsort(classes)
            objects = [objects[i] for i in class_sort_idxs]
            bboxs = [bboxs[i] for i in class_sort_idxs]
            classes = [classes[i] for i in class_sort_idxs]
        
        else:
            # Extract list of classes
            class_num = [[key]*len(db[key]) for key in db.keys()]
            classes = []
            [classes.extend(class_list) for class_list in class_num]

            # Extract list of images
            obj_names = [[db[key][j]['img_name'] for j in range(len(db[key]))] for key in db.keys()]
            objects = []
            [objects.extend(image_list) for image_list in obj_names]

            # Extract list of bbox
            bbox_list = [[db[key][j]['bbox'] for j in range(len(db[key]))] for key in db.keys()]
            bboxs = []
            [bboxs.extend(bbox) for bbox in bbox_list]
            
            
        #Remove objects not trained on
        classes_keep_mask = [i for i in range(len(classes)) 
                                if classes[i] in self.classes_trained_on]
        objects = [objects[i] for i in classes_keep_mask]
        bboxs = [bboxs[i] for i in classes_keep_mask]
        classes = [classes[i] for i in classes_keep_mask]
        
        
        # Get index hash for each class
        class_hash = {}
        count = 0
        keys, vals = np.unique(classes, return_counts=True)
        for i, key in enumerate(keys):
            class_hash[key] = (count,count+vals[i])
            count += vals[i]
            
            
        return objects, bboxs, classes, class_hash


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

