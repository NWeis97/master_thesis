# Standard
import os
import pdb
import json
import logging
import numpy as np
import multiprocessing

# Torch
import torch
from torch import Tensor
import torch.utils.data as data
from torchvision import transforms
from typing import Tuple

# Load own libs
from src.loaders.generic_loader import ObjectsFromList, image_object_loader
from src.models.image_classification_network import ImageClassifierNet_BayesianTripletLoss

# Init logger
logger = logging.getLogger('__main__')

# Inspired on 'cnnimageretrieval-pytorch/cirtorch/datasets/traindataset.py' by 'filipradenovic'
class TuplesDataset(data.Dataset):
    """Dataset that loads training and validation images for the BTL model

    Parameters
    ----------
    ``mode`` : str
        Specify 'train' or 'val' for selecting the training or validation dataset.
    ``nnum`` : int, optional
        Number of negatives mined using the hard-negative mining procedure, by default 10
    ``qsize_class`` : int, optional
        Number of tuples for each class (based on anchor), by default 20
    ``npoolsize`` : int, optional
        Number of random samples from sample pool used in the negative pool, by default 2000
    ``poolsize_class`` : int, optional
        _description_, by default 200
    ``transform`` : transforms, optional
        Transforms object for transforming PIL images to torch.Tensor (with other tranforms), 
        by default None
    ``classes_not_trained_on`` : list, optional
        What classes should not be trained on, by default []
    ``approx_similarity`` : bool, optional
        Should the approximative similarity be used to extract hard-negatives?, by default True
        
    Attributes
    ----------
    ``update_backbone_repr_pool`` : func 
        Update backbone representation of training images
    ``create_epoch_tuples`` : func
        Create a batch of tuples (used in __getitem__)
    """

    def __init__(self, mode: str, nnum: int=10, qsize_class:int = 20, npoolsize: int = 2000,
                 poolsize_class:int = 200, transform: transforms=None,
                 classes_not_trained_on: list=[], approx_similarity: bool=True):
        
        if not (mode == 'train' or mode == 'val' or mode == 'test' or mode == 'trainval'):
            raise(RuntimeError("MODE should be either train, val, test, trainval"))

        # Define attributes
        self.mode = mode
        self.loader = image_object_loader
        self.approx_similarity = approx_similarity
        self.classes_not_trained_on = classes_not_trained_on
        self.print_freq = 500
        self.require_grad = True
        if (self.mode == 'val') | (self.mode == 'test'):
            self.require_grad = False
        if transforms == None:
            self.transform = transforms.Compose([transforms.PILToTensor(),
                                                 transforms.ConvertImageDtype(torch.float)])
        else:
            self.transform = transform
        
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
        

        # size of training subset for an epoch
        self.nnum = nnum
        self.poolsize_class = poolsize_class
        self.poolsize = np.min([self.poolsize_class*len(self.classes_trained_on),len(self.objects)])
        self.qsize_class = qsize_class
        self.npoolsize = np.min([npoolsize,self.poolsize])
        self.average_class_size = self.poolsize/len(self.classes_trained_on)
        logger.info(f"The average # samples per class is {self.average_class_size:.0f}")

        # Init idxs list
        self.qidxs = [] # Query idxs (from pool)
        self.pidxs = [] # Positives idxs (from pool)
        self.nidxs = [] # Negatives idxs (from pool)

        # Init tensors for storing backbone output
        self.backbone_repr = None

    def __getitem__(self, index: int) -> Tuple[list[Tensor], Tensor, list[str]]:
        """Get item

        Parameters
        ----------
        ``index`` : int
            Index

        Returns
        -------
        ``Tuple[list[Tensor], Tensor, list[str]]``
            Tuple containing input and target for training
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

        # Create target
        target = torch.Tensor([-1, 1] + [0]*self.nnum)
        
        # Create list of classes for each tuple elemt
        classes = [self.pool_classes_list[self.qidxs[index]], self.pool_classes_list[self.pidxs[index]]]
        ([classes.append(self.pool_classes_list[self.nidxs[index][i]]) for i in 
                                                            range(len(self.nidxs[index]))])

        return output, target, classes


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


    def update_backbone_repr_pool(self, net: ImageClassifierNet_BayesianTripletLoss) -> None:
        """Updates pool of backbone representationd of images, which are used in 
        ``create_epoch_tuples()``

        Parameters
        ----------
        ``net`` : ImageClassifierNet_BayesianTripletLoss
            Network for extracting backbone reprensentations

        Raises
        ------
        ``MemoryError``
            If GPU does not have enough memory to store all backbone representations.
        """
        
        logger.info('\n\n §§§§ Updating pool for *{}* dataset... §§§§\n'.format(self.mode))
        
        #prepare net
        net.cuda()
        if self.mode == 'train':
            net.eval() #net.train() 
        else:
            net.eval()
        
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
        self.pool_idx_to_db_idx = {indx:i for indx,i in enumerate(idxs2pool_list)}
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
            # Init backpone_repr
            if self.require_grad & (net.fixed_backbone == False):
                o = net.forward_backbone(input.cuda())
            else:
                with torch.no_grad():
                    o = net.forward_backbone(input.cuda())
            
            if i == 0: #init backbone representation tensor
                try:
                    self.backbone_repr = torch.empty(len(loader),o.shape[1],o.shape[2],o.shape[3]).cuda()
                    mem_cuda = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                    logger.info(f"Total memory consumption after storing '{self.mode}' data "+
                                f"backbone representations on "+
                                f"Cuda: {mem_cuda:.3f}GB")
                except:
                    logger.error(f"Memory needed for storing pool of backbone "+
                                f"representations exceeds available memory. Consider lower the"+
                                f" the number of class samples in pool.")
                    raise MemoryError(f"Memory needed for storing pool of backbone "+
                                f"representations exceeds available memory. Consider lower the"+
                                f" the number of class samples in pool.")
            self.backbone_repr[i] = o[0]
            
            if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2pool_list):
                logger.info('>>>> {}/{} done... '.format(i+1, len(idxs2pool_list)))
        
        self.pool_classes_list = []
        ([self.pool_classes_list.extend([i]*len(self.pool_idx_to_class[i])) for 
                                            i in self.pool_idx_to_class.keys()])
        
        # Extract pool hash
        self.pool_class_hash = {}
        count = 0
        for key in self.pool_idx_to_class.keys():
            self.pool_class_hash[key] = (count,count+len(self.pool_idx_to_class[key]))
            count += len(self.pool_idx_to_class[key])
        

    def create_epoch_tuples(self, net: ImageClassifierNet_BayesianTripletLoss, 
                                  num_classes_per_neg: int,
                                  skip_closeset_neg_prob: float = 0.0) -> Tuple[float,
                                                                                 Tensor,
                                                                                 Tensor,
                                                                                 list]:
        """Creates a batch of tuples, which can then be extracted using the __getitem__() function.

        Parameters
        ----------
        ``net`` : ImageClassifierNet_BayesianTripletLoss
            Network for extracting embeddings.
        ``num_classes_per_neg`` : int
            Number of negatives per class per tuple.
        ``skip_closeset_neg_prob`` : float, optional
            Skip hard negative probability, by default 0.0

        Returns
        -------
        ``Tuple[float, Tensor, Tensor, list]``
            (Average distance to hard-negative,
            Means of queries (anchors), 
            Variances of queries (anchors),
            List of classes of queries (anchors))

        Raises
        ------
        ``RuntimeError``
            If ``skip_closeset_neg_prob``is too high, it will sometimes take too long to construct 
            tuples. After a certain amount of re-tries, the algorithm terminates if tuples has not 
            been created.
        """
        
        logger.info('\n>> Creating tuples for *{}* dataset...'.format(self.mode))

        # prepare network
        net.cuda()
        if self.mode == 'train':
            net.train()
        else:
            net.eval()
    
        # **********************************
        # **** SELECTING POSITIVE PAIRS ****
        # **********************************
        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
        idxs2qpool = torch.IntTensor()
        idxs2ppool = torch.IntTensor()
        classes_list = {key:[] for key in self.pool_idx_to_class.keys()}

        count = 0
        for key in self.pool_class_hash.keys():
            min = self.pool_class_hash[key][0]
            max = self.pool_class_hash[key][1]
            pool = min+torch.randperm(max-min)
            max_len = int(np.min([self.qsize_class*2,len(self.pool_idx_to_class[key])])/2)
            idxs2qpool = torch.cat((idxs2qpool,pool[:max_len]))
            idxs2ppool = torch.cat((idxs2ppool,pool[max_len:max_len*2]))
               
            classes_list[key]= np.arange(count, count+max_len)
            count += max_len

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
        
        # **************************************************
        # ***** Query and Positive images extraction *******
        # **************************************************
        # extract query vectors
        logger.info('>> Extracing pool images...')
        q_means = torch.zeros(len(self.qidxs),net.mean_dim).cuda()
        q_vars = torch.zeros(len(self.qidxs),net.var_dim).cuda()
        q_rnd = torch.randperm(len(self.qidxs))
        
        for i in range(int(np.ceil(len(q_rnd)/80))):
            with torch.no_grad():
                q_rnd_idx = q_rnd[i*80:(i+1)*80]
                qidxs_rand = np.array(self.qidxs)[q_rnd_idx]
                output = net.forward_head(self.backbone_repr[qidxs_rand])
                q_means[q_rnd_idx,:] = output[:,:-net.var_dim]
                q_vars[q_rnd_idx,:] = output[:,-net.var_dim:]
                
                if (i+1) % self.print_freq == 0 or (i+1) == int(np.ceil(len(q_rnd)/80)):
                    logger.info('>>>> {}/{} done... '.format(i+1, int(np.ceil(len(q_rnd)/80))))


        # ****************************************
        # ***** Negative images extraction *******
        # ****************************************   
        # Select npoolsize random images from pool
        pool_means = torch.zeros(self.npoolsize,net.mean_dim).cuda()
        rnd_idxs = torch.randperm(self.backbone_repr.shape[0])[:self.npoolsize]
        for i in range(int(np.ceil(len(rnd_idxs)/80))):
            with torch.no_grad():
                output = net.forward_head(self.backbone_repr[rnd_idxs[i*80:(i+1)*80]])
            pool_means[i*80:(i+1)*80,:] = output[:,:-net.var_dim]       
            
            if (i+1) % self.print_freq == 0 or (i+1) == int(np.ceil(len(rnd_idxs)/80)):
                logger.info('>>>> {}/{} done... '.format(i+1, int(np.ceil(len(rnd_idxs)/80)))) 
        
        
        logger.info('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        if self.approx_similarity:
            scores = torch.mm(pool_means,q_means.T)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
        else:
            scores = torch.cdist(pool_means,q_means,2)
            scores, ranks = torch.sort(scores, dim=0, descending=False)
        
        avg_ndist = torch.tensor(0).float().cuda()  # for statistics
        n_ndist = torch.tensor(0).float().cuda()  # for statistics
       
        # ****************************************
        # ******** Hard Negative Mining **********
        # **************************************** 
        # selection of negative examples
        self.nidxs = []
        for q in range(len(self.qidxs)):
            # do not use query class,
            # those images are potentially positive
            qclass = self.pool_classes_list[self.qidxs[q]]
            num_class = {class_:0 for class_ in self.pool_idx_to_class.keys()}
            num_class[qclass] += num_classes_per_neg
            nidxs = []
            r = 0
            count = 0
            limit_reset = 0
            while len(nidxs) < self.nnum:
                potential = ranks[r, q]
                class_potential = self.pool_classes_list[rnd_idxs[potential].item()]
                if num_class[class_potential] < num_classes_per_neg:
                    if (torch.rand(1).item() > skip_closeset_neg_prob):
                        nidxs.append(rnd_idxs[potential])
                        num_class[class_potential] += 1
                        avg_ndist += (torch.pow(q_means[q,:]-pool_means[potential,:]+1e-6, 2)
                                            .sum(dim=0).sqrt())
                        n_ndist += 1
                r += 1
                if r == ranks.shape[0]: #Reset if by chance nnum negatives not chosen yet
                    r = 0
                    limit_reset += 1
                    if limit_reset == 10:
                        raise RuntimeError("Tuple creation reset search now done 10 times..."+
                                           " You possible need to lower the skip_closest_neg_prob")
            count += 1
            self.nidxs.append(nidxs)
            
            if q % 500 == 0:
                logger.info(f">>>>>> {q}/{len(self.qidxs)}")
     
        logger.info('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
        logger.info('>>>> Done\n')
        
                    
        # return average negative l2-distance, query vectors, variances, class (for plotting)
        return (avg_ndist/n_ndist).item(), q_means.detach(), q_vars.detach(), classes_list




