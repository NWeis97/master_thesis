# Imports
import os
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from PIL.Image import Image as IMG
from torch import Tensor
from typing import Union
import PIL

class ObjectsFromList(data.Dataset):
    """A generic data loader that loads images (objects) from a list 
           (Based on 'cnnimageretrieval-pytorch/cirtorch/datasets/genericdataset.py' by 
            'filipradenovic')

    Args:
        root (str): Root directory path to images
        obj_names (list[str]): List of image filenames
        obj_bbox (list[list]]): List of bounding boxes (list)
        transform (transforms): A function/transform that takes in an PIL image
                                and returns a transformed version. Should be resized to n*n 
                                image corresponding to the necesarry model input.
    """

    def __init__(self, root: str, obj_names: list[str], obj_bbox: list[list], 
                       transform: transforms):
        
        objects_fn = [os.path.join(root,obj_names[i]) for i in range(len(obj_names))]
        objects_bbox = [obj_bbox[i] for i in range(len(obj_names))]

        if len(objects_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))

        self.root = root
        self.objects = obj_names
        self.objects_fn = objects_fn
        self.bbox = objects_bbox
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            image (PIL): Loaded image
        """

        # Load image, crop to bbox, and transform
        path = self.objects_fn[index]
        img = Image.open(path)
        img = img.crop(self.bbox[index])
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.objects_fn)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of objects: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + 
                                                                            ' ' * len(tmp)))
        return fmt_str


def image_object_loader(path: str, bbox: list, transformer: transforms) -> Union[IMG,Tensor]:
    """ This function is a generic PIL image loader of an object on an image with path 'path' and 
        bounding box 'bbox'. After loading the image is loaded it will undergo transformation given 
        by transformer arg.
        
    Args:
        path (str): Path to image
        bbox (list): Bounding box (given as list)
        transformer (transforms): Transformer object that transforms pil image into Tensor or Image
    
    Returns:
        image (Image or Tensor): Return transformed image. Either PIL image or torch.Tensor 
                                 depending on the transformer arg.
    """

    img = Image.open(path)
    img = img.crop(bbox)
    img = transformer(img)

    return img

