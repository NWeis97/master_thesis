# Imports
import os
import torch.utils.data as data
from PIL import Image



class ObjectsFromList(data.Dataset):
    """A generic data loader that loads images (objects) from a list 
        (Based on 'cnnimageretrieval-pytorch/cirtorch/datasets/genericdataset.py' by 'filipradenovic')
    Args:
        root (string): Root directory path.
        objects (list[dict]): List of image/object dicts: contains path, class, bbox, and size
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. Should be resized to n*n image corresponding to
            the necesarry model input.
     Attributes:
        objects_fn (list): List of full object filename
    """

    def __init__(self, root, obj_names, obj_bbox, transform):

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

        # NB: consider randommizing the bbox (to avoid overfitting)
        # Also augmentation 

        return img

    def __len__(self):
        return len(self.objects_fn)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of objects: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def image_object_loader(path,bbox,transformer):
    """
    This function is a generic PIL image loader of an object on
    an image with path 'path' and bounding box 'bbox'.
    After loading the image is loaded it will undergo transformation
    given by transformer arg.
    """

    img = Image.open(path)
    img = img.crop(bbox)
    img = transformer(img)

    return img
