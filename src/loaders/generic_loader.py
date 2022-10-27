import os
import pdb

import torch
import torch.utils.data as data
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageEnhance



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


def image_object_full_bbox_resize_class(path,bbox,img_size,t_class,p_class,probs,var=None):
    """
    This function is a generic PIL image loader of an object on
    an image with path 'path' and bounding box 'bbox'.
    The function returns the origin PIL image with bbox annotation,
    the bbox image, and the bbox image resized.
    The three images will be put side-by-side
    """
    
    img = Image.open(path).convert("RGB")
    img_bbox = img.crop(bbox)
    img_resize = img_bbox.resize((img_size,img_size))
    
    img_full = ImageDraw.Draw(img)
    img_full.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='Red', width=3)
    img_full = img_full._image  # type: ignore
    
    images = [img_full,img_bbox,img_resize]
    widths, heights = zip(*(i.size for i in images))

    total_width = int(sum(widths)*1.1)
    max_height = max(heights)
    
    probs_space = 0
    if probs is not None:
        probs_space = 200
        

    concat_img = Image.new('RGB', (max(total_width+20,1400)+probs_space, max_height+70))

    x_offset = 10 + max(int((1400-total_width-20)/2),0) + probs_space
    y_offset = 70
    x_space_between = int(np.floor((total_width-sum(widths))/2))
    for im in images:
        concat_img.paste(im, (x_offset,y_offset))
        x_offset += im.size[0]+x_space_between


    # Custom font style and font size
    myFont = ImageFont.truetype('FreeMono.ttf', 40)
    # Add Text to an image
    concat_img = ImageDraw.Draw(concat_img)
    concat_img.text((10, 10), f"True class: {t_class}, Pred class: {p_class}, variance: {var:.4f}",
                    font=myFont, fill =(255, 255, 255))
    
    # add probs if parsed
    if probs is not None:
        myFont2 = ImageFont.truetype('FreeMono.ttf', 20)
        probs_top20 = probs.sort_values()[::-1][:20]
        probs_top20_idx = []
        for i in range(len(probs_top20)):
            if probs_top20.index[i] == t_class:
                probs_top20_idx.append('*'+probs_top20.index[i]+'*')
            else:
                probs_top20_idx.append(probs_top20.index[i])
        probs_top20.index = probs_top20_idx
        concat_img.text((10, 80), f"{probs_top20}", font=myFont2, fill =(255, 255, 255))
    concat_img = concat_img._image #type: ignore

    return concat_img