# -*- coding: utf-8 -*-
import logging
from datetime import datetime
import pdb
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# Set seed and train/val/test split ratio
np.random.seed(1234)
train_val_rat = 0.6

def main():
    """ Runs data processing scripts to extract train/val/test sets
        for classifier and segmentation. Output will be a list of 
        image names. The png files will not be moved from ./data/raw.
        The list cointaining the file names of train/val/test sets
        will be saved in ./data/processed.
    """
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("--- NEW RUN ---")
    
    # Find sets of images
    logger.info("Find Main img. not segmentated, vice versa, and the intersection")
    mNs, sNm, ms = find_MainNSeq_SegNMain_MainSeg()
    logger.info("Succeeded")

    # Extract information from images
    logger.info("Extracting objects from images with corresponding bbox")
    mNs_obj = extract_objects(mNs)
    sNm_obj = extract_objects(sNm)
    ms_obj = extract_objects(ms)
    logger.info("Succeeded")

    # Divide into train/val/test
    logger.info("Split samples into train/val/test")
    data_train, data_val, data_test = split_train_val_test(mNs_obj,
                                                           sNm_obj,
                                                           ms_obj,
                                                           train_val_rat)
    logger.info("Succeeded")
    
    # Split image elements into object elements (objects on images)
    logger.info("Break images up into objects")
    data_train_obj, train_classes = break_into_objects(data_train)
    data_val_obj, val_classes = break_into_objects(data_val)
    data_test_obj, test_classes = break_into_objects(data_test)
    logger.info("Succeeded")

    # Log number of samples
    num_train = len(data_train_obj)
    num_val = len(data_val_obj)
    num_test = len(data_test_obj)
    num_total = num_train+num_val+num_test
    logger.info(f"The number of image-samples is:\nTrain (objects): \t{num_train} "+
                f"({num_train/num_total*100:.1f}%)\n"+
                f"Val (objects): \t\t{num_val}  ({num_val/num_total*100:.1f}%)"+
                f"\nTest (objects): \t{num_test}  ({num_test/num_total*100:.1f}%)\n")
    
    # Visualize distribution of classes in each set
    logger.info("Visualize class distributions for each set")
    visualize_class_dist({'train': train_classes,
                          'val': val_classes,
                          'test': test_classes})
    logger.info("Succeeded")

    # Save dicts
    logger.info("Saving dictionaries")
    out_file = open("./data/processed/train.json", "w")
    json.dump(data_train_obj,out_file)
    out_file.close()
    out_file = open("./data/processed/val.json", "w")
    json.dump(data_val_obj,out_file)
    out_file.close()
    out_file = open("./data/processed/test.json", "w")
    json.dump(data_test,out_file)
    out_file.close()
    logger.info("Succeeded")
    #


def visualize_class_dist(classes: dict[dict]):
    """
    This function prints a bar plot of the class distribution for
    each set.

    Input: 
    classes = {'train': classes_train, 'val': classes_val, 'test': classes_test}
    num_objects = {'train': num_train, 'val': num_val, 'test': num_test}
    """
    sorted_keys = sorted(classes['train'].keys())
    df_classes = pd.DataFrame(index=sorted_keys)

    for set_ in classes.keys():
        df_classes = pd.concat([df_classes,pd.Series(classes[set_],name=set_)],axis=1,join='outer')

    # Find per set class distribution 
    df_perc = df_classes.div(df_classes.sum(axis=0), axis=1)

    # Bar plot of distribution (perc)
    ax = df_perc.plot.bar(rot=45)
    ax.set_title('Distribution of classes',fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    ax.figure.savefig('reports/figures/data/class_distribution_perc.png', bbox_inches='tight')

    # Bar plot of distribution
    ax = df_classes.plot.bar(rot=45)
    ax.set_title('Distribution of classes',fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    ax.figure.savefig('reports/figures/data/class_distribution_abs.png', bbox_inches='tight')
    
    # Print distribution to log
    logger.info(f"\n{np.round(df_perc*100,2)}")

    return


def break_into_objects(data_images):
    """
    This function takes a list of dicts on image level
    and breaks it up into a list of dicts on object level.
    The amount of information stays the same¨

    Furthermore it finds the distribution of classes for
    the set
    """
    data_objects = {}
    classes = {}
    for i in range(len(data_images)):
        for j in range(len(data_images[i]['classes'])):
            class_ = data_images[i]['classes'][j]
            try:
                classes[class_] += 1
                data_objects[class_].append({'img_name': data_images[i]['img_name'],
                                     'img_size': data_images[i]['img_size'],
                                     'class': class_,
                                     'bbox': data_images[i]['bbox'][j]})
            except:
                classes[class_]  = 1
                data_objects[class_] = [{'img_name': data_images[i]['img_name'],
                                         'img_size': data_images[i]['img_size'],
                                         'class': class_,
                                         'bbox': data_images[i]['bbox'][j]}]

    data_objects = dict(sorted(data_objects.items()))
    return data_objects, classes


def split_train_val_test(mNs_obj: list[dict], 
                         sNm_obj: list[dict],
                         ms_obj: list[dict],
                         train_val_rat: float): 
    """
    This function splits the images into train_val_test.
    This is done on image level to avoid information leakage.
    Later these images will be split up on object level.

    train_val_rat: ratio between train and validation data

    Note: we have a limited number of segmented images, thus all 
          of these will be used for testing. Hence no 
          trainval_test_rat
    """
    num_total = len(mNs_obj)+len(sNm_obj)+len(ms_obj)
    num_trainval = len(mNs_obj)
    num_test = len(sNm_obj)+len(ms_obj)
    num_train = int(num_trainval*train_val_rat)
    num_val = num_trainval-num_train
    logger.info(f"The number of image-samples is:\nTrain (images): \t{num_train} "+
                f"({num_train/num_total*100:.1f}%)\n"+
                f"Val (images): \t\t{num_val}  ({num_val/num_total*100:.1f}%)"+
                f"\nTest (images): \t\t{num_test}  ({num_test/num_total*100:.1f}%)\n")

    # Get test objects
    data_test = sNm_obj
    data_test.extend(ms_obj)
    train_idx_trainval = np.random.choice(np.arange(0,num_trainval),num_train,replace=False)
    data_train = [mNs_obj[i] for i in train_idx_trainval]
    data_val = [mNs_obj[i] for i in np.arange(0,num_trainval) if i not in train_idx_trainval]

    return data_train, data_val, data_test


def extract_objects(set_: list):
    """
    For each image in a list of images, extract information
    such as:
    - image size
    - objects present on image
    - bbox  for objects
    """
    images_with_classes = []
    for elem in set_:
        with open(f'data/raw/Annotations/{elem}.xml', 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        # Get size of image
        img_width = int(Bs_data.find('width').get_text())
        img_height = int(Bs_data.find('height').get_text())
        img_channels = int(Bs_data.find('depth').get_text())
        img_size = [img_width,img_height,img_channels]

        # Get objects and bbox of objects 
        img_objs = Bs_data.find_all('object')
        obj_type = []
        obj_bbox = []
        for obj in img_objs:
            obj_type.append(obj.find('name').get_text())
            xmin = int(obj.find('xmin').get_text())
            ymin = int(obj.find('ymin').get_text())
            xmax = int(obj.find('xmax').get_text())
            ymax = int(obj.find('ymax').get_text())
            obj_bbox.append([xmin,ymin,xmax,ymax])

        images_with_classes.append({'img_name': f'{elem}.jpg',
                                    'img_size': img_size,
                                    'classes': obj_type,
                                    'bbox': obj_bbox})

    return images_with_classes





def find_MainNSeq_SegNMain_MainSeg():
    """
    Find all the images that are in the main category, but has
    no segmentations attached. Same vice versa. And Union of the
    two:

    returns:
        (set) M\S
        (set) S\M
        (set) S intersection M
    """
    # Read file names of main images
    main_images_list = []
    num_main = 0
    with open("./data/raw/ImageSets/Main/trainval.txt", "r") as f:
        for line in f:
            main_images_list.append(line[:-1])
            num_main += 1

    # Read file names of segmented images
    segm_images_list = []
    num_segm = 0
    with open("./data/raw/ImageSets/Segmentation/trainval.txt", "r") as f:
        for line in f:
            segm_images_list.append(line[:-1])
            num_segm += 1

    # Find images not segmented
    main_not_in_seg = [x for x in main_images_list if x not in segm_images_list]
    # Find images segmented but not in main
    segm_not_in_main = [x for x in segm_images_list if x not in main_images_list]
    # Find images segmented but not in main
    main_in_seg = [x for x in main_images_list if x in segm_images_list]

    # Number of images should not change
    if (len(main_not_in_seg) + 
        len(segm_not_in_main) + 
        len(main_in_seg) != num_main + num_segm - len(main_in_seg)):
            logging.error("Number of images in sets are not coherent!!")
            exit()

    return main_not_in_seg, segm_not_in_main, main_in_seg





if __name__ == '__main__':
        
    # Define logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    file_handler = logging.FileHandler('./logs/data/make_dataset/logs.log')
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').disabled = True

    main()
