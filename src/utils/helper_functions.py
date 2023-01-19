# Imports
import shortuuid
import logging
import os
import configparser
import warnings
import cv2

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]], [batch[0][2]]
    return ([batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))],
            [batch[i][2] for i in range(len(batch))])
    
    
def check_config(config):
    """This function checks if the specific config being run at call-time has been run before.
       If so, return the name of said config file, otherwise generate new filename and save
       config to version history.

    Args:
        config (config): Config file of type configParser

    Returns:
        config_filename (str): name of either existing config file or new config file.
        config_run_no (int): number of times this config has been run
    """
    # get logger
    logger = logging.getLogger('__main__')
    
    # assign directory
    directory = './configs/configs_hist'
    
    # bool for checking if config already has been run
    config_exists = False
    
    # iterate over configs in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            config_hist =  configparser.ConfigParser()
            config_hist.read(f)
            break_outer = False
            
            for key1 in config:
                for key2 in config[key1]:
                    try:
                        if config[key1][key2] != config_hist[key1][key2]:
                            break_outer = True
                            break;
                    except:
                        break_outer = True
                        break;
                
                if break_outer:
                    break;

            if break_outer == False:
                config_exists = True
                break;
     
    if config_exists:
        
        config_filename = filename[:-4]
        
        # Define config run number
        config_run_no = 0
        while True:
            if os.path.exists(f'./logs/training_test/train_model/{str(config_filename)}'+
                            f'/run_{str(config_run_no)}.log'):
                config_run_no += 1
            else:
                break;
    else:
        config_filename = str(shortuuid.uuid())
        os.makedirs(f'./logs/training_test/train_model/{str(config_filename)}',exist_ok=True) 
        with open(f'configs/configs_hist/{config_filename}.ini', 'w') as configfile:
            config.write(configfile)  
        config_run_no = 0
    
    return config_filename, config_run_no
    

def get_logger_old(config_filename: str, config_run_no: str):
    """This function defines and returns logger that save output to file in path:
       './logs/training_test/train_model/{config_filename}/run_{config_run_no}.logs'

    Args:
        mode (str): training or testing
        config_filename (str): config filename
        config_run_no (str): config run number

    Returns:
       logger (Logger): A logger for logging
    """
    # Define logger
    log_fmt = '%(message)s'
    log_file_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger() 
    file_handler = logging.FileHandler(f'./logs/training_test/train_model/{str(config_filename)}/'+
                                       f'run_{str(config_run_no)}.log')
    file_handler.setFormatter(logging.Formatter(log_file_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    
    return logger

def get_logger(name: str):
    """This function defines and returns logger that save output to file in path:
       './logs/training_test/train_model/{name}.logs'

    Args:
        name (str): name of test

    Returns:
       logger (Logger): A logger for logging
    """
    # Define logger
    log_fmt = '%(message)s'
    log_file_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger() 
    file_handler = logging.FileHandler(f'./logs/training_test/train_model/{name}.log')
    file_handler.setFormatter(logging.Formatter(log_file_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    
    return logger


def get_logger_test(name: str):
    """This function defines and returns logger that save output to file in path:
       './logs/training_test/test_model/{name}.logs'

    Args:
        name (str): name of test

    Returns:
       logger (Logger): A logger for logging
    """
    # Define logger
    log_fmt = '%(message)s'
    log_file_fmt = '%(asctime)s - %(name)s - %(levelname)s:\n\t%(message)s'
    logging.basicConfig(filemode='a',
                        format=log_fmt,
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger() 
    file_handler = logging.FileHandler(f'./logs/training_test/test_model/{str(name)}.log')
    file_handler.setFormatter(logging.Formatter(log_file_fmt))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    
    return logger


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()