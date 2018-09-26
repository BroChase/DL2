# Chase Brown
# SID 106015389
# DeepLearning PA 2: CNN using Keras

# data_process: Contains module class for loading data from zip files.
from zipfile import ZipFile
import numpy as np


class LoadDataModule(object):
    def __init__(self):
        self.DIR = './'
        pass

    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load(self, mode):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels
