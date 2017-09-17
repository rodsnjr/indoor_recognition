"""
    Provides a interface to access, download, process, convert, and load the SUN Database/Dataset:

    https://groups.csail.mit.edu/vision/SUN/

    Using Slim Dataset Provider
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py

    Sample:
    https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py
    
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
import pandas as pd

slim = tf.contrib.slim

class PascalParser:
    """
        Sun Dataset is available in Pascal Format
        So this class should be able to parse the files, and create the correct 
        definitions of images_path -> labels for image recognition networks
    """
    pass

class TFRecord_Converter:
    """
        Based on: 
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
    """
    pass

class GVC_Dataset:
    _FILE_PATTERN = 'sun_%s.tfrecord'
    
    _SPLITS_TO_SIZES = {'train': 60000, 'test': 10000}

    _NUM_CLASSES = 10

    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [28 x 28 x 1] grayscale image.',
        'label': 'A single integer between 0 and 9',
    }

    def __init__(self):
        pass
    
    def check_available(self):
        pass
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def convert_to_tf(self):
        pass
    
    def get_slim(self):
        pass
    
    def save(self):
        pass
