"""
    Provides a interface to access, download, process, convert, and load the GVC Indoor Dataset

    https://github.com/rodsnjr/
    
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

from . import DIR, tfr_present
from . import file_helpers

slim = tf.contrib.slim

class TFRecord_Converter:
    """
        Based on: 
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
    """
    pass

class GVC_Dataset:
    _FILE_PATTERN = 'gvc_%s.tfrecord'
    
    _SPLITS_TO_SIZES = {'train': 60000, 'test': 10000}

    _NUM_CLASSES = 3

    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [28 x 28 x 1] grayscale image.',
        'label': 'A single integer between 0 and 9',
    }

    _DESCRIPTION = "GVC Indoor Dataset"

    _DATASET_URL = 'https://github.com/rodsnjr/gvc_dataset.git'

    _DATASET_DIR = os.path.join(DIR, 'gvc_dataset')

    def __init__(self):
        pass
    
    def check_available(self):
        """ 
            Check if the dataset is in its directory
            and download/create it if not
        """
        if not os.path.exists(self._DATASET_DIR):
            os.makedirs(self._DATASET_DIR)
            self.download()
    
    def download(self):
        file_helpers.downloader(url=self._DATASET_URL, 
            directory=self._DATASET_DIR,
            filename='gvc_dataset.zip',
            desc=self._DESCRIPTION
        )
    
    def extract(self):
        filename = os.path.join(self._DATASET_DIR, 'gvc_dataset.zip')
        print(filename)
        file_helpers.extract_zip(filename, self._DATASET_DIR)
    
    def process(self, size=(224, 224, 3)):
        """
            Proccess the dataset images to the defined size
        """
    
    def convert_to_tf(self):
        pass
    
    def get_slim(self):
        pass
    
    def save(self):
        pass

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.
  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)