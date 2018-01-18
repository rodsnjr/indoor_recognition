from enum import Enum
import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import csv
from sklearn.preprocessing import LabelBinarizer
from keras.utils import Sequence
import math

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dumps')

class Directory(Enum):
    train=("Treinamento", "/media/rodsnjr/Files/Datasets/gvc_dataset_final/train/")
    validation=("Validação", "/media/rodsnjr/Files/Datasets/gvc_dataset_final/validation/")
    test=("Testes", "/media/rodsnjr/Files/Datasets/gvc_dataset_final/testing/")

    def __init__(self, title, path):
        self.title = title
        self.dir = path
        self.classes = self.load_dir()

    def load_dir(self):
        contents = os.listdir(self.dir)
        classes = [each for each in contents if os.path.isdir(self.dir + each)]
        
        return classes

    def show_statistics(self):
        print(self.title)
        for each in self.classes:
            class_path = self.dir + each
            files = os.listdir(class_path)
            print("%s files in %s" % (len(files), each))
        print('\n')

    def __get_images(self):
        x = []
        for each in self.classes:
            class_path = self.dir + each
            files = os.listdir(class_path)
            x.extend([os.path.join(self.dir, each, filename) 
                            for filename in files])
        return x

    def __get_classes(self):
        y = []
        for each in self.classes:
            class_path = self.dir + each
            files = os.listdir(class_path)
            y.extend([each for _ in range(len(files))])
        return y

    def build_sequence(self, batch_size, image_size=(224, 224), 
                        y_processing='one-hot'):
        
        x = self.__get_images()
        y = self.__get_classes()
        
        if y_processing=='one-hot':
            y, _ = get_data_labels(y)

        sequence = GVCSequence(x, y, batch_size, image_size)
        return sequence

class GVCSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, image_size):
        """
            Here, `x_set` is list of path to the images
            and `y_set` are the associated classes.
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.array([load_image(file_name, self.image_size) 
                    for file_name in batch_x])

        return x, np.array(batch_y)

def load_image(path, size=(224, 224)):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, size, mode='constant')
    return resized_img

def save_to_file(file, codes, labels):
    # write codes to file
    #with open(os.path.join(DIR, 'codes_%s' % file), 'w') as f:
    # codes.tofile(f)
    print("Saving Shape -> ", codes.shape)
    np.savez(os.path.join(DIR, 'codes_%s' % file), codes)

    # write labels to file
    with open(os.path.join(DIR, 'labels_%s' % file), 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)

# read codes and labels from file
def read_file(file):
    
    reshape_array = False
    codes_filename = os.path.join(DIR, 'codes_%s' % file)
    
    if os.path.exists(codes_filename + '.npy'):
        codes_filename += '.npy'    
    else:
        codes_filename += '.npz'
        reshape_array = True

    with open(os.path.join(DIR, 'labels_%s' % file)) as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    
    
    codes = np.load(codes_filename)
    if reshape_array:
        codes = np.array(codes['arr_0'])

    return codes, labels

def get_data_labels(labels):
    lb = LabelBinarizer()
    lb.fit(labels)
    labels_vecs_train = lb.transform(labels)
    
    return labels_vecs_train, lb
