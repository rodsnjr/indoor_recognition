import os

import numpy as np
import tensorflow as tf

from tensorflow_vgg import utils

from enum import Enum

from urllib.request import urlretrieve
from os.path import isfile, isdir
import csv

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
from scipy.ndimage import imread

class Classifier:
    def __init__(self, checkpoint, train_x, train_y):
        self.inputs_ = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
        self.labels_ = tf.placeholder(tf.int64, shape=[None, train_y.shape[1]])

        fc = tf.contrib.layers.fully_connected(self.inputs_, 256)
        fc1 = tf.contrib.layers.fully_connected(fc, 256)

        self.logits = tf.contrib.layers.fully_connected(fc1, train_y.shape[1], activation_fn=None)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_, logits=self.logits)
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        self.predicted = tf.nn.softmax(self.logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicted, 1), tf.argmax(self.labels_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.checkpoint = checkpoint

    def train(self, train_x, train_y, val_x, val_y, epochs=30):
        iteration = 0

        # TODO - Falta arrumar esse Saver, tá dando resultados diferentes ...
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
        
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                for x, y in get_batches(train_x, train_y):
                    feed = {self.inputs_: x,
                            self.labels_: y}
                    loss, _ = sess.run([self.cost, self.optimizer], feed_dict=feed)
                    print("Epoch: {}/{}".format(e+1, epochs),
                        "Iteration: {}".format(iteration),
                        "Training loss: {:.5f}".format(loss))
                    iteration += 1
                    
                    if iteration % 5 == 0:
                        feed = {self.inputs_: val_x,
                                self.labels_: val_y}
                        val_acc = sess.run(self.accuracy, feed_dict=feed)
                        print("Epoch: {}/{}".format(e, epochs),
                            "Iteration: {}".format(iteration),
                            "Validation Acc: {:.4f}".format(val_acc))
            self.saver.save(sess, "checkpoints/%s.ckpt" % self.checkpoint)

    def predict(self, vgg, test_img_path, lb):
        # Run this cell if you don't have a vgg graph built
        test_img = imread(test_img_path)

        with tf.Session() as sess:
            input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            vgg.build(input_)

        with tf.Session() as sess:
            img = utils.load_image(test_img_path)
            img = img.reshape((1, 224, 224, 3))

            feed_dict = {input_: img}
            code = sess.run(vgg.relu6, feed_dict=feed_dict)
                
        with tf.Session() as sess:
            self.saver.restore(sess, "checkpoints/%s.ckpt" % self.checkpoint)
            
            feed = {self.inputs_: code}
            prediction = sess.run(self.predicted, feed_dict=feed).squeeze()
        
        plt.imshow(test_img)
        plt.show()

        plt.barh(np.arange(5), prediction)
        _ = plt.yticks(np.arange(5), lb.classes_)
        plt.show()

    def test(self, test_x, test_y):
        self.saver = tf.train.Saver()

        # TODO - Falta arrumar esse Saver, tá dando resultados diferentes ...
        with tf.Session() as sess:
            tf.global_variables_initializer()
            
            self.saver.restore(sess, "checkpoints/%s.ckpt" % self.checkpoint)
            
            feed = {self.inputs_: test_x,
                    self.labels_: test_y}
            test_acc = sess.run(self.accuracy, feed_dict=feed)
            print("Test accuracy: {:.4f}".format(test_acc))

def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x)//n_batches

    for ii in range(0, n_batches*batch_size, batch_size):
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y

def generate_features(classes, data_dir, vgg, batch_size=10):
    codes_list = []
    labels = []
    batch = []
    
    codes = None
    
    with tf.Session() as sess:
        # vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)

        for each in classes:
            print("Starting {} images".format(each))
            class_path = data_dir + each
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):
                # Add images to the current batch
                # utils.load_image crops the input images for us, from the center
                try:
                    img = utils.load_image(os.path.join(class_path, file))
                    batch.append(img.reshape((1, 224, 224, 3)))
                    labels.append(each)

                    # Running the batch through the network to get the codes
                    if ii % batch_size == 0 or ii == len(files):
                        images = np.concatenate(batch)

                        feed_dict = {input_: images}
                        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                        # Here I'm building an array of the codes
                        if codes is None:
                            codes = codes_batch
                        else:
                            codes = np.concatenate((codes, codes_batch))

                        # Reset to start building the next batch
                        batch = []
                        print('{} images processed'.format(ii))
            
                except Exception as e:
                    print(e)
            
        return codes, labels
    
# read codes and labels from file
def read_file(file):
    
    with open('labels_%s' % file) as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    with open('codes_%s' % file) as f:
        codes = np.fromfile(f, dtype=np.float32)
        codes = codes.reshape((len(labels), -1))
    
    return codes, labels

def get_data_labels(labels):
    lb = LabelBinarizer()
    lb.fit(labels)
    labels_vecs_train = lb.transform(labels)
    
    return labels_vecs_train, lb

def save_to_file(file, codes, labels):
    # write codes to file
    with open('codes_%s' % file, 'w') as f:
        codes.tofile(f)

    # write labels to file
    with open('labels_%s' % file, 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)

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
