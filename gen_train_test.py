import numpy as np
import h5py
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

dataset_dir = "/media/rodsnjr/Files/Datasets/gvc_dataset-master"
numpy_dataset_dir = "/media/rodsnjr/Files/Datasets/gvc_npy/"

num_train_samples = 10000

if not (os.path.exists(numpy_dataset_dir)):
    os.mkdir(numpy_dataset_dir)
    
labels = [name for name in os.listdir(dataset_dir)]

def get_label(one_hot_encode):
    intw = np.where(one_hot_encode == 1)[0][0]
    print(intw)
    return labels[intw]

def load_dataset(file, batch_size=10000):
    with h5py.File(os.path.join(numpy_dataset_dir, file),'r') as hf:
        x = hf["gvc_x"]
        y = hf["gvc_y"]

        return x[:batch_size], y[:batch_size]

class FitGenerator():
    def __init__(self):
        self.current_dataset = load_dataset("gvc_full.h5")
        self.current_batch = 0
        self.current_file = 0

    def generator(self, batch_size=32):
        max_range = 20000 // batch_size
        for i in range(max_range):
            yield self.get_batch(batch_size)
    
    def get_batch(self, batch_size):
        batch_x = self.current_dataset[0][self.current_batch : self.current_batch+batch_size]
        batch_y = self.current_dataset[1][self.current_batch : self.current_batch+batch_size]
        print(self.current_batch, self.current_batch+batch_size, batch_x.shape)
        self.current_batch += batch_size
        return batch_x, batch_y


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

generator = FitGenerator()

model.fit_generator(generator.generator(), epochs=1, steps_per_epoch=20)
score = model.evaluate(x_test, y_test, batch_size=128)