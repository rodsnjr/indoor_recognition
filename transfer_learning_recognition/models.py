from enum import Enum

from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
import utils
import numpy as np

img_width, img_height = 224, 224

def build_model(model):
    if type(model) != str:
        model = model.name
    
    if model == "VGG16":
        return applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model  == "VGG19":
        return applications.vgg19.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model == "INCEPTION":
        return applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model == "XCEPTION":
        return applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model == "RESNET50":
        return applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model == "MOBILENET":
        return applications.mobilenet.MobileNet(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    elif model == "INCEPTIONRESNET":
        return applications.inception_resnet_v2.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

class Models(Enum):
    """
    Vou guardar a lista de models que usaremos para o Transfer Learning.
    É bom dar uma completada nas descrições.
    """
    
    VGG16=("Depth 23, Top 1: 0.715 (Imagenet)"),
    VGG19=("Depth 26, Top 1: 0.0.727 (Imagenet)"),
    INCEPTION=("Depth 159, Top 1: 0.788 (Imagenet)"),
    XCEPTION=("Depth 126, Top 1: 0.790 (Imagenet)"),
    RESNET50=("Depth 168, Top 1: 0.759 (Imagenet)"),
    MOBILENET=("Depth 88, Top 1: 0.665 (Imagenet)"),
    INCEPTIONRESNET=("Depth 572, Top 1: 0.804 (Imagenet)")

    def __init__(self, description):
        self.description = description
        self.model = None
        self.layers = None

    def freeze_layers(self):
        for layer in self.model.layers:
            layer.trainable = False
        self.layers = self.model.layers

    def build(self):
        self.model = build_model(self.name)
        self.freeze_layers()
        print("Model Build ...")
        return self.layers

    def get_features(self, directory, batch_size=10):
        if type(directory) != utils.Directory:
            raise Exception("Must be of type Directory enum (in utils package)")
        
        codes_list = []
        labels = []
        batch = []    
        codes = None

        for each in directory.classes:
            print("Starting {} images".format(each))
            class_path = directory.dir + each
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
                        
                        codes_batch = self.model.predict(images)
                        #print(codes_batch.shape)
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


def get_classifier_model(train_x):
    clf_model = Sequential()
    clf_model.add(Dense(512, activation="relu", input_shape=[train_x.shape[1],]))
    clf_model.add(Dropout(0.5))
    clf_model.add(Dense(256, activation="relu"))
    clf_model.add(Dropout(0.5))
    clf_model.add(Dense(256, activation="relu"))
    clf_model.add(Dense(5, activation="softmax"))
    clf_model.compile(loss = "categorical_crossentropy", 
              optimizer = "adam", 
              metrics=["accuracy"])

    return clf_model

