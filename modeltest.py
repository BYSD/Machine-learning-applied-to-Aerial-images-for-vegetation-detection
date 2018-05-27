
# coding: utf-8

# In[ ]:


from __future__ import print_function
from __future__ import division


# In[ ]:
from collections import defaultdict
import rasterio
import glob
import matplotlib
import time
import pickle
import os
import itertools
import matplotlib.pyplot as plt
import cv2
from os.path import join
plt.style.use('ggplot')


# In[ ]:


import warnings as w
w.simplefilter(action = 'ignore', category = FutureWarning)
w.simplefilter(action = 'ignore', category = ResourceWarning)


import numpy      as np
import random     as rn
import tensorflow as tf
import os
# In[ ]:
os.environ['PYTHONHASHSEED']='0'
np.random.seed(10)
rn.seed(7)
tf.set_random_seed(42)




import keras

from PIL                       import Image
from keras                     import backend as K
from keras.layers              import Input, Dense, Dropout, Flatten, Activation
from keras.layers              import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers              import UpSampling2D,concatenate
from keras.models              import Sequential, Model
from keras.models              import model_from_json
from keras.optimizers          import Adam
from keras.optimizers          import SGD, RMSprop
from keras.applications.vgg16  import VGG16
from keras.callbacks           import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn                   import metrics
from sklearn.model_selection   import train_test_split


# In[ ]:

def simple_model(input_shape=(64, 64, 3), num_classes=4096):
    inputs = Input(shape=input_shape)
    # 64

    down2 = Conv2D(64, (3, 3), padding='same')(inputs)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(64, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(128, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(128, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(256, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(256, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(512, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    conv = Conv2D(64, (1, 1), activation='relu')(up2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    poop = Dropout(0.25)(pool)
    flat = Flatten()(pool)
    flat = Dense(num_classes, activation='relu')(flat)
    flat = Dropout(0.25)(flat)
    classify = Dense(num_classes, activation='sigmoid')(flat)
    model = Model(inputs=inputs, outputs=classify)
    
    return model

# In[ ]:


# paths to an image and its groundtruth 
train_image_path = '/home/ybouchareb/yasser/Input_training_data/training/*.JPG'
train_gt_path    = '/home/ybouchareb/yasser/Output_training_data/training_label/*.jpg'


# In[ ]:


class GenerateDataset():
    
    def __init__(self, image_path, gt_path, patch_size):
        
        self.patch_size=patch_size
        self.image_path = image_path
        self.gt_path = gt_path
    # Convert image and groundtruth to matrices
        def load_image( self ) :
        DATA = {}
        DATA = defaultdict(list)
        for filename in sorted(glob.glob(self.image_path))[:5]:
            img = Image.open(filename)
            img.load()
            imgarr =  np.asarray(img, dtype=np.float32)
            outputname = filename.split('/')[-1].split('.')[0]
            DATA[outputname]=imgarr

        return DATA
    
    def load_gt( self ) :
        DATA = {}
        DATA = defaultdict(list)
        for filename in sorted(glob.glob(self.gt_path))[:5]:
            img = Image.open(filename)
            img.load()
            imgarr =  np.asarray(img, dtype=np.float32)
            outputname = filename.split('/')[-1].split('.')[0]
            DATA[outputname]=imgarr

        return DATA

    def create_patches(self, bands_data, outputname):
        #, alpha):
        
        """ Cut the image and groundtruth which is given as a matrix into patches of the given size 
        inputs: 
        bands_data (ndarray): matrix form of image or groundtruth
        patch_size (int): size of the cutouts
        path_to_geotiff: 

        output: 
        List of all cutouts/patches
        """
        #overlap = alpha* self.patch_size
        
        rows, cols = bands_data.shape[0], bands_data.shape[1]
        all_patched_data = []
        patch_indexes = itertools.product(range(0, rows, self.patch_size), range(0, cols, self.patch_size))
        # for in range(len(patch_indexes)):
        # patch_indexes[i+1]
        for (row, col) in patch_indexes:
            in_bounds = row + self.patch_size < rows and col + self.patch_size < cols
            if in_bounds:
                new_patch = bands_data[row:row + self.patch_size, col:col + self.patch_size]
                if new_patch.sum() != 0:
                    all_patched_data.append((new_patch, (row, col), outputname))
                

        return all_patched_data


    def get_matrix_form(self, features, labels):
        """ Transform a list of tuples of features and labels to a matrix which contains
        only the patches used for training a model."""
        features = [patch for patch, position, path in features]
        labels =[patch for patch, position, path in labels]

        # The model will have one output corresponding to each pixel in the feature patch.
        # So we need to transform the labels which are given as a 2D bitmap into a vector.
        labels = np.reshape(labels, (len(labels), self.patch_size * self.patch_size))
        return (np.array(features), np.array(labels))
    def normalise_input(self, features):
        """ Normalise the features/data such that all values are in the range [0,1]. """
        #features = features.astype(np.float32)

        return np.multiply(features, 1.0 / 255.0)
    
    def binarize_gt(self, gt_data):
        """ create the bitmap for a given image. """

        #gt_data = rasterio.open(self.gt_path).read(1)
        gt_data[gt_data > 0] = 1
    
        return gt_data


# In[ ]:


data_gen = GenerateDataset(train_image_path, train_gt_path, 64)


# In[ ]:


# Read tiff files (image and the corresponding groundtruths)
x = data_gen.load_image()
y = data_gen.load_gt()


# In[ ]:


# Binarize grounstruth
for i in y.keys():
    y[i] = data_gen.binarize_gt(y[i])

# In[ ]:


# Create patches of size n x n
# Create patches of size n x n
raw_data = []
for i in x.keys():
    raw_data += data_gen.create_patches(x[i], i)
raw_gt = []
for i in y.keys():
    raw_gt += data_gen.create_patches(y[i], i)
# In[ ]:

DATA = {}
from collections import defaultdict
DATA = defaultdict(list)
for i in raw_gt:
    name = i[2]
    pos = i[1]
    if name not in DATA:
        DATA[name]=[pos]
    else:
        DATA[name].append(pos)

raw_data = [(_, position, path) for i, (_, position, path) in enumerate(raw_data) if position in DATA[path]]


# In[ ]:


# Extract only the intensity values from raw data and raw groundtruth
data, gt = data_gen.get_matrix_form(raw_data, raw_gt)


# In[ ]:


# Normalize data
data = data_gen.normalise_input(data)


# In[ ]:

w.resetwarnings()
# Create train-test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, gt, test_size=0.25, random_state=1)


# In[ ]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


MODELS_DIR = "HOME"

def init_model(nb_filters, filter_size, stride, patch_size, pool_size,
        pool_stride, learning_rate, momentum, decay, model_dir):
    
    #intialize model
    num_channels = 3
    model = Sequential()

    model = simple_model()
    model = compile_model(model, learning_rate, momentum, decay)
    
    # Print a summary of the model to the console.

    print("Summary of the model")
    model.summary()
        
    save_model(model, model_dir)
    
    return model


# In[ ]:


def compile_model(model, learning_rate, momentum, decay):
    """ Compile the keras model with the given hyperparameters."""
    optimizer = RMSprop(lr=learning_rate) #SGD(lr=learning_rate, momentum=momentum, decay=decay)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# In[ ]:


def save_model(model, path):
    """ Save a keras model and its weights at the given path. """
    
    print("Save trained model to {}.".format(path))
    model_path = os.path.join(path, "model.json")
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
        
    weights_path = os.path.join(path, "weights.h5")
    model.save_weights(weights_path)

def save_makedirs(path):
    """ Create directory if and only if it does not exist yet"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# In[ ]:


def train_model(model, x, y, patch_size, model_dir,
                nb_epoch=10, checkpoints=False, tensorboard=False,
               earlystop=False):
    """ Train the model with the given features and labels """
    
    print("Start training.")
    model.fit(x, y, epochs=nb_epoch, validation_split=0.15)
    save_model(model, model_dir)
    
    return model 


# In[ ]:


timestamp = time.strftime("%d_%m_%Y_%H%M")
model_id = "{}".format(timestamp)
model_dir = os.path.join(MODELS_DIR, model_id)
save_makedirs(model_dir)


# In[ ]:


model = init_model(64, 9, (2,2), 64, 2, 1, 0.0005, 0.9, 0.0, model_dir)


# In[ ]:


model = train_model(model, x_train, y_train,64, model_dir)


# In[ ]:

def evaluate_model(model, features, labels, patch_size, out_path, out_format='GeoTIFF'):
    """ Calculate several metrics for the model and create a visualisation of the test dataset. """
    
    print('_' * 100)
    print('Start evaluating model.')
    
    #X, y_true = data_gen.get_matrix_form(features, labels)
    #X = data_gen.normalise_input(X)
    X = features
    y_true = labels
    y_predicted = model.predict(X)
    predicted_bitmap = np.array(y_predicted)
    
    # Since the model only outputs probabilities for each pixel we have 
    # to transform them into 0s and 1s. For the sake of of simplicity we 
    # simply use a cut of value of 0.5.
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0
    
    false_positives = get_false_positives(predicted_bitmap, y_true)
    visualise_predictions(predicted_bitmap, labels, false_positives, patch_size, out_path, out_format=out_format)
    
    # We have to flatten our predictions and labels since by default the metrics are calculated by 
    # comparing the elements in the list of labels and predictions elementwise. So if we would not flatten
    # our results we would only get a true positive if we would predict every pixel in an entire patch right.
    # But we obviously only care about each pixel individually.
    y_true = y_true.flatten()
    y_predicted = y_predicted.flatten()
    predicted_bitmap = predicted_bitmap.flatten()
    out_file = os.path.join(out_path, "confusion_matrix.npy")
    np.save(out_file, metrics.confusion_matrix(y_true, predicted_bitmap))
    print("Accuracy on test set: {}".format(metrics.accuracy_score(y_true, predicted_bitmap)))
    print("Precision on test set: {}".format(metrics.precision_score(y_true, predicted_bitmap, average='weighted')))
    print("Recall on test set: {}".format(metrics.recall_score(y_true, predicted_bitmap, average='weighted')))
    precision_recall_curve(y_true, y_predicted, out_path)

def visualise_predictions(predictions, labels, false_positives, patch_size, out_path, out_format="GeoTIFF"):
    """ Create a new GeoTIFF image which overlays the predictions of the model. """
    
    predictions = np.reshape(predictions,
                             (len(labels), patch_size, patch_size, 1))
    #print('predictions shape: ', predictions.shape)
    #print('predictions: ', predictions)
    false_positives = np.reshape(false_positives,
                                 (len(labels), patch_size, patch_size, 1))
    #print('False positives shape: ', false_positives.shape)
    #print('False positives: ', false_positives)
    results = []
    # We want to overlay the predictions and false positives on a GeoTIFF but we don't
    # have any information about the position in the source for each
    # patch in the predictions and false positives. We get this information from the labels.
    
    for i, (_, position, path_to_geotiff) in enumerate(labels):
        prediction_patch = predictions[i, :, :, :]
        false_positive_patch = false_positives[i, :, :, :]
        label_patch = labels[i][0]
        results.append(((prediction_patch, label_patch, false_positive_patch), position, path_to_geotiff))
    #print("RESULTS: ", results)
    #visualise_results(results, patch_size, out_path, out_format=out_format) 
        
def precision_recall_curve(y_true, y_predicted, out_path):
    """ Create a PNG with the precision-recall curve for our predictions """
    
    print("Calculate precision recall curve.")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_predicted)
    #print("y_true: {}, y_predicted: {}".format(y_true, y_predicted))
    #print("precision: {}, recall: {}, thresholds: {}".format(precision, recall, thresholds))
    # Save the raw precision and recall results to a pickle since we might want to
    # analyse them later
    out_file = os.path.join(out_path, "precision_recall.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds
                }, out)
        
def get_false_positives(predictions, labels):
    """ Get false positives for the given predicitions and labels. """
    
    FP = np.logical_and(predictions == 1, labels == 0)
    false_positives = np.copy(predictions)
    false_positives[FP] = 1
    false_positives[np.logical_not(FP)] = 0
    #print("FALSE POSITIVES: ", false_positives)
    return false_positives


# In[ ]:


evaluate_model(model, x_test, y_test, 64, model_dir)

