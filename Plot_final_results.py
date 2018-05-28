
# coding: utf-8

# # Import useful librarys

# In[ ]:


import PIL
from PIL import Image
from PIL import ImageChops # used for multiplying images
import numpy as np
import os
import pickle
import cv2
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # Upload and plot the saved result 

# ## 1. History

# In[ ]:


hist = pickle.load( open( "/home/yasser/Documents/callbacks/history.pickle", "rb" ) )


# In[ ]:


def plot_history(history):
    out_file1 = os.path.join('/home/yasser/Documents/callbacks/', "history-loss.png")
    out_file2 = os.path.join('/home/yasser/Documents/callbacks/', "history-accu.png")
    loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history[l], 'b', label='Training loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_file1)
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out_file2)
    plt.show()


# In[ ]:


plot_history(hist)


# ## 2. Precision-recall curve

# In[ ]:


precision_recall = pickle.load( open( "/home/yasser/Documents/callbacks/precision_recall.pickle", "rb" ) )
precision_recall.keys()


# In[ ]:


keys = list(precision_recall.keys())
values = precision_recall.values()
recall = list(values)[2]
precision  = list(values)[1]
recall.shape,precision.shape


# In[ ]:


# Create the precision-recall curve.
out_file = os.path.join('/home/yasser/Documents/callbacks/', "precision_recall.png")
plt.plot(recall, precision, label="Precision-Recall curve")
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.savefig(out_file)


# ## 3. Confusion matrix 

# In[ ]:


confusion_matrix = np.load("/home/yasser/Documents/callbacks/confusion_matrix.npy")


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
np.set_printoptions(precision=2)


# In[ ]:


out_file01 = os.path.join('/home/yasser/Documents/callbacks/', "CM_without_normalization.png")
out_file02 = os.path.join('/home/yasser/Documents/callbacks/', "CM_with_normalization.png")


# In[ ]:


# Plot non-normalized confusion matrix
print('Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['1','0'],
                      title='Confusion matrix, without normalization')
plt.savefig(out_file01)
plt.show()


# In[ ]:


# Plot normalized confusion matrix
print("Normalized confusion matrix")
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['1','0'], normalize=True,
                      title='Normalized confusion matrix')
plt.savefig(out_file02)
plt.show()


# ## 4. Plot the predicted images

# In[ ]:


res = np.load("/home/yasser/Documents/callbacks/result.npy")
res.shape


# In[ ]:


data_predection = defaultdict(list)
data_label = defaultdict(list)
data_false_pos = defaultdict(list)


# In[ ]:


for i, ((prediction_patch, label_patch, false_positive_patch), position, path) in enumerate(res):
    if path not in data_label:
        data_label[path]=[(label_patch,position)]
    else:
        data_label[path].append((label_patch,position))
    if path not in data_false_pos:
        data_false_pos[path]=[(false_positive_patch.reshape(64,64),position)]
    else:
        data_false_pos[path].append((false_positive_patch.reshape(64,64),position))
    if path not in data_predection:
        data_predection[path]=[(prediction_patch.reshape(64,64),position)]
    else:
        data_predection[path].append((prediction_patch.reshape(64,64),position))


# In[ ]:


data_predection.keys()


# In[ ]:


def CollectBatches(file_name):
    rows, cols = 3000, 4000
    all_patched_data = []
    patch_indexes = itertools.product(range(0, rows, 64), range(0, cols, 64))
    default = np.asarray(np.zeros((64,64)), dtype=np.float32)
    new_image = {}
    image = file_name#data_predection['DJI_0177']
    index_kept = []
    n = len(image)
    for i in range(n):
        index=image[i][1]
        index_kept.append(index)
        new_image[index]=image[i][0]
    for index in patch_indexes:
        if index not in index_kept:
            new_image[index]=default
    #sorted(new_image.keys())
    #new_image[(0,0)]
    img_mixed = new_image[(0,0)]
    for j in range(64, 3968, 64):
        img_mixed = np.concatenate((img_mixed, new_image[(0,j)]), axis =1)
    for i in range(64, 2944, 64):
        img = new_image[(i,0)]
        for j in range(64, 3968, 64):
            img = np.concatenate((img, new_image[(i,j)]), axis =1)
        img_mixed = np.concatenate((img_mixed, img), axis =0)
    
    return img_mixed


# In[ ]:


plt.imshow(CollectBatches(data_predection['DJI_0177']))
plt.imsave("/home/yasser/Documents/sgd005200/prediction.jpg", CollectBatches(data_predection['DJI_0177']))


# In[ ]:


plt.imshow(CollectBatches(data_label['DJI_0177']))


# In[ ]:


plt.imshow(CollectBatches(data_false_pos['DJI_0177']))
plt.imsave("/home/yasser/Documents/sgd005200/false_pos.jpg", CollectBatches(data_false_pos['DJI_0177']))


# ## 5. Overlaping the predicted image with the falsely predicted label on the original image

# In[ ]:


img = cv2.imread('/home/yasser/Documents/AIMS Essay/Test/how_to_do_math_for_deep_learning-master/training/DJI_0177.JPG')
#plt.imshow(img)


# In[ ]:


crop_img = img[0:2944, 0:3968]
#plt.imshow(crop_img)


# In[ ]:


mask1 = cv2.imread('/home/yasser/Documents/callbacks/prediction.jpg')
#plt.imshow(mask1)


# In[ ]:


mask2 = cv2.imread('/home/yasser/Documents/callbacks/false_pos.jpg')
#plt.imshow(mask2)


# In[ ]:


img2gray = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)


# In[ ]:


backtorgb = cv2.cvtColor(img2gray,cv2.COLOR_GRAY2BGR)
#plt.imshow(backtorgb)


# In[ ]:


masked = np.ma.masked_where(backtorgb == 0, backtorgb)
masked2 = np.ma.masked_where(mask2 == 0, mask2)
plt.figure(1)
#plt.subplot(1,2,1)
plt.imshow(crop_img, 'gray', interpolation='none')


# In[ ]:


#plt.subplot(1,2,2)
plt.figure(2)
plt.imshow(crop_img, 'gray', interpolation='none')
plt.imshow(masked, 'jet', interpolation='none', alpha=0.32)
plt.imshow(masked2, 'jet', interpolation='none', alpha=0.52)
plt.savefig("/home/yasser/Documents/callbacks/compare.jpg")


# In[ ]:







