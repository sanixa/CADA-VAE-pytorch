

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
from sklearn import preprocessing

import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import cifar10
from keras_applications.resnet import ResNet101


'''
path = '/home/uscc/cvamc/data/plant/'   #文件读取路径
image_size = 128

###################load data-----------------------------------------------------
traindata = np.load(path+'traindata.npy')
testdata = np.load(path+'testdata.npy')
trainlabel = np.load(path+'trainlabel.npy')
testlabel = np.load(path+'testlabel.npy')
#### note. train=seen calss , test=unseen class

print(traindata.shape)
print(testdata.shape)
print(trainlabel.shape)
print(testlabel.shape)

model = ResNet101(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3), backend=keras.backend,
                  layers=keras.layers, models=keras.models, utils=keras.utils, pooling='avg')

traindata = model.predict(traindata)
testdata = model.predict(testdata)

seen_x_train, seen_x_test, seen_y_train, seen_y_test = train_test_split(traindata, trainlabel, test_size=0.20, random_state=42)
unseen_x_train, unseen_x_test, unseen_y_train, unseen_y_test = train_test_split(testdata, testlabel, test_size=0.20, random_state=42)
###note. here, train=real trainset, test=real testset
##test_seen = 2500, test_unseen=1300

train_data = np.concatenate((np.array(seen_x_train), np.array(unseen_x_train)), axis=0)
train_label = np.concatenate((seen_y_train, unseen_y_train), axis=0)
print(train_data.shape, train_label.shape)
test_data = np.concatenate((np.array(seen_x_test), np.array(unseen_x_test)), axis=0)
test_label = np.concatenate((seen_y_test, unseen_y_test), axis=0)
print(test_data.shape, test_label.shape)
data = np.concatenate((train_data, test_data), axis=0)
label = np.concatenate((train_label, test_label), axis=0)
print(data.shape, label.shape)
np.save('data.npy', data)
np.save('label.npy', label)
'''
data = np.load('data.npy')
label = np.load('label.npy')


attr = np.load('/home/uscc/cvamc/data/plant/class_attr.npy')

indices = list(range(19000))
np.random.shuffle(indices)

data_list = []
label_list = []
trainval_loc = []
test_seen_loc = []
test_unseen_loc = []
cnt = 0
for i in indices:
    data_list.append(data[i])
    label_list.append(label[i]+1)
    if i < 15200:
        trainval_loc.append([cnt])
    elif i < 17700 and i >= 15200:
        test_seen_loc.append([cnt])
    elif i < 19000 and i >= 17700:
        test_unseen_loc.append([cnt])
    cnt = cnt +1

data_list = np.row_stack(data_list)
label_list = np.row_stack(label_list)
trainval_loc = np.row_stack(trainval_loc)
test_seen_loc = np.row_stack(test_seen_loc)
test_unseen_loc = np.row_stack(test_unseen_loc)

path= 'data/' + 'plant'+ '/res101.mat'

sio.savemat(path, {
                   'features': data_list.transpose(),
                   'labels': label_list})

path= 'data/' + 'plant'+ '/att_splits.mat'
sio.savemat(path, {
                   'trainval_loc': trainval_loc,
                   'test_seen_loc': test_seen_loc,
                   'test_unseen_loc': test_unseen_loc,
                   'att': attr.transpose(),
                   'train_loc': test_unseen_loc,  ##non used in code
                   'val_loc': test_unseen_loc})    ##non used in code
'''
path= 'data/' + 'plant'+ '/att_splits.mat'
matcontent = sio.loadmat(path)
print(matcontent['att'])
print(np.array(matcontent['trainval_loc']).shape)
print(np.array(matcontent['train_loc']).shape)
print(np.array(matcontent['val_loc']).shape)
print(np.array(matcontent['test_seen_loc']).shape)
print(np.array(matcontent['test_unseen_loc']).shape)
print(np.array(matcontent['att']).shape)

path= 'data/' + 'AWA2'+ '/att_splits.mat'
matcontent = sio.loadmat(path)
print(matcontent['att'])
print(np.array(matcontent['trainval_loc']).shape)
print(np.array(matcontent['train_loc']).shape)
print(np.array(matcontent['val_loc']).shape)
print(np.array(matcontent['test_seen_loc']).shape)
print(np.array(matcontent['test_unseen_loc']).shape)
print(np.array(matcontent['att']).shape)

path= 'data/' + 'plant'+ '/res101.mat'
matcontent = sio.loadmat(path)
print(matcontent['labels'])
print(np.array(matcontent['features']).shape)
print(np.array(matcontent['labels']).shape)

path= 'data/' + 'CUB'+ '/res101.mat'
matcontent = sio.loadmat(path)
print(matcontent['labels'])
print(np.array(matcontent['features']).shape)
print(np.array(matcontent['labels']).shape)
'''
