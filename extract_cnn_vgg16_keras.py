# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
#import deeplearning
from numpy import linalg as LA

from sklearn.decomposition import PCA

#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
from deeplearning.vgg16 import VGG16
from deeplearning.resnet50 import ResNet50
from keras.preprocessing import image
from deeplearning.imagenet_utils import preprocess_input

from keras.models import Model

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        base_model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        print(np.shape(feat))
#        print(feat[0])
		#        feat = feat.ravel()
        norm_feat = feat[0]/LA.norm(feat[0])

        norm_feat = norm_feat.T
        # print(np.shape(norm_feat))

        # norm_feat = norm_feat.reshape(256,-1)

        # try:
            
            # print(np.shape(norm_feat))
            # pca = PCA(n_components=128)
            # pca.fit(norm_feat)
            # norm_feat = pca.transform(norm_feat)
            # norm_feat = norm_feat/LA.norm(norm_feat)
        # except:
            # print("SVD did not converge")
            # print("--------------------------------------------------------------------------------------------------------------")
            # # norm_feat = norm_feat[~np.isnan(norm_feat)]
            # # print(np.shape(norm_feat))
            # # pca = PCA(n_components=128)
            # # pca.fit(norm_feat)
            # # norm_feat = pca.transform(norm_feat)

        return norm_feat


def max_mask(fea):
    mask = np.zeros((np.shape(fea)[1],np.shape(fea)[2]),dtype=bool)
    for j in range(0, np.shape(fea)[0]):
        temp = fea[j, :, :]
#        print(np.shape(temp))
        m = temp.max(1)
        p1 = np.argmax(temp,axis=1)
        p2 = np.argmax(m)
#        print (p1,p2)
        mask[p1[p2], p2] = 1

    mask = mask[:]
    return mask


class ResNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        base_model = ResNet50(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn3d_branch2c').output)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        #print(np.shape(feat))
#        print(feat[0])
		#        feat = feat.ravel()
        norm_feat = feat[0]/LA.norm(feat[0])

        norm_feat = norm_feat.T
        print(np.shape(norm_feat))

        # norm_feat = norm_feat.reshape(256,-1)

        # try:
            
            # print(np.shape(norm_feat))
            # pca = PCA(n_components=128)
            # pca.fit(norm_feat)
            # norm_feat = pca.transform(norm_feat)
            # norm_feat = norm_feat/LA.norm(norm_feat)
        # except:
            # print("SVD did not converge")
            # print("--------------------------------------------------------------------------------------------------------------")
            # # norm_feat = norm_feat[~np.isnan(norm_feat)]
            # # print(np.shape(norm_feat))
            # # pca = PCA(n_components=128)
            # # pca.fit(norm_feat)
            # # norm_feat = pca.transform(norm_feat)

        #try:
        mask = max_mask(norm_feat)
#        print(mask)
        mask = np.tile(mask,[np.shape(norm_feat)[0],1,1])
        print(np.shape(mask))
        masked_fea = norm_feat[mask]
        print (np.shape(masked_fea))
            #norm_feat = np.reshape()
        #except:
            #print("mask error")

        return norm_feat
