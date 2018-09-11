# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet
from extract_cnn_vgg16_keras import ResNet

import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = True,
	help = "Path for output retrieved images")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
def load_h5(num=0):
    h5f = h5py.File(args["index"],'r')
    feats = h5f['dataset_1'][num*100:num*100+100]
#    imgNames = h5f['dataset_2'][:]
    h5f.close()
    return feats

        
print ("--------------------------------------------------")
print ("               searching starts")
print ("--------------------------------------------------")
    
# read and show query image
queryDir = args["query"]

# init VGGNet16 model
model = ResNet()
#model = VGGNet()

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


correct = 0
img_list = get_imlist(queryDir)
for i, img_path in enumerate(img_list):
    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(img_path)
    print (np.shape(queryVec))
    queryVec = queryVec.ravel()
    print (np.shape(queryVec))
    
    scores = []
    # for j in range(103):
        # feats = load_h5(j)
        # score = np.dot(queryVec, feats.T)
        # scores.append(score)
        # print("%d block search" %j)
    h5f = h5py.File(args["index"],'r')
    feats = h5f['dataset_1'][:]
    scores = np.dot(queryVec, feats.T)
    #scores = np.dot(queryVec, feats.T)/(np.linalg.norm(queryVec)*np.linalg.norm(feats.T))
    scores = np.array(scores)
    scores = scores.ravel()
#    h5f = h5py.File(args["index"],'r')
    imgNames = h5f['dataset_2'][:]
    h5f.close()
	
    rank_ID = np.argsort(scores)[::-1]
    print(np.shape(rank_ID))
    print (rank_ID[0:10])
    rank_score = scores[rank_ID]
    print (rank_score[0:10])
    # number of top retrieved images to show
    maxres = 1
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])] 
    #imlist = bytes.decode(imlist)
    print("For image {}".format(img_path))
    print ("top %d images in order are: " %maxres, imlist)
	
    for k,im in enumerate(imlist):
        im = bytes.decode(im)
        image = mpimg.imread(args["result"]+"/"+im)
        plt.title("search output %d" %(i+1))
        plt.imshow(image)
        plt.show()

 

# show top #maxres retrieved result one by one
# for i,im in enumerate(imlist):
    # #print(args["result"])
    # im = bytes.decode(im)
    # print(im)
    # image = mpimg.imread(args["result"]+"/"+im)
    # plt.title("search output %d" %(i+1))
    # plt.imshow(image)
    # plt.show()
