# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse

from extract_cnn_vgg16_keras import VGGNet
from extract_cnn_vgg16_keras import ResNet

from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def save_h5(times=0,name='data.h5',data=0,size=1000):
    if times == 0:
        h5f = h5py.File(name, 'w')
        dataset = h5f.create_dataset('dataset_1', (100, 401408),
                                     maxshape=(None, 401408),
                                     #chunks=True,
                                     dtype='float32')
    else:
        h5f = h5py.File(name, 'a')
        dataset = h5f['dataset_1']
    
    
    # space resize
    dataset.resize([times*100+100, size])
    # read data into memory 
    dataset[times*100:times*100+100] = data
    # print(sys.getsizeof(h5f))
    h5f.close()

'''
 Extract features and index the images
'''
if __name__ == "__main__":

    db = args["database"]
    img_list = get_imlist(db)
    
    print ("--------------------------------------------------")
    print ("         feature extraction starts")
    print ("--------------------------------------------------")
    
    feats = []
    names = []

    model = ResNet()
#    model = VGGNet()
    count = 0
    times = 0
#    pca = PCA(n_components=128, svd_solver='full')
    for i, img_path in enumerate(img_list):
        count = count + 1
        norm_feat = model.extract_feat(img_path)
#        norm_feat = norm_feat.reshape(-1,1)
#        norm_feat = norm_feat.T
        print(np.shape(norm_feat))
#        norm_feat = pca.fit_transform(norm_feat)
        norm_feat = norm_feat.ravel()
#        print(len(norm_feat))
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print ("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
        if count >= 100:
            feats = np.array(feats)
            print(np.shape(feats))
#            print(feats.dtype)
            save_h5(times,args["index"],feats,len(norm_feat))
            times = times + 1
            count = 0
            feats = []

#    feats = np.array(feats)
#    print(np.shape(feats))
#    feats = pca.fit_transform(feats)
#    print(np.shape(feats))
    # directory for storing extracted features
    output = args["index"]
    
    print ("--------------------------------------------------")
    print ("      writing feature extraction results ...")
    print ("--------------------------------------------------")
	
#    save_h5(times,args["index"],feats,len(norm_feat))
	
    h5f = h5py.File(output, 'a')
#    h5f.create_dataset('dataset_1', data = feats)
    dataset = h5f['dataset_1']
    
    
    # space resize
    dataset.resize([times*100+count, len(norm_feat)])#58
    # read data into memory 
    dataset[times*100:times*100+count] = feats
	
    #---------------
    namess = []
    for j in names:
        namess.append(j.encode())
    
    h5f.create_dataset('dataset_2', data = namess)

    #---------------
    h5f.close()
