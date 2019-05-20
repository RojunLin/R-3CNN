#!/usr/bin/env python
import random
import skimage
import numpy as np
import matplotlib.pyplot as plt
import math
import caffe
import os

def get_mean_npy(mean_bin_file, crop_size=None):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_bin_file, 'rb').read())

    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    _shape = mean_npy.shape
    mean_npy = mean_npy.reshape(_shape[1], _shape[2], _shape[3])

    if crop_size:
        mean_npy = mean_npy[
            :,
            (_shape[2] - crop_size[0]) / 2:(_shape[2] + crop_size[0]) / 2,
            (_shape[3] - crop_size[1]) / 2:(_shape[3] + crop_size[1]) / 2]
    return mean_npy

def crop_img(img, crop_size, crop_type='center_crop'):
    '''
        crop_type is one of 'center_crop',
                            'random_crop', 'random_size_crop'
    '''

    if crop_type == 'center_crop':
        sh = crop_size 
        sw = crop_size
        hh = (img.shape[0] - sh) / 2
        ww = (img.shape[1] - sw) / 2
    elif crop_type == 'random_crop':
        sh = crop_size
        sw = crop_size
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    elif crop_type == 'random_size_crop':
        sh = random.randint(crop_size[0], img.shape[0])
        sw = random.randint(crop_size[1], img.shape[1])
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    img = img[hh:hh + sh, ww:ww + sw]
    if crop_type == 'random_size_crop':
        img = skimage.transform.resize(img, crop_size, mode='reflect')    
    return img


def load_img(path, resize=128, isColor=True,
             crop_size=112, crop_type='center_crop',
             raw_scale=1, means=None):
    '''
        crop_type is one of None, 'center_crop',
                            'random_crop', 'random_size_crop'
    '''
    img = skimage.io.imread(path)

    if resize is not None and img.shape != resize:
        img = skimage.transform.resize(img, resize, mode='reflect')
    if crop_size and crop_type:
        img = crop_img(img, crop_size, crop_type)
    if isColor:
        img = skimage.color.gray2rgb(img)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :]
    else:
        img = skimage.color.rgb2gray(img)
        img = img[np.newaxis, :, :]
    img = skimage.img_as_float(img).astype(np.float32) * raw_scale 

    if means is not None:
        if means.ndim == 1 and isColor:
            means = means[:, np.newaxis, np.newaxis]
        img -= means
        img = img / 255

    return img



def main():
'''
    5-fold cross validation
'''
    root = '../data/faces'
    network_file = './resnext_deploy.prototxt'
    pretrained_model = ['../models/1/hinge_R3CNN.caffemodel', '../models/2/hinge_R3CNN.caffemodel', \
        '../models/3/hinge_R3CNN.caffemodel', '../models/4/hinge_R3CNN.caffemodel', '../models/5/hinge_R3CNN.caffemodel']
    # pretrained_model = ['../models/1/lsep_R3CNN.caffemodel', '../models/2/lsep_R3CNN.caffemodel', 
    #     '../models/3/lsep_R3CNN.caffemodel', '../models/4/lsep_R3CNN.caffemodel', '../models/5/lsep_R3CNN.caffemodel']     
    
    mean_file = ["../data/1/256_train_mean.binaryproto", "../data/2/256_train_mean.binaryproto", \
        "../data/3/256_train_mean.binaryproto", "../data/4/256_train_mean.binaryproto", "../data/5/256_train_mean.binaryproto"]
        
    test_file = ['../data/1/test_1.txt', '../data/2/test_2.txt', '../data/3/test_3.txt', \
            '../data/4/test_4.txt', '../data/5/test_5.txt']

    for i in range(5):
        print('start testing------')

        # get mean file
        batch_shape = (1, 3, 224, 224)
        means = get_mean_npy(mean_file[i], crop_size = batch_shape[2:])

        # set mode
        caffe.set_mode_gpu()

        # set caffe model
        null_fds = os.open(os.devnull, os.O_RDWR)
        out_orig = os.dup(2)
        os.dup2(null_fds, 2)
        net = caffe.Net(network_file, pretrained_model[i], caffe.TEST)
        os.dup2(out_orig, 2)
        os.close(null_fds)

        # open test file
        with open(test_file[i], 'r') as f:
            lines = f.readlines()

        label_list = []
        prec_list = []

        for line in lines:
            linesplit = line.split(' ')
            label = float(linesplit[1].split("\r")[0])
            img = os.path.join(root, linesplit[0])
            img_data = load_img(img, resize = (256, 256), isColor = True, crop_size = 224, crop_type = 'center_crop',
                     raw_scale = 255, means = means)

            net.blobs['data'].data[...] = img_data
            out = net.forward()
            prec = net.blobs['feat1'].data[...][0][0]
            label_list.append(label)
            prec_list.append(prec)

        label_list = np.array(label_list)
        prec_list = np.array(prec_list)
        correlation = np.corrcoef(label_list, prec_list)[0][1]
        mae = np.mean(np.abs(label_list - prec_list))
        rmse = np.sqrt(np.mean(np.square(label_list - prec_list)))

        print('Model: {name}\t'
            'Correlation: {correlation:.4f}\t'
            'Mae: {mae:.4f}\t'
            'Rmse: {rmse:.4f}\t'.format(name=pretrained_model[i], correlation=float(correlation), mae=float(mae), rmse=float(rmse)))


if __name__ == '__main__':
    main()
