caffe_root = '/media/lin/Disk2/caffe-latest/python'
import sys
sys.path.insert(0, caffe_root)
import caffe
import numpy as np
from PIL import Image
import random
import math
import skimage
import matplotlib.pyplot as plt
import pdb
import os

class CorrelationLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self. network_file= params['network_file']
        self.snapshot_prefix = params['snapshot_prefix']
        self.count = 0
        self.snap_iter = params['snapshot_iter']
        self.mean_file = params['mean_file']
        self.roots = params['roots']
        self.test_file = params['file']

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)

    def forward(self, bottom, top):
        pretrained_model = self.snapshot_prefix + str(self.count) + '.caffemodel'
        batch_shape = (1, 3, 224, 224)
        means = self.get_mean_npy(self.mean_file, crop_size = batch_shape[2:])
        caffe.set_mode_gpu()
        # #uncomment for debug when testing
        # null_fds = os.open(os.devnull, os.O_RDWR)
        # out_orig = os.dup(2)
        # os.dup2(null_fds, 2)
        net = caffe.Net(self.network_file, pretrained_model, caffe.TEST)  #set caffe model
        # os.dup2(out_orig, 2)
        # os.close(null_fds)

        file = open(self.test_file,'r')
        line = file.readlines()
        file.close()
        ground_label = []
        predict_label = []
        # loss = 0.0
        for i in range(len(line)):
            linesplit = line[i].split(' ')
            filename = linesplit[0]
            ground = float(linesplit[1].split('\n')[0])
            ground_label.append(ground)
            imgdir = self.roots + filename
            _load_img = self.load_img(imgdir, resize = (256, 256), isColor = True, crop_size = 224, crop_type = 'center_crop',
                     raw_scale = 255, means = means)
            net.blobs['data'].data[...] = _load_img  
            out = net.forward()
            predict = net.blobs['feat1'].data[...][0][0]
            predict_label.append(predict)
            # loss = loss + diff * diff
    
        labellist = np.array(ground_label)
        preclist = np.array(predict_label)

        pearson_correlation = np.corrcoef(labellist, preclist)[0][1]
        top[0].data[...] = pearson_correlation
        top[1].data[...] = np.mean(np.abs(labellist - preclist))
        # loss = loss / len(line) 
        top[2].data[...] = np.sqrt(np.mean(np.square(labellist - preclist)))
        # print pearson_correlation
        self.count = self.count + self.snap_iter
        

    def backward(self, top, propagate_down, bottom):
        pass

    def get_mean_npy(self, mean_bin_file, crop_size=None):
        mean_blob = caffe.proto.caffe_pb2.BlobProto()
        mean_blob.ParseFromString(open(mean_bin_file, 'rb').read())
        mean_npy = caffe.io.blobproto_to_array(mean_blob)
        _shape = mean_npy.shape
        mean_npy = mean_npy.reshape(_shape[1], _shape[2], _shape[3])

        if crop_size:
            mean_npy = mean_npy[
                :, (_shape[2] - crop_size[0]) / 2:(_shape[2] + crop_size[0]) / 2, 
                (_shape[3] - crop_size[1]) / 2:(_shape[3] + crop_size[1]) / 2]
        return mean_npy

    def crop_img(self, img, crop_size, crop_type='center_crop'):
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


    def load_img(self,path, resize=128, isColor=True,
                 crop_size=112, crop_type='center_crop',
                 raw_scale=1, means=None):
        '''
            crop_type is one of None, 'center_crop',
                                'random_crop', 'random_size_crop'
        '''
        img = skimage.io.imread(path)
        # pdb.set_trace()

        if resize is not None and img.shape != resize:
            img = skimage.transform.resize(img, resize, mode='reflect')
        if crop_size and crop_type:
            img = self.crop_img(img, crop_size, crop_type)
        if isColor:
            img = skimage.color.gray2rgb(img)
            img = img.transpose((2, 0, 1))
            img = img[(2, 1, 0), :, :]
        else:
            img = skimage.color.rgb2gray(img)
            img = img[np.newaxis, :, :]
        img = skimage.img_as_float(img).astype(np.float32) * raw_scale  #skimage float32 is between[0,1]

        if means is not None:
            if means.ndim == 1 and isColor:
                means = means[:, np.newaxis, np.newaxis]
            img -= means
            img = img / raw_scale
       
        return img
