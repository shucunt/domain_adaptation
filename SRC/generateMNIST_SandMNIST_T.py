# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:44:06 2017

@author: shucun Tian
"""

'''
生成MNIST-M  MNIST数据集
将图片保存到/home/zhangshu/faq/shucunt/temp/domainAdaption/data/imageMNIST-M
          /home/zhangshu/faq/shucunt/temp/domainAdaption/data/imageMNIST中

只生成了test数据集，MNIST和MNIST-M一一对应

'''
import cPickle
import gzip
import os
import sys
import time
import random
import numpy


import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from PIL import Image
from scipy.misc import imsave


"""
加载MNIST数据集load_data()
mnist.pkl将60000个训练数据分成了50000个训练数据和10000校正数据集；
每个数组由两部分内容组成，一个图片数组和一个标签数组，图片数组的每一行代表一个图片的像素，有784个元素（28×28）
"""
def to_image(dataset_s, dataset_t):

    print '... loading data'
    
    
    #从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
    #主要用到python里的gzip.open()函数,以及 cPickle.load()。
    #‘rb’表示以二进制可读的方式打开文件
    f = open(dataset_s, 'rb')
    train_set_source, valid_set_source, test_set_source = cPickle.load(f)
    f.close()

    f = open(dataset_t, 'rb')
    train_set_target, valid_set_target, test_set_target = cPickle.load(f)
    f.close()


    def to_image_source(data_xy, image_num, set_class):
        data_x, data_y = data_xy
        
        file_name = 0
        for k in range(image_num):
            temp = [([0] * 28) for i in range(28)]
            for i in range(28):
                for j in range(28):
                    temp[i][j] = data_x[k][i * 28 + j]
            imsave(os.path.join(os.path.abspath('..') ,'data/imageMNIST_S/' + set_class + str(file_name) + '.png'), temp)
            file_name += 1 #命名新生成文件

    def to_image_target(data_xyz, image_num, set_class):
        data_x, data_y, data_z = data_xyz
        file_name = 0
        for k in range(image_num):
            temp = [([0] * 28) for i in range(28)]
            for i in range(28):
                for j in range(28):
                    temp[i][j] = data_x[k][i * 28 + j]
            imsave(os.path.join(os.path.abspath('..') ,'data/imageMNIST_T/' + set_class + str(file_name) + '.png'), temp)
            file_name += 1 #命名新生成文件


    to_image_source(train_set_source, 10, 'train_')
    to_image_source(test_set_source, 10, 'test_')
    to_image_source(valid_set_source, 10, 'valid_')
    to_image_target(train_set_target, 10, 'train_')
    to_image_target(test_set_target, 10, 'test_')
    to_image_target(valid_set_target, 10, 'valid_')



to_image(os.path.join(os.path.abspath('..'), 'data/source.pkl'), os.path.join(os.path.abspath('..'), 'data/target0.7.pkl'))


