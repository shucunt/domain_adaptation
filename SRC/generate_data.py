# -*- coding: utf-8 -*-
'''
将target数据和st数据存在文件中

======================================
target数据是根据source和bsds500中数据做差而成的
其中，在做差时，如果像素点是数字上的像素点，不进行做差处理
如果像素点是背景上的像素点，使用如下公式处理
target = |source - bsds500 * theta|

target包含train（50000），valid（10000），test（10000）三组
每组中包含括号中个数的样本
每个样本包含3部分，第一部分是图片，一共28*28个值，代表一张图片
第二部分是图片的标签，是一个0-9的值，表示图片中的数字是几
第三部分是图片的域标签，是一个0或1的值，0表示是源域，1表示是目标域，target集合中所有域标签都是1

======================================
st数据是source和target的混合数据
混合方式是source和target交替存储，一样一个
st数据集包含train（1000000），valid（20000），test（20000）三组
其中每个样本结构和target样本结构完全相同

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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from PIL import Image
from scipy.misc import imsave

theta = 0.7

def load_data(dataset, theta):
    # dataset是数据集的路径，程序首先检测该路径下有没有MNIST数据集，没有的话就下载MNIST数据集
    #这一部分就不解释了，与softmax回归算法无关。
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        
        urllib.urlretrieve(origin, dataset)

    print '... loading data'
    
#以上是检测并下载数据集mnist.pkl.gz，不是本文重点。下面才是load_data的开始
    
#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
#主要用到python里的gzip.open()函数,以及 cPickle.load()。
#‘rb’表示以二进制可读的方式打开文件
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def cpickle_dataset(valid_set, train_set, test_set, theta, borrow = True):
        #loc1 is the location of t_data
        #loc2 is the location of st_data
        print '...train data'
        train_source_x, train_source_y = train_set
        
        train_target_x = []
        train_source_label = numpy.zeros(train_source_y.size)
        train_target_label = numpy.ones(train_source_y.size)
        fileList = []
        fileList = os.listdir(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/'))
        fileList.remove("Thumbs.db")
        
        '''
        从bsds500中随机选一张图片，将图片转化为灰度图像，
        将像素对应的位置的灰度值相加，存到data_x中
        data_x中是为了应用在训练中，在那一份代码中会返回源域图像和目标域图像
        '''
        for k in range(train_source_x.shape[0]):
            if k % 10000 == 0:
                print k
            indx = random.randint(0, len(fileList) - 1)
            img = Image.open(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/' + fileList[indx]))
            img1 = img.convert("L")
            temp = [0] * 28 * 28
            for i in range(28):
                for j in range(28):
                    temp[i * 28 + j] = train_source_x[k][i * 28 + j]
                    if(train_source_x[k][i * 28 + j] < 0.1 ):
                        temp[i * 28 + j] -= ((img1.getpixel((i,j)) / 255.0) * theta)
                        temp[i * 28 + j] = abs(temp[i * 28 + j])
            train_target_x.append(temp)
        train_target_x_np = numpy.array(train_target_x)
        train_target = []
        train_target.append(train_target_x_np)
        train_target.append(train_source_y)
        train_target.append(train_target_label)
        train_source_y_list = train_source_y.tolist()

        
        print '...valid data'
        valid_source_x, valid_source_y = valid_set
        
        valid_target_x = []
        valid_source_label = numpy.zeros(valid_source_y.size)
        valid_target_label = numpy.ones(valid_source_y.size)
        fileList = []
        fileList = os.listdir(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/'))
        fileList.remove("Thumbs.db")
        
        '''
        从bsds500中随机选一张图片，将图片转化为灰度图像，
        将像素对应的位置的灰度值相加，存到data_x中
        data_x中是为了应用在训练中，在那一份代码中会返回源域图像和目标域图像
        '''
        for k in range(valid_source_x.shape[0]):
            if k % 10000 == 0:
                print k
            indx = random.randint(0, len(fileList) - 1)
            img = Image.open(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/' + fileList[indx]))
            img1 = img.convert("L")
            temp = [0] * 28 * 28
            for i in range(28):
                for j in range(28):
                    temp[i * 28 + j] = valid_source_x[k][i * 28 + j]
                    if(valid_source_x[k][i * 28 + j] < 0.1 ):
                        temp[i * 28 + j] -= ((img1.getpixel((i,j)) / 255.0) * theta)
                        temp[i * 28 + j] = abs(temp[i * 28 + j])
            valid_target_x.append(temp)
        valid_target_x_np = numpy.array(valid_target_x)
        valid_target = []
        valid_target.append(valid_target_x_np)
        valid_target.append(valid_source_y)
        valid_target.append(valid_target_label)
        valid_source_y_list = valid_source_y.tolist()

        print '...test data'
        test_source_x, test_source_y = test_set
        
        test_target_x = []
        test_source_label = numpy.zeros(test_source_y.size)
        test_target_label = numpy.ones(test_source_y.size)
        fileList = []
        fileList = os.listdir(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/'))
        fileList.remove("Thumbs.db")
        
        '''
        从bsds500中随机选一张图片，将图片转化为灰度图像，
        将像素对应的位置的灰度值相加，存到data_x中
        data_x中是为了应用在训练中，在那一份代码中会返回源域图像和目标域图像
        '''
        for k in range(test_source_x.shape[0]):
            if k % 10000 == 0:
                print k
            indx = random.randint(0, len(fileList) - 1)
            img = Image.open(os.path.join(os.path.abspath('..'),'data/BSR/BSDS500/data/images/train/' + fileList[indx]))
            img1 = img.convert("L")
            temp = [0] * 28 * 28
            for i in range(28):
                for j in range(28):
                    temp[i * 28 + j] = test_source_x[k][i * 28 + j]
                    if(valid_source_x[k][i * 28 + j] < 0.1 ):
                        temp[i * 28 + j] -= ((img1.getpixel((i,j)) / 255.0) * theta)
                        temp[i * 28 + j] = abs(temp[i * 28 + j])
            test_target_x.append(temp)
        '''
        f = open(os.path.abspath(), 'w')
        for i in range(len(test_target_x)):
            for j in range(28 * 28):
                f.write(str(test_target_x[i][j]))
                f.write('\t')
            f.write('\n')
        f.close()
        '''
        test_target_x_np = numpy.array(test_target_x)
        test_target = []
        test_target.append(test_target_x_np)
        test_target.append(test_source_y)
        test_target.append(test_target_label)
        test_source_y_list = test_source_y.tolist()

        target = []
        target.append(train_target)
        target.append(valid_target)
        target.append(test_target)
        cPickle.dump(target, open(os.path.join(os.path.abspath('..'), 'data/target' + str(theta) + '.pkl'), 'wb'))

        st = []
        train_st = []
        valid_st = []
        test_st = []

        train_st_x = []
        train_st_y = []
        train_st_label = []
        for i in range(len(train_target_x)):
            train_st_x.append(train_target_x[i])
            train_st_x.append(train_source_x[i])
            train_st_y.append(train_source_y_list[i])
            train_st_y.append(train_source_y_list[i])
            train_st_label.append(float(1.0))
            train_st_label.append(float(0.0))
        train_st_x_np = numpy.array(train_st_x)
        train_st_y_np = numpy.array(train_st_y)
        train_st_label_np = numpy.array(train_st_label)
        print train_st_x_np.shape
        print train_st_y_np.shape
        print train_st_label_np.shape
        train_st.append(train_st_x_np)
        train_st.append(train_st_y_np)
        train_st.append(train_st_label_np)

        valid_st_x = []
        valid_st_y = []
        valid_st_label = []
        for i in range(len(valid_target_x)):
            valid_st_x.append(valid_target_x[i])
            valid_st_x.append(valid_source_x[i])
            valid_st_y.append(valid_source_y_list[i])
            valid_st_y.append(valid_source_y_list[i])
            valid_st_label.append(float(1.0))
            valid_st_label.append(float(0.0))
        valid_st_x_np = numpy.array(valid_st_x)
        valid_st_y_np = numpy.array(valid_st_y)
        valid_st_label_np = numpy.array(valid_st_label)
        print valid_st_x_np.shape
        print valid_st_y_np.shape
        print valid_st_label_np.shape
        valid_st.append(valid_st_x_np)
        valid_st.append(valid_st_y_np)
        valid_st.append(valid_st_label_np)

        test_st_x = []
        test_st_y = []
        test_st_label = []
        for i in range(len(test_target_x)):
            test_st_x.append(test_target_x[i])
            test_st_x.append(test_source_x[i])
            test_st_y.append(test_source_y_list[i])
            test_st_y.append(test_source_y_list[i])
            test_st_label.append(float(1.0))
            test_st_label.append(float(0.0))
        '''
        f = open(os.path.abspath(), 'w')
        for i in range(len(test_target_x)):
            for j in range(28 * 28):
                f.write(str(test_st_x[i][j]))
                f.write('\t')
            f.write('\n')
        f.close()
        '''
        test_st_x_np = numpy.array(test_st_x)
        test_st_y_np = numpy.array(test_st_y)
        test_st_label_np = numpy.array(test_st_label)
        print test_st_x_np.shape
        print test_st_y_np.shape
        print test_st_label_np.shape
        test_st.append(test_st_x_np)
        test_st.append(test_st_y_np)
        test_st.append(test_st_label_np)

        st.append(train_st)
        st.append(valid_st)
        st.append(test_st)

        cPickle.dump(st, open(os.path.join(os.path.abspath('..'), 'data/st' + str(theta) + '.pkl'), 'wb'))




    cpickle_dataset(valid_set, train_set, test_set, borrow = True)

dataset=os.path.join(os.path.abspath('..'), 'data/mnist.pkl.gz')
load_data(dataset, theta)