#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[1]:


import sys
sys.path.append('/home/aistudio/work/external-libraries')


# In[2]:


import paddle as paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image   
import matplotlib.pyplot as plt
import os
import pickle
import platform
import random
from imgaug import augmenters as iaa
from imgaug import random as iar


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


# !mkdir -p  /home/aistudio/.cache/paddle/dataset/cifar/
# !wget "http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz
# !mv cifar-10-python.tar.gz  /home/aistudio/.cache/paddle/dataset/cifar/
# !ls -a /home/aistudio/.cache/paddle/dataset/cifar/


# In[ ]:


# !mkdir /home/aistudio/work/external-libraries
# !pip install imgaug -t /home/aistudio/work/external-libraries


# In[3]:


class Dataloader:

    def __init__(self, data_path):
        # # 读取配置
        # option = yaml.load(open(config_path, 'r'))

        self.load_cifar10(data_path)
        self.data_augmentation(self.train_images)
        self._split_train_valid()
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
        print('\n' + '=' * 20 + ' load data ' + '=' * 20)
        print('# train data: %d' % (self.n_train))
        print('# valid data: %d' % (self.n_valid))
        print('# test data: %d' % (self.n_test))
        print('=' * 20 + ' load data ' + '=' * 20 + '\n')

    def load_cifar10(self, directory):
        # 读取训练集
        images, labels = [], []
        for filename in ['%s/data_batch_%d' % (directory, j) for j in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo,encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                # image = np.reshape(cifar10[b"data"][i], (3, 32, 32))
                # image = np.transpose(image, (1, 2, 0))
                image = cifar10[b"data"][i]
                # image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = np.array(images, dtype='float')
        labels = np.array(labels, dtype='int')
        self.train_images, self.train_labels = images, labels

        # 读取测试集
        images, labels = [], []
        for filename in ['%s/test_batch' % (directory)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo,encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                # image = np.reshape(cifar10[b"data"][i], (3, 32, 32))
                # image = np.transpose(image, (1, 2, 0))
                image = cifar10[b"data"][i]
                # image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = np.array(images, dtype='float')
        labels = np.array(labels, dtype='int')
        self.test_images, self.test_labels = images, labels

    def _split_train_valid(self, valid_rate=1.0):
        images, labels = self.train_images, self.train_labels
        thresh = int(images.shape[0] * valid_rate)
        self.train_images, self.train_labels = images[0:thresh,:], labels[0:thresh]
        self.valid_images, self.valid_labels = images[thresh:,:], labels[thresh:]
    
    #获取的是float格式0-255的array
    def get_train_data(self):
        dataset = []
        for i in range(self.train_images.shape[0]):
            dataset.append([self.train_images[i],self.train_labels[i]])
        return dataset
    
    def get_test_data(self):
        test_dataset = []
        for i in range(self.test_images.shape[0]):
            test_dataset.append([self.test_images[i],self.test_labels[i]])
        return test_dataset
    
    def get_valid_data(self):
        valid_dataset = []
        for i in range(self.valid_images.shape[0]):
            valid_dataset.append([self.valid_images[i],self.valid_labels[i]])
        return valid_dataset
    #返回的像素范围在0-1
    # def train_reader_creator(self):
    #     def reader():
    #         for d in range(self.n_train):
    #             yield (self.train_images[d].transpose(2,0,1).reshape(-1)/255.0).astype(np.float32),self.train_labels[d]
    #     return reader
    
    #返回像素为0-255
    def train_reader_creator(self):
        def reader():
            for d in range(self.n_train):
                yield self.train_images[d].transpose(2,0,1).reshape(-1),
                self.train_labels[d]
        return reader
    
    #返回的像素范围在0-1之间    
    # def test_reader_creator(self):
    #     def reader():
    #         for d in range(self.n_test):
    #             yield (self.test_images[d].transpose(2,0,1).reshape(-1)/255.0).astype(np.float32),self.test_labels[d]
    #     return reader
    
    #返回像素为0-255
    def test_reader_creator(self):
        def reader():
            for d in range(self.n_test):
                yield self.test_images[d].transpose(2,0,1).reshape(-1),
                self.test_labels[d]
        return reader
    def data_augmentation(self,images,mode='train', flip=False,
                          crop=False, crop_shape=(24, 24, 3), whiten=False,
                          noise=False, noise_mean=0, noise_std=0.01):
        
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)

        return images

    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return np.array(new_images)

    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return np.array(new_images)

    # def _image_flip(self, images):
    #     # 图像翻转
    #     for i in range(images.shape[0]):
    #         # old_image = images[i, :, :, :]
    #         # if np.random.random() < 0.5:
    #         #     new_image = cv2.flip(old_image, 1)
    #         # else:
    #         #     new_image = old_image
    #         # images[i, :, :, :] = new_image
    #         images[i,:,:,:] = np.fliplr(images[i,:,:,:])

    #     return images
    
    def _image_flip(self,images):
        result = []
        for iterm in images:
            raw_image = iterm[0]
            image = np.array(raw_image).reshape((3,32,32)).transpose(1,2,0)
            flip_image = np.fliplr(image)
            iterm[0] = flip_image.transpose(2,0,1).reshape(-1)
            result.append((iterm[0],iterm[1]))
        return result
        
    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            mean = np.mean(old_image)
            std = np.max([np.std(old_image),
                             1.0 / np.sqrt(images.shape[1] * images.shape[2] * images.shape[3])])
            new_image = (old_image - mean) / std
            images[i, :, :, :] = new_image

        return images

    def _image_noise(self, images, mean=0, std=0.1):
        # 图像噪声
        # for i in range(images.shape[0]):
        #     old_image = images[i, :, :, :]
        #     new_image = old_image
        #     for i in range(image.shape[0]):
        #         for j in range(image.shape[1]):
        #             for k in range(image.shape[2]):
        #                 new_image[i, j, k] += random.gauss(mean, std)
        #     images[i, :, :, :] = new_image
      
        for i in range(images.shape[0]):
            for i in range(images.shape[1]):
                for i in range(images.shape[2]):
                    images += random.gauss(0,0.01)
        return images


# In[ ]:


list = np.array([1,2,3])
print(list)
list1 = list.reshape((3,1))
list1[0] = 6
list1,list


# In[4]:


class Reader_creator:
    @staticmethod
    def reader_creator(dataset):
        def reader():
            for iterm in dataset:
                yield iterm[0]/255,iterm[1]
        return reader


# In[5]:


class Data_augmentation:
    @staticmethod
    def flip(dataset):
        new_dataset = []
        for iterm in dataset:
            image = np.array(iterm[0][:])
            image = image.reshape((3,32,32)).transpose(1,2,0)
            flip_image = np.fliplr(image)
            res_image = flip_image.transpose(2,0,1).reshape(-1)
            # iterm[0] = flip_image.transpose(2,0,1).reshape(-1)
            new_dataset.append((res_image,iterm[1]))
        return new_dataset
    
    
    @staticmethod
    def nope(dataset):
        new_dataset = []
        for iterm in dataset:
            image = np.array(iterm[0][:])  #iterm[0]为图像的列表形式
            # image = (image.reshape((3,32,32)).transpose(1,2,0)).astype(np.uint8)
            
            # image = (image.reshape((32,32,3))*255).astype(np.uint8)
            image = image.reshape((32,32,3)).astype(np.uint8)
            seq = iaa.Sequential([
                iaa.Noop()
                ])
            # iterm[0] = image.transpose(2,0,1).reshape(-1)
            image = seq(images=image)
            res_image = image.reshape((3,32,32))
            # res_image = (res_image.transpose(2,0,1).reshape(-1)).astype(np.float32)
            res_image = (image.reshape(-1)).astype(np.float32)
            new_dataset.append([res_image,iterm[1]])
            
            # plt.imshow((res_image.reshape((3,32,32)).transpose(1,2,0)).astype(np.uint8))
          
        return new_dataset
        
        
    #imgaug的图片通道数在最后
    @staticmethod
    def gauss_noise(dataset,mean=0, std=0.05):
        new_dataset = []
        for iterm in dataset:
            image = np.array(iterm[0][:])
            image = image.reshape((32,32,3)).astype(np.uint8)
            seq = iaa.Sequential([
                iaa.AdditiveGaussianNoise(mean,std*255,True)
                ])
            # iterm[0] = image.transpose(2,0,1).reshape(-1)
            image = seq(images=image)
            noise_image = (image.reshape(-1)).astype(np.float32)
            new_dataset.append([noise_image,iterm[1]])
        return new_dataset
    
    @staticmethod
    def crop(dataset):
        new_dataset = []
        for iterm in dataset:
            image = np.array(iterm[0][:])
            image = image.reshape((32,32,3)).astype(np.uint8)
            seq = iaa.Sequential([
                iaa.Pad(px=(0,4,4,0)),
                iaa.Crop(px=(0,4,4,0))
                ])
            # iterm[0] = image.transpose(2,0,1).reshape(-1)
            image = seq(images=image)
            noise_image = (image.reshape(-1)).astype(np.float32)
            new_dataset.append([noise_image,iterm[1]])
        return new_dataset
     
    @staticmethod
    def crop_pad(dataset):
        new_dataset = []
        for iterm in dataset:
            image = np.array(iterm[0][:])
            image = image.reshape((32,32,3)).astype(np.uint8)
            seq = iaa.Sequential([
                iaa.Pad(px=(4,4,4,4)),
                iaa.Crop(px=(4,4,4,4))
                ])
            image = seq(images=image)
            noise_image = (image.reshape(-1)).astype(np.float32)
            new_dataset.append([noise_image,iterm[1]])
        return new_dataset
    


# In[6]:


dataloader = Dataloader("work/paddle/dataset/cifar")


# In[7]:


dataset = dataloader.get_train_data()
train_data = dataset[:]
len(train_data)


# In[ ]:


valid_data = dataloader.get_valid_data()
len(valid_data)


# In[ ]:


#查看图像矩阵的值是0-255还是0-1
# for iterm in dataset:
#     print(iterm[0])
#     break


# In[8]:


test_data = dataloader.get_test_data()
len(test_data)


# In[ ]:


# for x in Reader_creator.reader_creator(dataset)():
#     print(x)
#     break


# In[ ]:


train_data[0:2]


# In[ ]:


crop_pic = Data_augmentation.crop_pad(train_data[0:2])
print(crop_pic)
for batch_id,i in enumerate(crop_pic) :
    print(batch_id,i)
    plt.imshow((i[0].reshape((3,32,32)).transpose(1,2,0)/255))
    break


# In[9]:


#查看原始图像
for batch_id,i in enumerate(train_data) :
    print(batch_id,i)
    plt.imshow((i[0].reshape((3,32,32)).transpose(1,2,0)/255))
    break


# In[10]:


fliped_data = Data_augmentation.flip(train_data)
dataset.extend(fliped_data)
len(dataset)


# In[11]:


#查看翻转后的图像
for batch_id,i in enumerate(fliped_data) :
    print(batch_id,i)
    plt.imshow((i[0].reshape((3,32,32)).transpose(1,2,0)/255))
    break


# In[ ]:


whiten_data = Data_augmentation.gauss_noise(train_data)


# In[ ]:


dataset.extend(whiten_data)
len(dataset)


# In[ ]:


#查看白化后的图片
for batch_id,i in enumerate(whiten_data) :
    print(batch_id,i)
    plt.imshow((i[0].reshape((3,32,32)).transpose(1,2,0)/255))
    break


# In[ ]:


crop_data = Data_augmentation.crop(train_data)
dataset.extend(crop_data)
len(dataset)


# In[ ]:


# noop_data = Data_augmentation.nope(train_data)
# noop_data


# In[ ]:


#查看裁剪后的图片
for batch_id,i in enumerate(crop_data) :
    print(batch_id,i)
    plt.imshow(i[0].reshape((3,32,32)).transpose(1,2,0)/255)
    break


# In[ ]:


aug_data = Data_augmentation.flip(train_data)
aug_data = Data_augmentation.gauss_noise(aug_data)
aug_data = Data_augmentation.crop_pad(aug_data)
dataset.extend(aug_data)
len(dataset)


# In[ ]:


#查看增强后的图片
for batch_id,i in enumerate(aug_data) :
    print(batch_id,i)
    plt.imshow(i[0].reshape((3,32,32)).transpose(1,2,0)/255)
    break


# In[ ]:


# BATCH_SIZE = 128
# #用于训练的数据提供器
# train_reader = paddle.batch(
#     paddle.reader.shuffle(paddle.dataset.cifar.train10(), 
#                           buf_size=128*100),           
#     batch_size=BATCH_SIZE)                                
# #用于测试的数据提供器
# test_reader = paddle.batch(
#     paddle.dataset.cifar.test10(),                            
#     batch_size=BATCH_SIZE)


# In[12]:


BATCH_SIZE = 128
#用于训练的数据提供器
train_reader = paddle.batch(
    paddle.reader.shuffle(Reader_creator.reader_creator(dataset), 
                          buf_size=128*100),           
    batch_size=BATCH_SIZE)                                
#用于测试的数据提供器
test_reader = paddle.batch(
    Reader_creator.reader_creator(test_data),                            
    batch_size=BATCH_SIZE)


# In[ ]:


# for i in train_reader():
#     print(i)
#     break


# In[ ]:


# def flip(image):
#     return np.fliplr(image)


# In[ ]:


def make_noise(image):
    for i in range(image.shape[0]):
        for i in range(image.shape[1]):
            for i in range(image.shape[2]):
                image += random.gauss(0,0.01)
    return image


# In[ ]:


# raw_image = next(train_reader())[0]
# image_array = raw_image[0].reshape(3,32,32)
# flag = raw_image[1]
# print(image_array)
# plt.imshow(image_array.transpose(1,2,0))
# print(flag)
# image = Image.fromarray((image_array.transpose(1,2,0)*255).astype(np.uint8))
# plt.imshow(image)


# In[ ]:


#测试np.pad的用法
# array1 = np.pad(array1, [[1,1], [1, 1], [1,1]], 'constant')
# array1


# In[ ]:


#构造ResNet

#创建带BN层的卷积层
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)   
    return fluid.layers.batch_norm(input=tmp,act=act)

#构造直连路径
def shortcut(input,channel_in,channel_out,stride):
    #若输入通道数与输出通道数不同，直连路径需要调整通过直连路径的输出通道数
    if channel_in != channel_out:
        return conv_bn_layer(input,channel_out,1,stride,0,None)
    return input

#构造一个基本的残差模块
def basicblock(input,channel_in,channel_out,stride):
    tmp = conv_bn_layer(input,channel_out,3,stride,1)
    tmp = conv_bn_layer(tmp,channel_out,3,1,1)
    short = shortcut(input,channel_in,channel_out,stride)
    return fluid.layers.elementwise_add(x=tmp,y=short,act='relu')

#将多个残差模块串联
def layer_warp(blockfunc,input,channel_in,channel_out,count,stride):
    tmp = blockfunc(input,channel_in,channel_out,stride)
    for i in range(1,2):
        tmp = basicblock(tmp,channel_in,channel_out,1)
    return tmp

#构造cifar10的ResNet
def resnet_cifar10(input,depth=32):
    assert (depth-2)%6 == 0
    n = (depth-2) // 6
    conv1 = conv_bn_layer(input,ch_out=16,filter_size = 3,stride=1,padding=1)
    res1 = layer_warp(basicblock,conv1,16,16,n,1)
    res2 = layer_warp(basicblock,res1,16,32,n,2)
    res3 = layer_warp(basicblock,res1,32,64,n,2)
    pool = fluid.layers.pool2d(input=res3,pool_size=8,pool_type='avg',pool_stride=1)
    droped = fluid.layers.dropout(pool,dropout_prob=0.5)
    dense = fluid.layers.fc(input=droped,size=1024,act="relu")
    predict = fluid.layers.fc(input=dense,size=10,act='softmax')
    return predict


# In[13]:


#构造ResNet

#创建带BN层的卷积层
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding=1,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)   
    return fluid.layers.batch_norm(input=tmp,act=act)

#构造直连路径
def shortcut(input,channel_in,channel_out,stride):
    #若输入通道数与输出通道数不同，直连路径需要调整通过直连路径的输出通道数
    if channel_in != channel_out:
        return conv_bn_layer(input,channel_out,1,stride,0,None)
    return input

#构造一个基本的残差模块
def basicblock(input,channel_in,channel_out,stride):
    tmp = conv_bn_layer(input,channel_out,3,stride,1)
    tmp = conv_bn_layer(tmp,channel_out,3,1,1) #残差块内的第二个卷积层步伐必定为1
    short = shortcut(input,channel_in,channel_out,stride)
    return fluid.layers.elementwise_add(x=tmp,y=short,act='relu')

#将多个残差模块串联
def layer_warp(blockfunc,input,channel_in,channel_out,count,stride):
    tmp = blockfunc(input,channel_in,channel_out,stride)
    for i in range(1,count): #tmp已经构造了一个残差块了，故从1开始
        tmp = blockfunc(tmp,channel_out,channel_out,1)
    return tmp

#构造cifar10的ResNet18
def resnet(input,block_num):
    conv1 = conv_bn_layer(input,ch_out=64,filter_size = 3,stride=1,padding=1)
    res1 = layer_warp(basicblock,conv1,64,64,block_num[0],1)
    res2 = layer_warp(basicblock,res1,64,128,block_num[1],2)
    res3 = layer_warp(basicblock,res2,128,256,block_num[2],2)
    res4 = layer_warp(basicblock,res3,256,512,block_num[3],2)
    pool = fluid.layers.pool2d(input=res4,pool_size=4,pool_type='avg',pool_stride=1)
    droped = fluid.layers.dropout(pool,dropout_prob=0.5)
    dense = fluid.layers.fc(input=pool,size=1024,act="relu")
    predict = fluid.layers.fc(input=dense,size=10,act='softmax')
    return predict


# In[ ]:


#定义一个四层卷积-池化层，三个BN层，一个全连接层的神经网络模型
def convolutional_neural_network1(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像的定义
        filter_size=2,     # 卷积核的大小
        num_filters=20,    # 卷积核的通道数
        pool_size=2,       # 池化层大小2*2
        pool_stride=2,     # 池化步长
        act="relu")        # 激活类型
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=2,
        num_filters=30,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=2,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_3 = fluid.layers.batch_norm(conv_pool_3)
    conv_pool_4 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_3,
        filter_size=2,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=conv_pool_4, size=10, act='softmax')
    return prediction


# In[ ]:


def basic_convolutional_neural_network(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像的定义
        filter_size=3,     # 卷积核的大小
        num_filters=64,    # 卷积核的通道数
        pool_size=2,       # 池化层大小2*2
        pool_stride=2,     # 池化步长
        act="relu")        # 激活类型
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=3,
        num_filters=128,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=3,
        num_filters=256,
        pool_size=2,
        pool_stride=2,
        act="relu")
    full_connection = fluid.layers.fc(input=conv_pool_3,size=1024,act='relu')
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=full_connection, size=10, act='softmax')
    return prediction


# In[ ]:


def better_convolutional_neural_network(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像的定义
        filter_size=3,     # 卷积核的大小
        num_filters=64,    # 卷积核的通道数
        pool_size=2,       # 池化层大小2*2
        pool_stride=2,     # 池化步长
        act="relu")        # 激活类型
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=3,
        num_filters=128,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=3,
        num_filters=256,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_3 = fluid.layers.batch_norm(conv_pool_3)
    full_connection = fluid.layers.fc(input=conv_pool_3,size=1024,act='relu')
    droped = fluid.layers.dropout(full_connection,dropout_prob=0.5)
    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=droped, size=10, act='softmax')
    return prediction


# In[14]:


#定义输入数据
data_shape = [3, 32, 32] #输入图片的大小为32*32,通道数为3(RGB)
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# In[15]:


# 获取分类器，用cnn进行分类
predict =  resnet(images,[2,2,2,2])


# In[16]:


# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label) # 交叉熵
avg_cost = fluid.layers.mean(cost)                            # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)       #使用输入和标签计算准确率


# In[17]:


# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法Adam
boundaries=[150,250]
values=[0.1,0.01,0.001]
optimizer =fluid.optimizer.Adam(
    learning_rate=0.001)
# optimizer = fluid.optimizer.SGD(learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries,values=values),
#             regularization=fluid.regularizer.L2Decay(regularization_coeff=5e-4))
optimizer.minimize(avg_cost)
print("完成")


# In[18]:


# 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


# In[19]:


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# In[20]:


#定义数据的提供器，feed_list的列表值必须是fluid.layers的对象
#此处规定了后面data的格式
feeder = fluid.DataFeeder( feed_list=[images, label],place=place)


# In[21]:


all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]
train_epoch_accs=[]
test_epoch_accs=[]
epoch=[]
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()


# In[ ]:


for batch_id, data in enumerate(train_reader()):
    # np.array(data).reshape((-1,3,32,32))
    print("data's length is {0}\ndata:{1}".format(len(data),data[0][0].reshape((3,32,32))))
    break


# In[22]:


# model_save_dir = "/home/aistudio/work/catdog.inference.model"
# model_save_dir = "/home/aistudio/work/ResNet_with_weight_decay"
model_save_dir = "/home/aistudio/work/ResNet_with_weight_decay_dynamic_lr"


# In[23]:


EPOCH_NUM =100


for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):                        #遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(np.array(data)),                        #喂入一个batch的数据,data是一个list
                             fetch_list=[avg_cost, acc])                    #fetch均方误差和准确率
                                                                            #fetch_list的值需是fluid.layers的对象
        
        all_train_iter=all_train_iter+BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        # print("{}'s train acc is{}".format(batch_id,train_acc))
        all_train_accs.append(train_acc[0])
        
        #每100次batch打印一次训练、进行一次测试
        if batch_id % 100 == 0:                                             
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % 
            (pass_id, batch_id, train_cost[0], train_acc[0]))
    train_epoch_accs.append(np.mean(all_train_accs))  

    # 开始测试
    test_costs = []                                                         #测试的损失值
    test_accs = []                                                          #测试的准确率
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,                 #执行测试程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_costs.append(test_cost[0])                                     #记录每个batch的误差
        test_accs.append(test_acc[0])                                       #记录每个batch的准确率

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))                         #计算误差平均值（误差和/误差的个数）
    test_acc = (sum(test_accs) / len(test_accs))                            #计算准确率平均值（ 准确率的和/准确率的个数）
    test_epoch_accs.append(np.mean(test_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
    epoch.append(pass_id+1)
# #保存模型
# # 如果保存路径不存在就创建
# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir)
# print ('save models to %s' % (model_save_dir))
# fluid.io.save_inference_model(model_save_dir,
#                               ['images'],
#                               [predict],
#                               exe)
# print('训练模型保存完成！')
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")


# In[ ]:


def plot_curve(loss_list, loss_idxs, train_precision_list, train_precision_idxs, valid_precision_list, valid_precision_idxs):
	fig = plt.figure(figsize=(12,6))
	plt.subplot(121)
	p1 = plt.plot(loss_idxs, loss_list, '.--', color='#6495ED')
	plt.grid(True)
	plt.title('cifar10 image classification loss')
	plt.xlabel('# of epoch')
	plt.ylabel('loss')
	plt.subplot(122)
	p2 = plt.plot(train_precision_idxs, train_precision_list, '.--', color='#66CDAA')
	p3 = plt.plot(valid_precision_idxs, valid_precision_list, '.--', color='#FF6347')
	plt.legend((p2[0], p3[0]), ('train_precision', 'valid_precision'))
	plt.grid(True)
	plt.title('cifar10 image classification precision')
	plt.xlabel('# of epoch')
	plt.ylabel('accuracy')
	plt.show()


# In[27]:


plot_curve(all_train_costs,all_train_iters,train_epoch_accs,epoch,test_epoch_accs,epoch)


# In[ ]:


# #保存模型
# # 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print ('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['images'],
                              [predict],
                              exe)
print('训练模型保存完成！')


# In[ ]:


infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()


# In[ ]:


def load_image(file):
        #打开图片
        im = Image.open(file)
        #将图片调整为跟训练数据一样的大小  32*32，                   设定ANTIALIAS，即抗锯齿.resize是缩放
        im = im.resize((32, 32), Image.ANTIALIAS)
        #建立图片矩阵 类型为float32
        im = np.array(im).astype(np.float32)
        #矩阵转置 
        im = im.transpose((2, 0, 1))                               
        #将像素值从【0-255】转换为【0-1】
        im = im / 255.0
        #print(im)       
        im = np.expand_dims(im, axis=0)
        # 保持和之前输入image维度一致
        print('im_shape的维度：',im.shape)
        return im


# In[ ]:


with fluid.scope_guard(inference_scope):
    #从指定目录中加载 推理model(inference model)
    [inference_program, # 预测用的program
     feed_target_names, # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
    
    infer_path='work/test_pict/'
    for i in range(10):
        absolute_path = infer_path+"{}.jpg".format(i)
        img = Image.open(absolute_path)
        plt.imshow(img)   
        # plt.show()    
        plt.pause(1)  #显示秒数
        plt.close()
        img = load_image(absolute_path)

        results = infer_exe.run(inference_program,                 #运行预测程序
                                feed={feed_target_names[0]: img},  #喂入要预测的img
                                fetch_list=fetch_targets)          #得到推测结果
        print('results',results)
        label_list = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
            "ship", "truck"
            ]
        print("infer results: %s" % label_list[np.argmax(results[0])])
    


# In[ ]:




