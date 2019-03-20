from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import numpy as np
import sys
import os
# # Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import StackGAN.misc # noqa: F401
    __package__ = "StackGAN.misc"

import os
import pickle
from ..misc.utils import get_image
import scipy.misc
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

#高清晰度和度清晰度的比
LR_HR_RETIO = 4
IMSIZE = 256
'''这个是什么？ Load size,相对于256做了一个放大，以这样的尺寸载入图片 有何目的？'''
LOAD_SIZE = int(IMSIZE * 76 / 64)
BIRD_DIR = '/home/jzz/hjr/gan/StackGAN-master/StackGAN/Data/birds'


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f: #r 只读 b二进制
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames

# 得到bbox文件，读入为pd.DataFrame
#
def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    print(bbox_path)
    #将bbox.txt中的参数按照whitespace分割提取 并转换格式为int
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    # print(df_bounding_boxes)
    #
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    # print(df_filenames)
    # 得到与df_bounding_boxes序号相对应的filenames aslist
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    # 创建一个字典： 有几张图就有几个keys，且keys为img_file[:-4]（从开头到倒数第四个！！！：只是去掉了‘.jpg’后缀） 用来记录类别的数组
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()
        # 这样穿不会出错 以key 对应
        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox

# inpath: 读入train的filenames.pickle的路径
# 然后从所有图的filenames和bbox信息的dict中拿出train 或者test的图片信息
# 输出作为以pick方式保存的list nested with np.arrays（RGB）（描述训练激活测试机内的所有图片）
def save_data_list(inpath, outpath, filenames, filename_bbox):
    hr_images = []
    lr_images = []
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    cnt = 0
    '''此处的filename 是一个list，存了所有训练集图片的名字，不带.jpg,这就是之前在获取所有图片的key的时候要求filenames的[:-4]的原因'''
    # 对所有训练集上的图片做这样的操作
    for key in filenames:
        #得到bbox
        bbox = filename_bbox[key]
        #inpath = train or test
        f_name = '%s/CUB_200_2011/images/%s.jpg' % (inpath, key)
        # 输入im地址，将其转化为一个np.array 3 通道 RGB， 这样的方法接受bbox操作
        img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox)
        img = img.astype('uint8')
        hr_images.append(img)
        # a=img.shape
        lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic')
        lr_images.append(lr_img)
        '''这样的记录习惯很好'''
        cnt += 1
        if cnt % 100 == 0:
            print('Load %d......' % cnt)

    # 表示已经读取完成 所有的给定的filenames的高分辨率低分辨率的图片都已经读取完成并且保存在  hr_images lr_images 中
    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    # 将hr_images写入，以pickle的形式
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    '''pickle:一种保存数据的方式，高维数组 字典等等'''
    # 写入或者某个文件：进入 关闭  所以要with open(outfile, '') as f_out: 创建了outfile且在with下以f_out名称调用
    with open(outfile, 'wb') as f_out:
        # pickle.dump 写入
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)


def convert_birds_dataset_pickle(inpath):
    # Load dictionary between image filename to its bbox
    filename_bbox = load_bbox(inpath)
    # ## For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames, filename_bbox)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames, filename_bbox)


if __name__ == '__main__':
    convert_birds_dataset_pickle(BIRD_DIR)
