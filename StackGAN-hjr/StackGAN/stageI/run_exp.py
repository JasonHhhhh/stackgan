from __future__ import division
from __future__ import print_function


import dateutil.tz
import datetime
import argparse
import pprint

import sys
import os
# Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#     import StackGAN.stageI # noqa: F401
#     __package__ = "StackGAN.stageI"

sys.path.append("../")
print (sys.path)

from misc.datasets import TextDataset
from stageI.model import CondGAN
from stageI.trainer import CondGANTrainer
from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file


def parse_args():
    # 创建一个解释器并且给他命名 或者 叙述
    parser = argparse.ArgumentParser(description='Train a GAN network')
    # config file 预先设置好的超参数列表
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    # 使用那个GPU
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=2, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 引入解释器
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1: #不等于
        cfg.GPU_ID = args.gpu_id
    # 打印接下来训练的配置
    print('Using config:')
    pprint.pprint(cfg)
    '''记录时间 先不看'''
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = '/home/jzz/hjr/gan/StackGAN-master/StackGAN/Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 1)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)

        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondGAN(
        image_shape=dataset.image_shape
    )

    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        algo.evaluate()
