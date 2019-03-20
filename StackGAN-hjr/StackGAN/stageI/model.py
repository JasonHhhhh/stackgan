from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf

import sys
import os
# # Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#     import StackGAN.stageI # noqa: F401
#     __package__ = "StackGAN.stageI"
import sys
sys.path.append('../')
from misc.custom_ops import leaky_rectify
from misc.config import cfg

# 对阶段1的所有网络结构进行搭建，输入参数为shape特殊化第一阶段的GAN网络
class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM # 生成器的进来的特征的维度
        self.df_dim = cfg.GAN.DF_DIM # 鉴别器进来的特征的维度
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)


        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            # 可见 __init__ 中  可以利用self.调用在 init往后def的函数，用法
            with tf.variable_scope("d_net"):
                # 定义默认设置的D的网络的图像解码器，文本解码器，以及最终形成输出的网络格式
                self.d_encode_img_template = self.d_encode_image()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        elif cfg.GAN.NETWORK_TYPE == "simple":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image_simple()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    # 此处写了一个有embedings转化为AC的输入的特征前处理
    # 此处难道不是前处理好的？？？？？
    def generate_condition(self, c_var):
        # 建立了一个简单的全连接层，在embedings之后
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        # 最终使用leaky_ReLu
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        '''什么意思？？？第一个维度是什么？？？'''
        return [mean, log_sigma]

    # 建立生成器网络：此时输入需要拼接噪声以及CA之后的TEXT Features
    def generator(self, z_var):
        ''''''
        '''Stage1 的G'''
        # 设立计算节点 利用prettytensor建立，超级简洁，就像在搭积木：
        # 节点分为0 1 为典型的残差模块：
        '''记住一点，padding时，只有步长影响featuremap大小'''
        '''这一操作 仿佛在上采样'''
        node1_0 =\
            (pt.wrap(z_var). # 将输入张量（holder）传给warp
             flatten(). # 所有特征纵向变为1维展开然后连接一个全连接层，不影响batchsize操作：与cfg.GAN.GF_DIM 有关
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm(). #reshape，维度自行推测形成多通道
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))# -1处的维度是batchsize
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).#定制版 没有加偏置
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        '''有时通道数不同不可以直接加
        此时可以
        1X1卷积增加通道数
        也可以：zero padding'''

        # 残差网络
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).# 最近邻插值上采样
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        '''最终经过三次上采样成为最终图片'''
        return output_tensor

    def generator_simple(self, z_var):
        output_tensor =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, z_var):
        # 是‘default’还是‘simple’NET ？
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template
    ''' D0 D1 基本相同 d_encode_image+discriminator 且值在训练中使用 所以保存为template'''
    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1
    '''context_embedding
    d_encode_image
    discriminator
    直接定以一种网络格式，固定，需要使用的时候直接调用返回template，这是pt特有的方式'''
    def d_encode_image_simple(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2))

        return template

    def discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template

    def get_discriminator(self, x_var, c_var):
        '''
        :param x_var: 图形特征输入
        :param c_var: 文字特征输入，需要复制扩大
        :return: 返回一个D网络
        '''
        x_code = self.d_encode_img_template.construct(input=x_var)

        c_code = self.d_context_template.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1) #增加一维，再增加一维
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1]) #在进行扩大

        x_c_code = tf.concat( [x_code, c_code],3) #拼接

        '''结合template 形成一个网络，PT特有的方式'''
        return self.discriminator_template.construct(input=x_c_code)
