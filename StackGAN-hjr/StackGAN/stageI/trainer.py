from __future__ import division
from __future__ import print_function

import sys
import os
# # Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import StackGAN.stageI # noqa: F401
    __package__ = "StackGAN.stageI"

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc

from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar

import sys
sys.path.append('../')

from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

'''建立这样一个类，通过一些超参定义gan的训练方式'''
class CondGANTrainer(object):
    ''''''
    '''在初始化中定义必须的参数：
    model类，之前定义好的CondGAN类
    dataset类，之前定义好的数据集大类;
    '''
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL
        # 记录变量，（变量，名字），为了summary将所有变量可视化，保存
        self.log_vars = []

        '''定义类属性，实现def之间的交互'''

        '''建立数据入口：
        真图，假图入口，文本入口
        通过定义该类的新的属性而实现'''
    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    '''给定embedings，返回ca操作提取的特征，以及loss中的一项： KL项
    以batch为单位'''
    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        '''产生embedings'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        # c_mean的格式：全连接层出来后 一半一半
        mean = c_mean_logsigma[0] # nested 的 list 直接拿出 mean 矩阵
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            #随机数 作随机采样
            epsilon = tf.truncated_normal(tf.shape(mean))
            # 由于经过了relu不是0-1是一个logit 进行
            stddev = tf.exp(c_mean_logsigma[1])

            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0
        # 输出最终采样值和loss cfg.TRAIN.COEFF.KL= TRUE 则表示我会在训练中加入这一loss项
        return c, cfg.TRAIN.COEFF.KL * kl_loss

    '''初始化操作: 在建立一张图 并且给出loss计算方法
    调用入口
    调用g网络
    产生假图
    进入D（fixed）计算LOSS
    '''
    def init_opt(self):
        self.build_placeholder()
        # 建立这样一个域 告诉别人：我在训练模型
        with pt.defaults_scope(phase=pt.Phase.train):
            # 开始的产生假图
            with tf.variable_scope("g_net"):
                # ####get output from G network################################
                # 由入口给我文本特征我来用self.sample_encoded_context合成ca特征以及计算出KLloss
                # Z ：给我batchsize 和 Z的维度 我来产生 噪声
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                # 记录 拼接
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                # 拼接后送入之前定义好的函数get_generator 输出假图
                '''ok 一个batch的假图已经形成'''
                fake_images = self.model.get_generator(tf.concat([c, z],1))
            '''每个batch中，给定假图，真图，text，即可算出loss，由compute_losses计算
            ：分别给出D 以及 G的 LOSS
            '''
            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings)
            # G的loss要加这一项 kl_loss， 但是discriminator_loss不需要
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()
        # 建立这样一个域 告诉别人：我在测试模型
        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("g_net", reuse=True):
                self.sampler()
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")
    # ???????
    def sampler(self):
        c, _ = self.sample_encoded_context(self.embeddings)
        if cfg.TRAIN.FLAG:
            z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        else:
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat([c, z],1))
    '''给我一批数据(dataset)， 我会计算出总体的loss（主网络的）
    需要弄清楚数据的来源：fake wrong real ims ， embeddings 
    ？？？？？？？？？？如何使用？'''
    def compute_losses(self, images, wrong_images, fake_images, embeddings):
        ''''''
        '''三种数据类型：
        真图，真文
        假图，真文
        不匹配的真图（wrong），真文
        注意：任何时候喂入的数据 设计的函数 一般都是以batch为单位设计'''
        real_logit = self.model.get_discriminator(images, embeddings)
        wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
        fake_logit = self.model.get_discriminator(fake_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logit,
                                                    labels = tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits = wrong_logit,
                                                    labels = tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logit,
                                                    labels = tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)

        '''一般都会加入：Wrong的训练
        为什么会有1/2 很自然的：他俩都是负样本，real为正样本，希望正负样本多loss的贡献占比平均'''
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_fake", fake_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(logits =fake_logit,
                                                    labels =tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables() # 给出所有的训练的变量
        # 所有变量命名格式 g_ d_ 来区分 G D 变量
        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]
        # 定义G的优化op：
        # 先定义优化器
        generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                               beta1=0.5)
        #好，我现在的定义好了G的训练器 （优化器的选择， 训练的loss目标， 以及训练的变量（其他变量不求导，fixed））
        self.generator_trainer =\
            pt.apply_optimizer(generator_opt,
                               losses=[generator_loss],
                               var_list=g_vars)
        discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                   beta1=0.5)
        self.discriminator_trainer =\
            pt.apply_optimizer(discriminator_opt,
                               losses=[discriminator_loss],
                               var_list=d_vars)
        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.hist_sum = tf.summary.merge(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img,1))
        imgs = tf.expand_dims(tf.concat(stacked_img,0), 0)
        current_img_summary = tf.summary.image(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train = \
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test = \
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat([superimage_train, superimage_test],0)
        self.image_summary = tf.summary.merge([fake_sum_train, fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary =\
            sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/train.jpg' % (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/test.jpg' % (self.log_dir), gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.initialize_all_variables())

        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            restore_vars = tf.all_variables()
            # all_vars = tf.all_variables()
            # restore_vars = [var for var in all_vars if
            #                 var.name.startswith('g_') or
            #                 var.name.startswith('d_')]
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=2)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

                keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch != 0:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        # training d
                        images, wrong_images, embeddings, _, _ =\
                            self.dataset.train.next_batch(self.batch_size,
                                                          num_embedding)
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        # train d
                        feed_out = [self.discriminator_trainer,
                                    self.d_sum,
                                    self.hist_sum,
                                    log_vars]
                        _, d_sum, hist_sum, log_vals = sess.run(feed_out,
                                                                feed_dict)
                        summary_writer.add_summary(d_sum, counter)
                        summary_writer.add_summary(hist_sum, counter)
                        all_log_vals.append(log_vals)
                        # train g
                        feed_out = [self.generator_trainer,
                                    self.g_sum]
                        _, g_sum = sess.run(feed_out,
                                            feed_dict)
                        summary_writer.add_summary(g_sum, counter)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_sum, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def save_super_images(self, images, sample_batchs, filenames,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        numSamples = len(sample_batchs)
        for j in range(len(filenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            superimage = [images[j]]
            # cfg.TRAIN.NUM_COPY samples for each text embedding/sentence
            for i in range(len(sample_batchs)):
                superimage.append(sample_batchs[i][j])

            superimage = np.concatenate(superimage, axis=1)
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, filenames, _ =\
                dataset.next_batch_test(self.batch_size, start, 1)
            print('count = ', count, 'start = ', start)
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                for j in range(np.minimum(16, cfg.TRAIN.NUM_COPY)):
                    samples = sess.run(self.fake_images,
                                       {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                self.save_super_images(images, samples_batchs,
                                       filenames, i, save_dir,
                                       subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
