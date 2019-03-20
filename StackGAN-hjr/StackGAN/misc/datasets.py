from __future__ import division
from __future__ import print_function


import numpy as np
import pickle
import random

'''给我images的数组以及其他如下，
我将为你定义一种准备数据的方法如下，注意：你需要给出imsize，来获得64，或者是256的图片用于不同的D的训练'''
class Dataset(object):
    # 整个数据集level的信息
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        '''此处的embeddings 涵盖所有images中的所有字母对应的特征： size：图片数，句子数，特征长度 '''
        self._embeddings = embeddings #这里的embedings是与选用cnn-rnn做出来的特征 对应到 TEXT文件夹每个目录下的txt中的不同字母

        self._filenames = filenames
        # 需要建立一个目录存储数据的呀什么的 即：/Data/birds 或者 /Data/flowers
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        '''打乱了的ID 0 1 2 3 排序'''
        self._saveIDs = self.saveIDs()
        self._aug_flag = aug_flag

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    '''property:装饰起的作用是什么？？？对一个class下定义的属起到监控作用
    在绑定属性时，如果我们直接把属性暴露出去，虽然写起来很简单，但是，没办法检查参数，导致可以把成绩随便改：
    s = Student()
    s.score = 9999
    这显然不合逻辑。为了限制score的范围，可以通过一个set_score()方法来设置成绩，再通过一个get_score()来获取成绩
    但是这样做会很复杂！你需要对每个属性都做这样的操作！'''

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property

    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    '''对所有的照片进行shuufle操作：
    产生一个与i长度相当的ID数组
    randomshuffle之  这一样操作._saveIDs
    根据什么？：根据你类的基本属性：images的长度'''
    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs

    '''给我一个filename和对应的class_id（后续调用会用for循环对所有做处理） 
     我会返回相应的字符描述，为一个list 有很多句！！！但意思相同
     问题？？？？：鸟儿也是这个函数？'''
    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # 是否存在jpg/这四个字符？？？即：是否在前处理中已经将描述解压？
            # 是的话进行如下操作
            class_name = 'class_%05d/' % class_id #五位整数
            name = name.replace('jpg/', class_name)

        # name： 剔除jpg/换作 class_id 这样才能访问到对的text目录
        cap_path = '%s/text_c10/%s.txt' %\
                   (self.workdir, name)
        # 打开text文件 并且读取其中所有的描述 按行
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        # 返回一个LIST
        return captions

    '''将所有的照片（之前比这个尺寸大）尺寸全部随即裁剪为256 64 到底是256 还是 64 这个要看你怎么定义类了，你对定义Dataset256那么就是处于stage2的训练了
    输入images 为 ：np数组 4D （你给几张数据我帮你裁剪几张且反转） 且包含翻转操作'''
    def transform(self, images):


        if self._aug_flag:
            '''如果进行增强。即前处理'''
            # 创建一个空的矩阵
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 3])

            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                # 确定裁剪的像素的起始位置
                h1 = np.floor((ori_size - self._imsize) * np.random.random())
                w1 = np.floor((ori_size - self._imsize) * np.random.random())
                # cropped_image =\
                #     images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                # 索引必须是int
                original_image = images[i]
                cropped_image = original_image[int(w1): int(w1 + self._imsize),
                                int(h1): int(h1 + self._imsize),:]
                # 随即翻转图像 上下
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images


    '''给我一个batch的embeddings，以及其filenames——list和classID-》用来调 readCaptions 读取text中的描述语句
    这个embeddings包含了bacth中每一张图的很多句描述？？？
    我给你返回一个随机选择sample_num句的embeddings特征2D batchsize*vctorlenth'''
    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        # 如果只有一句话！！！！！！！！！！！！！！！！ 或者只有2D 直接squezze
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # embedding_num：有几句话就对几句做了embedding 从这几句中选几句？
            # _为embedding的特征向量的长度的数目
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            # 对batch中的每个做如下操作
            for i in range(batch_size):

                # # 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
                # a1 = np.random.choice(a=5, size=3, replace=False, p=None)

                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1: #如果只选一句话！！！！！！！
                    randix = int(randix)
                    # 对batch中的第i个： 读取其字幕
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else: #如果选了两句话
                    e_sample = embeddings[i, randix, :]
                    # 第i个特征图，每个图取 randix索引的行， 这些行的所有列都保留
                    # e_mean 为 一个text内的sample之后的embeddings 2D
                    # 此时我要沿着列求平均保持特征向量唯独不变 最终只提炼出来一个1维的向量 但是仍然有三个：【】
                    e_mean = np.mean(e_sample, axis=0)
                    # 录入sampled_embeddings for之后形成了整个batch的sample_embeddings
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            # 去掉多余维度：框【】 去掉了中间行（不同句子的）的维度
            return np.squeeze(sampled_embeddings_array), sampled_captions

    '''执行这句代表，我要为下一个batch提取数据了这将是下一个batch所需要的所有数据，输出一个list，有：
    sampled_embeddings, sampled_captions，sampled_images（前三个是同一张图（batch）的！！！）labels（这个labels是干嘛用的？？？？？），sampled_wrong_images
    输入为：
    定义好的Dataset大类：给我训练或测试的所有图片，对应的
    batch大小
    window：你要几句话我就从对应的text中提出几句话
    images, imsize, 
    embeddings,filenames, workdir,labels, aug_flag,class_id, class_range'''
    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        '''给我一个起点，即上一个batch的终点（末尾剔除---》》0~start-1）
        我返回一个what？'''
        # _index_in_epoch 初值记为 len（_num_examples）
        # 也就是在刚开始训练的时候就会创建以下
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #判刑一个epoch是否完成
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            # 每完成一个Epoch 即加一
            self._epochs_completed += 1
            # 美国一个epoch都需要将所有训练样本shuffle
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)
            # 开始下一个 epoch
            # Start next epoch
            # 设置起点
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        # 设置重点
        end = self._index_in_epoch

        # shuffle过的ID列队中拿出 start:end 这一batch的索引 （取真图）
        current_ids = self._perm[start:end]
        # _num_examples 是 生成数据集的类的超参images的第一维有多少！即有几张图片！！！
        # 这句话产生了错图！！！！！
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        # 所有训练的图对应的class 我们不希望 （错图，对的描述） 和 （真确图，正确描述）成为一类！！！（在同一索引处）
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        # 再找到的冲突的索引处将去对应的ID改为之前的100~200之间的随机数
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples
        # 取出真图
        sampled_images = self._images[current_ids]
        # 取出错图
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        # 改变数据类型
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        # 将数据缩放到-1~1
        # 先缩放到0~2 再平移到 -1~1
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.
        # 取代的不是64 256 我们需要进行裁剪
        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_wrong_images]

        if self._embeddings is not None:
            # 通过上述我们得到了这批训练的图片的编号，我们可以依次从_filenames和_class_id中取出
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            # 调用 sample_embeddings：
            # 给我一个batch的embeddings，以及其filenames——list和classID
            # 我给你返回一个随机选择sample_num句的embeddings特征2D batchsize*vctorlenth
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, class_id, window) # window是取多少句话？？？？？
            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)
        return ret_list

    '''给我batch_size，start点（在一个epoch中的），max_captions
    我返回一个[sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]
                ？？？不太懂：self._saveIDs[start:end] 干嘛用的？大概是最后检测用的把看效果用的？'''
    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        # 我执行这一个batch会超出epoch吗？ 如果超出了我下一个就不是默认的batchsize了！！！
        # 但是余下的例子我还想跑， 因此：？？？？？？什么意思？意义何在
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            #计算出我这一步的batchsize
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)

        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []

        sampled_captions = []
        sampled_filenames = self._filenames[start:end]
        sampled_class_id = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            # print(captions)
            # 拿到了这一batch的所有图的所有captions
            sampled_captions.append(captions)
        # 对每一张图的captions  我们要拿出其中max_captions个，数量不够？全拿出来
        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return [sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]


'''这个类生成了一系列方法，来定义数据集（返回一个dataset类）：给我images  embeddings，im.shape,filenames_list,class list
即：这样一来，我就创建了关于    birds或者flower  这一数据集的每个batch的数据读取操作：包括真，错图片，embedings，
必须TextDataset(object)：工作目录# 需要一个目录 即：/Data/birds 或者 /Data/flowers,
文字embedding来源是哪里
stage1 2 处理的图像大小比例 default=4 ？'''
class TextDataset(object):
    def __init__(self, workdir, embedding_type, hr_lr_ratio):
        lr_imsize = 64
        self.hr_lr_ratio = hr_lr_ratio
        if self.hr_lr_ratio == 1:
            self.image_filename = '/76images.pickle'
        elif self.hr_lr_ratio == 4:
            self.image_filename = '/304images.pickle'

        self.image_shape = [lr_imsize * self.hr_lr_ratio,
                            lr_imsize * self.hr_lr_ratio, 3]
        # 一张图有多少个数表示
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None
        self.workdir = workdir
        if embedding_type == 'cnn-rnn':
            self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            self.embedding_filename = '/skip-thought-embeddings.pickle'

    '''使用这个方法，我需要你先给我我的pickle（存储了相片和embeddings）所在的路径'''
    def get_data(self, pickle_path, aug_flag=True):
        # 打开照片存入 images 中！！！！！！！！！！！！
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f,encoding='bytes')
            images = np.array(images)
            print('images: ', images.shape)
        # 打开embeddings 存入 embeddings中！！！
        with open(pickle_path + self.embedding_filename, 'rb') as f:
            embeddings = pickle.load(f,encoding='bytes')
            embeddings = np.array(embeddings)
            self.embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        # 打开filenames_list 存入 list_filenames！！！
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f,encoding='bytes')
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        # class_info 存入 class_id！！！
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f,encoding='bytes')

        return Dataset(images, self.image_shape[0], embeddings,
                       list_filenames, self.workdir, None,
                       aug_flag, class_id)
