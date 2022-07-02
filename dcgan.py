#%% 选择GPU
import datetime
import os
import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from OMP.OMP import OMP
from RFD.RFD import RFD  # 导入局部聚焦误差函数
print(tf.__version__)
print("Eager execution: {}".format(tf.executing_eagerly()))
print('this is the first version of my github')

#设置gpu
# print(tf.test.is_gpu_available())#测试GPU是否可用
# physical_devices = tf.config.experimental.list_physical_devices('GPU') #列出GPU的信息
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.enable_eager_execution(config=config)
# layers = tf.keras.layers

# import tensorflow.compat.v1 as tf #使用1.0版本的方法
# tf.disable_v2_behavior() #禁用2.0版本的方法

# #TensorFlow按需分配显存
# config.allow_soft_placement = True
# config.log_device_placement = True

# # #指定显存分配比例
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

#%% 数据预处理
# data_root=Path('xxx') # 创建文件对象
# all_data_path=[str(item) for item in data_root.iterdir()] #将文件夹中的所有数据名称转化为字符串
# data_ds=[np.loadtxt(item) for item in all_data_path] #导入上述字符串所对应的txt or mat or sth

#######################读取构造的0.1s发射信号(无论是仿真还是真实的信道用的都是这个发射信号)#######################################
sig_src_init = h5py.File('data/sig_src2.mat',mode='r')
sig_src = sig_src_init['sig_src2']
sig_src = np.array(sig_src,dtype=np.float32)
sig_src = sig_src.T
# sig_src = sig_src.reshape(sig_src.shape[0],sig_src.shape[1],1)

####################### 读取构造的0.1s接收信号（发射信号经过真实信道后加4-8dB的高斯白噪声扩充）*600 #############################

data_init = h5py.File('data/enhance_true_sig2_2_real.mat',mode='r')
data_ds = data_init['enhance_true_sig2_2_real']
# print(data_init.values())
data_ds = np.array(data_ds,dtype=np.float32) #将这些数据转化成nparray
data_ds = data_ds.T#数据转置(非共轭转置)
data_ds = data_ds.reshape(data_ds.shape[0],data_ds.shape[1],1) #reshape数据维度，变成三维的数据

#######################读取（0.1s扩充的接收信号通过与omp的信道估计） 600 #######################################
constructed_signal_channel_estimation_init =  h5py.File('data/gouzao520_c_final.mat',mode='r')
constructed_signal_channel_estimation = constructed_signal_channel_estimation_init['gouzao520_c_final']
constructed_signal_channel_estimation = np.array(constructed_signal_channel_estimation,dtype=np.float32)
constructed_signal_channel_estimation = constructed_signal_channel_estimation.T#数据是2dims

####################### 读取（仿真信道接收信号）*600，给G的输入#######################################
noise_init = h5py.File('data/Sig_Sim.mat',mode='r')
noise = noise_init['Sig_Sim']
noise = np.array(noise,dtype=np.float32)
noise = noise.T#数据是2dims
# noise = noise.reshape(noise.shape[0],noise.shape[1],1)

#######################读取（仿真信道冲激响应由bellhop生成）600 #######################################
# sim_H_init = h5py.File('data/H_sim.mat',mode='r')
# sim_H = sim_H_init['H_sim']
# sim_H = np.array(sim_H,dtype=np.float32)
# sim_H = sim_H.T

#######################读取真实信道（由1s 实验cw信号估计出来的信道）#######################################
#
Real_impulse_response_init =  h5py.File('data/or2_c_final.mat',mode='r')
# Real_impulse_response = Real_impulse_response_init['or2_c_final']
# Real_impulse_response = np.array(Real_impulse_response,dtype=np.float32)
# data_ds=data_ds/np.max(data_ds)#数据归一化，可以换处理方法。
# data_ds.shape

#%% hyperparameters
SEQUNENCE_LENGTH= 9600 #样本长度
FEATUERE_DIMENTIONS = 1 #样本维度
LATENT_DIM = 5473 #隐变量维度，即G输入的维度
n_classes = 1 #类别个数
EPOCHS = 10000
BATCH_SIZE = 1 #实时训练

#%% 创建GAN

class GAN:
    def __init__(self):#创建一个实例的时候，__init__就会被调用
        self.OMP = OMP()
        self.rfd = RFD() #RFD类下面的函数调用方法，cal_RFD(self,data_standard,data_other,dist0)
        self.sequence_length = SEQUNENCE_LENGTH #样本长度,在超参数里面
        self.feature_dimentions = FEATUERE_DIMENTIONS #样本维度为1
        self.data_shape = (self.sequence_length, self.feature_dimentions)
        self.latent_dim = LATENT_DIM #噪声长度,如果用自己的数据的话需要修改。
        # self.optimizer = Adam(0.0001, 0.5)#优化器参数lr和beta
        #########D结构#############
        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)
        # self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer,
        #                            metrics=['accuracy'])#判别器构造
        #########G结构#############
        self.generator = self.build_generator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4,0.5)
        z = Input(shape=(self.latent_dim,)) # 定义一个输入层，输入维度是噪声的维度
        data = self.generator(z)#给G输入噪声,data是G的输出
        self.discriminator.trainable = True #为了组合模型（此时组合的模型是G+D只训练G）
        self.generator.trainable = True
        validity = self.discriminator(data) #判别器将由噪声生成的数据作为输入并确定有效性

        #########组合模型(i.e.把D和G堆叠起来)#############
        # self.combined = Model(z,validity) #训练生成器骗过判别器
        # self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        # self.generator.summary()#G模型参数打印
        # self.discriminator.summary()#D模型参数打印
        # self.combined.summary()#打印组合的模型参数

    def build_generator(self):
        model = Sequential() #根据数据复杂度修改该模型层数
        #         model.add(Dense(256, input_dim=self.latent_dim))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Dense(512))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Dense(1024))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Dense(64, input_dim=self.latent_dim))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(200,input_shape=(self.latent_dim,)))#200
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(Reshape((200, 1)))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'))
        model.add(UpSampling1D())#400
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling1D())#800
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        # model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(UpSampling1D())#1600
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=6, kernel_size=8, padding="same", activation='relu'))#9600
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Reshape((SEQUNENCE_LENGTH,1)))
        # model.add(Dense(np.prod(self.data_shape), activation='tanh'))
        # model.add(Reshape(self.data_shape))
        # model.summary()
        return model
        # noise = Input(shape=(self.latent_dim,))#构建输入层，输入层输入的是noise
        # data = model(noise)#输入噪音，输出G生成的数据
        # return Model(noise, data) #返回模型
    #构建判别器
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.data_shape))#该层输出(None, )None是batch_size
        # model.add(Flatten(input_shape=(self.data_shape,1)))#该层输出(None, )None是batch_size
        # model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(self.sequence_length, 1)))
        # model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same', input_shape=(self.sequence_length, 1)))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.5))
        # model.add(Dense(64))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((SEQUNENCE_LENGTH,1)))#real data的长度
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Dropout(0.25))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(128, activation='relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))#该层输出(None, 128)
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128,activation='relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(n_classes))
        # model.summary()
        #         model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(self.sequence_length, 1)))
        #         model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same', input_shape=(self.sequence_length, 1)))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(Dropout(0.25))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(Dropout(0.25))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
        #         # model.add(Conv1D(filters=64, kernel_size=3, strides=2, padding='same'))
        #         model.add(LeakyReLU(alpha=0.2))
        #         model.add(Dropout(0.25))
        #         model.add(BatchNormalization(momentum=0.8))
        #         model.add(Flatten())
        #         model.add(Dense(32, activation='relu'))
        #         model.add(Dropout(0.5))
        #         model.add(Dense(1, activation='sigmoid'))
        #         model.summary()
        return model
        # data = Input(shape=self.data_shape) #建立一个输入层，输入维度是数据的维度，输入数据是G生成的数据
        # validity = model(data) # 输出数据是D的判别结果
        # return Model(data, validity) #返回模型

    def discriminator_loss(self,gen_disc_output,real_disc_output): #sig_src是发射信号，signal
        """
        real_datas is real recieved signal i.e. data_ds
        """
        # real_datas = real_datas.reshape(real_datas.shape[0],real_datas.shape[1],1)

        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #接受模型的类别概率预测结果和预期标签，然后返回样本的平均损失。
        generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(gen_disc_output),
                                                                 logits= gen_disc_output)
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_disc_output),
                                                            logits=real_disc_output)
        total_disc_loss = tf.reduce_mean(real_loss)+tf.reduce_mean(generated_loss)
        return total_disc_loss

    def generator_loss(self,gen_datas,real_H,rfd_weight):
        '''
        :param gen_datas:
        :param constructed_signal_channel_estimation: perform_omp(sig_src,constructed_rcv_signal)
        :param rfd_weight:
        :return:
        '''
        gen_datas = gen_datas.T
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminator.predict(gen_datas)),
                                                            logits = self.discriminator.predict(gen_datas))
        # rfd loss
        Enhanced_impulse_response = self.OMP.perform_omp(s=sig_src,data=gen_datas)#sig_src是发射信号
        gen_rfd_loss = self.rfd.cal_RFD(data_standard=real_H,
                                     data_other=Enhanced_impulse_response,
                                     dist0=15)
        total_gen_loss = gen_loss + rfd_weight*gen_rfd_loss
        total_gen_loss = tf.reduce_mean(total_gen_loss)
        return total_gen_loss

    def train_step(self,gen_datas,noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # gen_tape.watch(self.generator.trainable_weights)
            # disc_tape.watch(self.discriminator.trainable_weights)
            gen_datas = tf.convert_to_tensor(self.generator.predict(noise)) #生成器预测噪声
            real_disc_output = tf.convert_to_tensor(self.discriminator.predict(data_ds[self.idx]))
            gen_disc_output = tf.convert_to_tensor(self.discriminator.predict(gen_datas.numpy()))
        start_time = datetime.datetime.now()
        gen_loss = self.generator_loss(gen_datas=gen_datas.numpy(),
                                       real_H=constructed_signal_channel_estimation,
                                       rfd_weight=0.1)   # gen loss
        elapsed_time = datetime.datetime.now() - start_time
        print('time:',  elapsed_time)
        disc_loss = self.discriminator_loss(gen_disc_output=gen_disc_output,
                                            real_disc_output=real_disc_output)  # disc loss
            # gen_loss = tf.constant(gen_loss)
            # disc_loss = tf.constant(disc_loss)
        # gradient
        generator_gradient = gen_tape.gradient(gen_loss,self.generator.trainable_weights)
        discriminator_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        # apply gradient
        self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_weights))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_weights))
        return gen_loss, disc_loss

    def train(self,data,noise,epochs,batch_size,sample_interval):
        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            self.idx = np.random.randint(0, data.shape[0], batch_size)#随机产生batch_size个索引
            gen_datas = self.generator.predict(noise[self.idx]) #生成器预测噪声
            #data.shape[0]为数据集样本的数量，随机生成batch_size个数量的随机数，作为数据的索引
            gen_loss, disc_loss = self.train_step(gen_datas=gen_datas,noise=noise[self.idx])#计算损失,gradient,and refresh weight
            # validity= self.discriminator.predict(real_datas)#disc_real_output
            # disc_generated_output = self.discriminator.predict(gen_datas)
            elapsed_time = datetime.datetime.now() - start_time
            if epoch % sample_interval == 0: #相除取余 (每一百个epoch打印一次verbose)
                self.X1 = self.sample_data(epoch)
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, disc_loss[0], 100 * disc_loss[1], gen_loss),
                      'time:',  elapsed_time)
    def sample_data(self, epoch): #每100个epoch保存数据
        gen_datas = self.generator.predict(noise[self.idx]) #生成的样本，即把随机噪声扔进去然后预测,这里需要重写,idx
        file_name='./gen_data/gen_datas.mat'.format(epoch)
        sio.savemat(file_name,{'gen_datas':gen_datas})
        X1 = gen_datas
        return X1


#%%
gan=GAN() #类实例化
gan.train(data_ds,epochs=EPOCHS,batch_size=BATCH_SIZE,sample_interval=100,noise=noise) #采样间隔100，数据集这里需要修改
#%%
# for i in range(25):
#     gan.sample_data(i+1)
#%% 保存最后的G模型
# gan.generator.save('./dcgan.h5')
#%%



