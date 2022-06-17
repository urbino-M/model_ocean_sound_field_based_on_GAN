#%%

import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.disable_v2_behavior() #禁用2.0版本的方法
print(tf.__version__)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%

from tensorflow.keras.models import load_model

#%%

model=load_model('./dcgan.h5')

#%%

model.summary()

#%%

for i in range(100):
    noise_dim=5000
    noise = np.random.normal(0, 1, (1, noise_dim))
    gen_datas = model.predict(noise)
    file_name='./gen_data/nor_mat_cut_c2.mat'.format(i)
    import scipy.io as sio
    sio.savemat(file_name,{'data':gen_datas})

#%%


