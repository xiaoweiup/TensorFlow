#多次测试

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import scipy.io as sio


##按需申请显存空间
#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(device=gpu, True)

#限制显存使用最大值
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])  #1024为1GB




#可调参数

EPOCHS = 40
n = 100 #增加预测次数，减小误差

#加载数据集
train_data = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Input1.mat")
train_data = train_data['Input1']
train_labels = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Output1.mat")
train_labels = train_labels['Output1']

test_data = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Test_data/Test_data3/Input1.mat")
test_data = test_data['Input1']
test_labels = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Test_data/Test_data3/Output1.mat")
test_labels = test_labels['Output1']


train_data = train_data.reshape(642, 24, 9, 1)
train_labels = train_labels.reshape(642, 1)

test_data = test_data.reshape(386, 24, 9, 1)
test_labels = test_labels.reshape(386, 1)


#初始化
test_true = np.empty(shape=(386,1))
test_pred = np.empty(shape=(386,n))
test_pred_mean = np.empty(shape=(386,1))
test_pred_sum = np.empty(shape=(386,n+1)) 

#训练
for i in range(n):
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 9 ,1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))

	model.compile(loss='mse',
		      optimizer='adam',
		      metrics=['mae', 'mse'])

	history = model.fit(
	  train_data, train_labels, epochs=EPOCHS, validation_data = (test_data,test_labels)
	)
	
	    
	test_pred[:,i] = model.predict(test_data).reshape(386,)
	print("第",i+1,"次循环已结束,共需进行",n,"次循环")

test_true = test_labels	
for j in range(386):
	test_pred_mean[j] = np.mean(test_pred[j])

test_pred_sum[:,0:n] = test_pred
test_pred_sum[:,n] = test_pred_mean.reshape(386,)

np.savetxt('test_true_100',test_true)
np.savetxt('test_pred_sum_100',test_pred_sum)

r=np.mean(np.multiply((test_pred_mean-np.mean(test_pred_mean)),(test_true-np.mean(test_true))))/(np.std(test_true)*np.std(test_pred_mean))

print(r*r)
	
