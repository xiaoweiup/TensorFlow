#预训练 

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import scipy.io as sio

#限制显存使用最大值
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)])  #1024为1GB

#读取数据
dataset = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Input1.mat")
dataset = dataset['Input1']
labels = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Output1.mat")
labels = labels['Output1']

#打乱顺序
data_num,_ = dataset.shape
index = np.arange(data_num)
np.random.shuffle(index)
dataset = dataset[index]
labels = labels[index]


#生成输入
train_data = dataset[0:500,0:216]
test_data = dataset[500:642,0:216]
train_labels = labels[0:500]
test_labels = labels[500:642]

train_data = train_data.reshape(500, 24, 9, 1)
train_labels = train_labels.reshape(500,1)

test_data = test_data.reshape(142, 24, 9, 1)
test_labels = test_labels.reshape(142,1)


#此处调参
n = 100 #隐层节点数
m = 100 #训练好模型预测m次求均值
test_num = 142 #测试集数量，一般取数据集的百分之20

#初始化
test_pred = np.empty(shape=(test_num,m))
test_pred_mean = np.empty(shape=(test_num,1))
Sr_statis = []
for i in range(n):
	for j in range(m): 
		model = models.Sequential()
		model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 9 ,1)))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))

		#model.summary()

		model.add(layers.Flatten())
		model.add(layers.Dense(i+1, activation='relu'))
		model.add(layers.Dense(1))

		#optimizer = tf.keras.optimizers.RMSprop(0.001)

		model.compile(loss='mse',
			      optimizer='adam',
			      metrics=['RootMeanSquaredError', 'mae'])

		#example_batch = train_atoms[:10]
		#example_result = model.predict(example_batch)
		#example_result


		EPOCHS = 150

		history = model.fit(
		  train_data, train_labels, epochs=EPOCHS, validation_data = (test_data,test_labels)
		  )

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		hist.tail()

		test_pred[:,j] = model.predict(test_data).reshape(test_num,)
		print("隐层节点为",i+1,"时第",j+1,"次预测")  

	for k in range(test_num):
		test_pred_mean[k] = np.mean(test_pred[k])
	r = np.mean(np.multiply((test_pred_mean-np.mean(test_pred_mean)),(test_labels-np.mean(test_labels))))/(np.std(test_labels)*np.std(test_pred_mean))
	RMSD = np.sqrt(np.mean(np.square(test_pred_mean-test_labels)))
	
	Sr_statis.append([r*r,RMSD])
	
	print("当隐层节点为",i+1,"时预测r方为:",r*r)

np.savetxt('Sr_statis',Sr_statis)
        

#np.savetxt('test_result_50',test_result)
#np.savetxt('test_labels_50',test_labels)

