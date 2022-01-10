#交叉验证  

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





folden = 1  #留下folden个做测试，遍历测试集。folden=1为Leave One Out 
n = 10 #训练好模型，增加预测次数，减小误差



dataset = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Input1.mat")
dataset = dataset['Input1']
labels = sio.loadmat("/home/user/Documents/wjm/TensorFlow/Training_data/Training_data3/Output1.mat")
labels = labels['Output1']


data_num,data_long = dataset.shape
folden_num = data_num//folden   #folden数量

test_true = np.empty(shape=(folden_num*folden,1))
test_pred = np.empty(shape=(folden_num*folden,n))
test_pred_mean = np.empty(shape=(folden_num*folden,1))
test_pred_sum = np.empty(shape=(folden_num*folden,n+1)) 
for k in range(n):
	for i in range(folden_num):
	    train_data = dataset
	    train_labels = labels
	    test_data = np.empty(shape=(folden,data_long))   
	    test_labels= np.empty(shape=(folden,1))
	    delete_index = [0]*folden
	    
	    for j in range(folden):
	    	delete_index[j] = i*folden+j
	    	test_data[j] = dataset[i*folden+j]
	    	test_labels[j] = labels[i*folden+j]
	    
	    train_data = np.delete(train_data,delete_index,axis=0)
	    train_labels = np.delete(labels,delete_index,axis=0)

	    train_data = train_data.reshape(data_num-folden, 24, 9, 1)
	    train_labels = train_labels.reshape(data_num-folden,1)

	    test_data = test_data.reshape(folden, 24, 9, 1)
	    test_labels = test_labels.reshape(folden,1)




	    model = models.Sequential()
	    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 9 ,1)))
	    model.add(layers.MaxPooling2D((2, 2)))
	    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

		#model.summary()

	    model.add(layers.Flatten())
	    model.add(layers.Dense(40, activation='relu'))
	    model.add(layers.Dense(1))

		#optimizer = tf.keras.optimizers.RMSprop(0.001)

	    model.compile(loss='mse',
			  optimizer='adam',
			  metrics=['RootMeanSquaredError', 'mae'])

		#example_batch = train_atoms[:10]
		#example_result = model.predict(example_batch)
		#example_result

		
	    EPOCHS = 100

	    history = model.fit(
	      train_data, train_labels, epochs=EPOCHS, validation_data = (test_data,test_labels),
	      )

	    hist = pd.DataFrame(history.history)
	    hist['epoch'] = history.epoch
	    hist.tail()

	    test_true[i*folden:(i+1)*folden] = test_labels	    
	    test_pred[i*folden:(i+1)*folden,k] = model.predict(test_data).reshape(folden,)
	    print("第",k+1,"轮第",i+1,"个folden预测结束","共需预测",n,"轮",folden_num,"个folden")

for i in range(folden_num*folden):
	test_pred_mean[i] = np.mean(test_pred[i])

test_pred_sum[:,0:n] = test_pred
test_pred_sum[:,n] = test_pred_mean.reshape(folden_num*folden,)

np.savetxt('test_true_100',test_true)
np.savetxt('test_pred_sum_100',test_pred_sum)

r=np.mean(np.multiply((test_pred_mean-np.mean(test_pred_mean)),(test_true-np.mean(test_true))))/(np.std(test_true)*np.std(test_pred_mean))

print(r*r)
	
