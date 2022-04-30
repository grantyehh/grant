
"""
Created on Fri Apr 22 12:48:38 2022

@author: USER
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
from PIL import Image
import numpy as np
import random

np.random.seed(10) 

x_train = np.empty((4000,3,100,100),dtype="uint8") # for train
y_train = np.empty((4000,),dtype="uint8")
x_test = np.empty((20,3,100,100),dtype="uint8") # for test
y_test = np.empty((20,),dtype="uint8")
imgs_1 = os.listdir("C:/Users/USER/Desktop/train")
num_1 = len(imgs_1)
for i in range(num_1):
    img_1 = Image.open("C:/Users/USER/Desktop/train/"+imgs_1[i])
    arr_1 = np.array(img_1)
    x_train[i,:,:,:] = [arr_1[:,:,0],arr_1[:,:,1],arr_1[:,:,2]]
    y_train[i] =int(imgs_1[i].split(' ')[0])
 
   
print(len(y_test))
imgs_2 = os.listdir("C:/Users/USER/Desktop/test")
num_2 = len(imgs_2)
for i in range(num_2):
	img_2 = Image.open("C:/Users/USER/Desktop/test/"+imgs_2[i])
	arr_2 = np.array(img_2)
	x_test[i,:,:,:] = [arr_2[:,:,0],arr_2[:,:,1],arr_2[:,:,2]]
	y_test[i] =int(imgs_2[i].split(' ')[0])



 
x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)

index_1 = [i for i in range(len(x_train))]
random.shuffle(index_1)
x_train = x_train[index_1]
y_train = y_train[index_1]

index_2 = [i for i in range(len(x_test))]
random.shuffle(index_2)
x_test = x_test[index_2]
y_test = y_test[index_2]

import tensorflow
import numpy as np

print("train data:",'images:',x_train.shape," labels:",y_train.shape) 
print("test data:",'images:',x_test.shape," labels:",y_test.shape) 

x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

y_train_OneHot = tensorflow.keras.utils.to_categorical(y_train)
y_test_OneHot = tensorflow.keras.utils.to_categorical(y_test)



from tensorflow.keras.layers import Flatten,Dropout,Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

model = Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(100, 100,3), 
                 activation='relu', 
                 padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(rate=0.2))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(10, activation='softmax'))

print(model.summary())

try:
    model.load_weights("./fruit.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=20, batch_size=200, verbose=1)          

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


scores = model.evaluate(x_test_normalize,  y_test_OneHot, verbose=1)
scores[1]



prediction=model.predict_classes(x_test_normalize)
prediction[:11]


label_dict={0:"coconut",1:"durian",2:"lemon",3:"litchi",4:"mango",
            5:"papaya",6:"pineapple",7:"polo",8:"banana",9:"rambutan"}
			
print(label_dict)		


def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    for i in range(0, len(labels)):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(x_test,y_test, prediction,0,10)


Predicted_Probability=model.predict(x_test_normalize)

def show_Predicted_Probability(y,prediction, x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],
          'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(100, 100,3)))
    plt.show()
    for j in range(10):
        print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

show_Predicted_Probability(y_test,prediction, x_test,Predicted_Probability,0)
show_Predicted_Probability(y_test,prediction, x_test,Predicted_Probability,3)


model.save_weights("./fruit.h5")
print("Saved model to disk")