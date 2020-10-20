#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:43:19 2020

@author: subhrohalder
"""

#43 different labels
#ClassId	 SignName
#0	Speed limit (20km/h)
#1	Speed limit (30km/h)
#2	Speed limit (50km/h)
#3	Speed limit (60km/h)
#4	Speed limit (70km/h)
#5	Speed limit (80km/h)
#6	End of speed limit (80km/h)
#7	Speed limit (100km/h)
#8	Speed limit (120km/h)
#9	No passing
#10	No passing for vechiles over 3.5 metric tons
#11	Right-of-way at the next intersection
#12	Priority road
#13	Yield
#14	Stop
#15	No vechiles
#16	Vechiles over 3.5 metric tons prohibited
#17	No entry
#18	General caution
#19	Dangerous curve to the left
#20	Dangerous curve to the right
#21	Double curve
#22	Bumpy road
#23	Slippery road
#24	Road narrows on the right
#25	Road work
#26	Traffic signals
#27	Pedestrians
#28	Children crossing
#29	Bicycles crossing
#30	Beware of ice/snow
#31	Wild animals crossing
#32	End of all speed and passing limits
#33	Turn right ahead
#34	Turn left ahead
#35	Ahead only
#36	Go straight or right
#37	Go straight or left
#38	Keep right
#39	Keep left
#40	Roundabout mandatory
#41	End of no passing
#42	End of no passing by vechiles over 3.5 metric tons

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


#importing the dataset
with open ("./dataset/train.p",mode = 'rb') as training_data:
    train = pickle.load(training_data)
    
with open ("./dataset/valid.p",mode = 'rb') as valid_data:
    valid = pickle.load(valid_data)
    
with open ("./dataset/test.p",mode = 'rb') as test_data:
    test = pickle.load(test_data)


#split X and y
X_train,y_train = train['features'],train['labels']
X_validation, y_validation = valid['features'],valid['labels']
X_test, y_test = test['features'],test['labels']


#checking the shape
X_train.shape
y_train.shape

X_validation.shape
y_validation.shape

X_test.shape
y_test.shape


#plotting the image
i  = 10
plt.imshow(X_train[i])
y_train[i]

plt.imshow(X_validation[i])
y_validation[i]

plt.imshow(X_test[i])
y_test[i]

#shuffling the dataset
from sklearn.utils import shuffle
X_train,y_train = shuffle(X_train,y_train)


#converting to gray scale
# eg: [28,  25,  24] = 25.66666667
X_train_gray = np.sum(X_train/3 , axis = 3,keepdims = True)
X_validation_gray = np.sum(X_validation/3 , axis = 3,keepdims = True)
X_test_gray = np.sum(X_test/3 , axis = 3,keepdims = True)

#ploting gray scale image
i  = 100
plt.imshow(X_train[i])
plt.imshow(X_train_gray[i].squeeze(),cmap ='gray')
plt.imshow(X_train_gray_norm[i].squeeze(),cmap ='gray')
y_train[i]

X_train_gray.shape
X_validation_gray.shape
X_test_gray.shape


#normalisation by dividing by 128
#0 -> (0-128)/128 = -1 -> Lowest is 0 and it became -1.
#128 -> (128-128)/128 = 0 -> Middle value 128 will become 0.
#255 -> (255-128)/128 = 0.99 ~ 1 -> Highest value 255 became 1.

X_train_gray_norm = (X_train_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128

#importing libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


#model building
cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 6,kernel_size = (5,5), activation='relu',input_shape =(32,32,1)))
cnn_model.add(AveragePooling2D())

cnn_model.add(Conv2D(filters = 16,kernel_size = (5,5), activation='relu'))
cnn_model.add(AveragePooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 120,activation ='relu' ))

cnn_model.add(Dense(units = 84,activation ='relu' ))

cnn_model.add(Dense(units = 43,activation ='softmax' ))


cnn_model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(lr =0.001),metrics =['accuracy'])

#checking summary
cnn_model.summary()

#fitting the data
history = cnn_model.fit(X_train_gray_norm,y_train,batch_size = 500,epochs = 50, verbose =1,validation_data= (X_validation_gray_norm,y_validation))

#checking score
score = cnn_model.evaluate(X_test_gray_norm,y_test)
print(f"Test Accuracy {score[1]}")

history.history.keys()

#Plotting Training and Validation Accuracy Training and Validation Loss
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs,accuracy,'bo',label ='Training Accuracy')
plt.plot(epochs,val_accuracy,'b',label ='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


plt.plot(epochs,loss,'ro',label ='Training Loss')
plt.plot(epochs,val_loss,'r',label ='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

#predicting
y_pred = cnn_model.predict_classes(X_test_gray_norm)

#plotting confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
plt.figure(figsize = (43,43))
sns.heatmap(cm,annot = True)

#ploting some predicted and actual output
l = 10
w = 10

fig, axes = plt.subplots(l,w,figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0,l*w):
    axes[i].imshow(X_test[i])
    axes[i].set_title(f"Predicted:{y_pred[i]} \n Actual:{y_test[i]}" )
    axes[i].axis('off')


plt.subplots_adjust(wspace = 1,hspace=1)