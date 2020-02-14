from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil

'------------------------------------------------------------------------------------------------------------------------------------------------------------------'
 # click photos and send to the system

cam=cv2.VideoCapture(0)
cv2.namedWindow('test')
img_counter=0
while True:
    ret, frame=cam.read()
    cv2.imshow('test',frame)
    if not ret:
        break
    k=cv2.waitKey(1)

    if(k%256==27):
        print('esc ')
        break
    elif(k%256==32):
        img_name='opencv_frmae_{}.png'.format(img_counter)
        cv2.imwrite(img_name,frame)
        shutil.move('C:\\Users\\user\\PycharmProjects\\untitled\\opencv_frmae_0.png','C:\\Users\\user\\Desktop\\important\\machine learning files\\plant_disease\\PlantVillage\\TEST\\test')
        print('{} written!'.format(img_name))
        img_counter+=1
cam.release()
cv2.destroyAllWindows()

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------'
    #  MODEL
d=ImageDataGenerator()
train_it=d.flow_from_directory('C:\\Users\\user\\Desktop\\important\\machine learning files\\plant_disease\\PlantVillage\\TRAIN',classes=['diseased','normal'])
valid_it=d.flow_from_directory('C:\\Users\\user\\Desktop\\important\\machine learning files\\plant_disease\\PlantVillage\\VALIDATION',classes=['diseased','normal'])
test=d.flow_from_directory('C:\\Users\\user\\Desktop\\important\\machine learning files\\plant_disease\\PlantVillage\\TEST')
x,y=train_it.next()
#print(x.shape[0])

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------'
 #  Binary label encoding
 #  display the images of the training data  and encoding the labels
 # DISEASED is encoded as [1,0] and NORMAL is encoded as [0,1]
p=[]

def ret(y):
    array2 = ['Diseased', 'Normal']
    array1=[1,0]
    b=y[0].astype('int')
    c = dict((i, j) for i, j in zip(array1, array2))
    return c[b]
def image_showcase(x,y):
    for i in range(x.shape[0]):
        x[i]=x[i]/255
        plt.matshow(x[i])
        plt.show()
        print(ret(y[i]))

image_showcase(x,y)

'---------------------------------------------------------------------------------------------------------------------------------------------------------------------'
       # build the model

from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
model=Sequential()
model.add(Conv2D(50,kernel_size=(3,3),input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(100,kernel_size=(3,3),input_shape=(256,256,3)))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(30,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_it,epochs=20,validation_data=valid_it,steps_per_epoch=10)
p.append(model.predict_generator(test,steps=1))
for i in p:
    print(np.argmax(i))
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------'
     #  show the test data
a,b=test.next()
def image_showcase(a,b):
    for i in range(a.shape[0]):
        a[i]=a[i]/255
        plt.matshow(a[i])
        plt.show()
image_showcase(a,b)
