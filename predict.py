import keras  
from keras.models import load_model  
from keras.models import Sequential  
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  
from keras import backend as K  
import numpy as np  
import h5py   
import os  

model_path = 'first_try.h5'
model = load_model(model_path)

img = load_img('test.jpg', target_size=(224, 224)) 
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

classes = model.predict(x )

print (classes,'\n') 