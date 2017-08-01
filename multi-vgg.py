from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


classes = 5
epochs = 50
batch_size = 32
img_width, img_height = 200,200
nb_train_samples = 500  # 5 classes
nb_validation_samples = 150 # 5 classes

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', name='fc1'))
model.add(Dense(64, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


train_data_dir = './data/train'
validation_data_dir = './data/validation'

train_datagen = ImageDataGenerator(
    rescale=2. / 255, #
    shear_range=0.2, #Shear angle in counter-clockwise direction as radians
    zoom_range=0.3, #[lower, upper] = [1-zoom_range, 1+zoom_range]
    rotation_range = 30,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    zoom_range=0.3,
    rotation_range = 30,
    rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	#save_to_dir='./data/new_data/train',
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size ,
	#save_to_dir='./data/new_data/test',
	class_mode='categorical')

model.fit_generator(
	train_generator,
	steps_per_epoch= batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=batch_size) 

model.save('multi-vgg.h5')
