from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_path = "/home/veerapandian/projects/fruits-360/Training"
test_path = "/home/veerapandian/projects/fruits-360/Test"

img = load_img(test_path + "/Apple Braeburn/3_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*')
num_of_classes = len(className)
print("Number of Classes : ", num_of_classes)

examplefruit = glob(train_path + "/Apple Braeburn" + '/*')
num_of_classesex = len(examplefruit)
print("Number of Classes : ", num_of_classesex)

classname = pd.DataFrame(className)
classname.nunique()

example = glob(train_path + '/*')
num_of_classes = len(className)
print("Number of Classes : ", num_of_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.3,
                                   horizontal_flip=True, zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=x.shape[:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=x.shape[:2],
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  class_mode="categorical")

hist = model.fit_generator(generator=train_generator, steps_per_epoch=1600 // batch_size,
                           epochs=75, validation_data=test_generator, validation_steps=800 // batch_size)

model.save_weights("trial.h5")

print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Test Loss")
plt.legend()
plt.show()

#=======================================================

print(hist.history.keys())
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Test Accuracy")
plt.legend()
plt.show()

import json 
with open("trial.json", "w") as f:
    json.dump(hist.history, f)

model.load_weights("/home/veerapandian/projects/trans_ln_vgg16/trial.h5")
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
print("Created model and loaded weights from file")
model.summary()

train_images_iter = train_datagen.flow_from_directory(train_path,
                                                      target_size=x.shape[:2],
                                                      batch_size=batch_size,
                                                      color_mode="rgb",
                                                      class_mode="categorical")

trained_classes_labels = list(train_images_iter.class_indices.keys())

import keras as k
loaded_image = k.preprocessing.image.load_img(path="/home/veerapandian/projects/verify_img/apple/0_100.jpg",
                                              target_size=(100, 100, 3))
img_array = k.preprocessing.image.img_to_array(loaded_image) / 255.
img_np_array = np.expand_dims(img_array, axis = 0)
predictions = model.predict(img_np_array)
classidx = np.argmax(predictions[0])
label = trained_classes_labels[classidx]
print(label)
