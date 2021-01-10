import os, random, math
from pprint import pprint
from datetime import datetime as dt
import numpy as np
import keras as k

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

SEED = 1337
np.random.seed(SEED)

VALID_FRUITS = ["Apricot","Avocado","Banana","Chestnut","Clementine","Granadilla","Kiwi","Limes",
          "Mango","Maracuja","Peach","Pear","Pomegranate","Raspberry","Pineapple","Strawberry","Walnut"]

IMG_WIDTH=35
IMG_HEIGHT=35
TARGET_SIZE=[IMG_WIDTH, IMG_HEIGHT]

CHANNELS=3

TRAIN_PATH = "/home/veerapandian/projects/fruits-360/Training"
TEST_PATH = "/home/veerapandian/projects/fruits-360/Test"
PREDICTION_PATH = "/home/veerapandian/projects/fruits-360/test-multiple_fruits"

BATCH_SIZE=32
EPOCHS=20

train_gen = k.preprocessing.image.ImageDataGenerator(rotation_range=0.1,width_shift_range=0.1,
                                                     height_shift_range=0.1,brightness_range=[0.5, 1.5],
                                                     channel_shift_range=0.05,rescale=1./255)

test_gen = k.preprocessing.image.ImageDataGenerator(rotation_range=0.1,width_shift_range=0.1,
                                                    height_shift_range=0.1,brightness_range=[0.5, 1.5],
                                                    channel_shift_range=0.05,rescale=1./255)

train_images_iter = train_gen.flow_from_directory(TRAIN_PATH,target_size = TARGET_SIZE,classes = VALID_FRUITS,
                                                  class_mode = 'categorical',seed = SEED)

test_images_iter = test_gen.flow_from_directory(TEST_PATH,target_size = TARGET_SIZE,classes = VALID_FRUITS,
                                                class_mode = 'categorical',seed = SEED)

def get_subplot_grid(mylist, columns, figwidth, figheight):
    plot_rows = math.ceil(len(mylist) / 2.)
    fig, ax = plt.subplots(plot_rows, 2, sharey=True, sharex=False)
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)
    fig.subplots_adjust(hspace=0.4)
    axflat = ax.flat
    #remove the unused subplot, if any
    for ax in axflat[ax.size - 1:len(mylist) - 1:-1]:
        ax.set_visible(False)
    return fig, axflat

test_images_classes = ["Avocado","Kiwi","Pear","Pineapple","Pomegranate","Strawberry"]
test_images=[]

plt.rc('font',family = 'sans-serif',  size=8)
fig, axflat = get_subplot_grid(mylist=test_images_classes, columns=2, figwidth=4, figheight=6)

for idx, label in enumerate(test_images_classes):
    image_folder = os.path.join(TRAIN_PATH, label)
    image_file = os.path.join(image_folder, random.choice(os.listdir(image_folder)) )
    loaded_image = k.preprocessing.image.load_img(path=image_file,target_size=(IMG_WIDTH,IMG_HEIGHT,CHANNELS))
    #convert to array and resample dividing by 255
    img_array = k.preprocessing.image.img_to_array(loaded_image) / 255.
    test_images.append({"idx":idx, "image":img_array, "label": label})
    axflat[idx].set_title(label, size=12)
    axflat[idx].imshow(img_array)
plt.show()
plt.gcf().clear()

trained_classes_labels = list(train_images_iter.class_indices.keys())
train_images_iter.class_indices

unique, counts = np.unique(train_images_iter.classes, return_counts=True)
print ("number of samples per class")
dict(zip(train_images_iter.class_indices, counts))

def build_model():
    rtn = k.Sequential()
    rtn.add(k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides=(1, 1),
                                    input_shape = (IMG_WIDTH, IMG_HEIGHT, CHANNELS),
                                    kernel_regularizer=k.regularizers.l2(0.0005),
                                    name='conv2d_1'
                                )
                )
    rtn.add(k.layers.BatchNormalization())
    rtn.add(k.layers.Activation('relu', name='activation_conv2d_1'))
    rtn.add(k.layers.SpatialDropout2D(0.2))
    rtn.add(k.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', name='conv2d_2'))
    rtn.add(k.layers.BatchNormalization())
    rtn.add(k.layers.LeakyReLU(0.5, name='activation_conv2d_2'))
    rtn.add(k.layers.MaxPooling2D(pool_size = (2,2)))
    rtn.add(k.layers.Flatten())
    rtn.add(k.layers.Dense(units = 250, name='dense_1' ) )
    rtn.add(k.layers.Activation('relu', name='activation_dense_1'))
    rtn.add(k.layers.Dropout(0.5))
    rtn.add(k.layers.Dense(units = len(trained_classes_labels), name='dense_2'))
    rtn.add(k.layers.Activation('softmax', name='activation_final'))
    return rtn

my_model = build_model()
my_model.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],
                 optimizer = k.optimizers.RMSprop(lr = 1e-4, decay = 1e-6))

start = dt.now()
history = my_model.fit_generator(
  train_images_iter,
  steps_per_epoch = train_images_iter.n // BATCH_SIZE, #floor per batch size
  epochs = EPOCHS,
  validation_data = test_images_iter,
  validation_steps = test_images_iter.n // BATCH_SIZE,
  verbose = 1,
  callbacks = [
    #early stopping in case the loss stops decreasing
    k.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    # only save the model if the monitored quantity (val_loss or val_acc) has improved
    k.callbacks.ModelCheckpoint("fruits_checkpoints.h5", monitor='val_loss', save_best_only = True),
    # only needed for visualising with TensorBoard
    k.callbacks.TensorBoard(log_dir = "logs/{:%d_%b_%Y_%H:%M:%S}".format(dt.now()) )
  ]
)
print(history.history.keys())

plt.style.use('fivethirtyeight')

xepochs = [i+1 for i in range(0, len(history.history['loss']))]
plt.figure(figsize=(5,3))
# Loss
#plt.ylim([-0.1,0.5])
plt.plot(xepochs, history.history['loss'])
plt.plot(xepochs, history.history['val_loss'])
plt.xticks(xepochs)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# Accuracy
#plt.ylim([0.7,1.05])
# plt.figure(figsize=(5,3))
# plt.plot(xepochs, history.history['acc'])
# plt.plot(xepochs, history.history['val_acc'])
# plt.xticks(xepochs)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['training', 'validation'], loc='upper left')
# plt.show()

df_out = {'val_loss': history.history['val_loss'][0],'val_acc': history.history['val_accuracy'][0],
          'elapsed_time': (dt.now() - start).seconds}
print(df_out)

my_model=build_model()
my_model.load_weights("fruits_checkpoints.h5")
my_model.compile(loss = 'categorical_crossentropy', 
                    metrics = ['accuracy'], 
                    optimizer = k.optimizers.RMSprop(lr = 1e-4, decay = 1e-6)
                )
print("Created model and loaded weights from file")

my_model.summary()

PREDICTION_PATH = "/home/veerapandian/projects/fruits-360/sample"
images_for_prediction = [filename for filename in sorted(os.listdir(PREDICTION_PATH)) if filename.endswith(".jpg")]

for filename in images_for_prediction:
    loaded_image = k.preprocessing.image.load_img(path=PREDICTION_PATH+'/'+filename, target_size=(IMG_WIDTH,IMG_HEIGHT,CHANNELS))
    #convert to array and resample dividing by 255
    img_array = k.preprocessing.image.img_to_array(loaded_image) / 255.

    #add sample dimension. the predictor is expecting (1, CHANNELS, IMG_WIDTH, IMG_HEIGHT)
    img_np_array = np.expand_dims(img_array, axis = 0)
    #img_class = my_model.predict_classes(img_np_array)

    predictions = my_model.predict(img_np_array)
    classidx = np.argmax(predictions[0])
    label = trained_classes_labels[classidx]

    predictions_pct = ["{:.2f}%".format(prob * 100) for prob in predictions[0] ]
    pprint(dict(zip(trained_classes_labels, predictions_pct)) )
    print("Prediction: %s (class %s) %s" % (label, classidx, predictions_pct[classidx])) 

    plt.figure(figsize=(3,4))
    plt.imshow(img_array)
    plt.title("%s %s" % (label, predictions_pct[classidx]))
    plt.show()

plt.gcf().clear()
