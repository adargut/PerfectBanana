import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
from keras import layers
from keras import backend as bk
from keras import Sequential
import numpy as np
import argparse
import random
import pickle
import cv2

matplotlib.use("Agg")


class BananaVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        # 3x3x3 convolution
        inputShape = (height, width, depth)
        chanDim = -1

        # support both tf and Theano backends for image ordering
        if bk.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1


        # first layer set, append layers to model CONV=>RELU=>POOL
        model.add(layers.Conv2D(32, (3, 3), padding="same",
                                input_shape=inputShape))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # second layer set
        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(64, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # third layer set
        model.add(layers.Conv2D(128, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(128, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(128, (3, 3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # flatten layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # softmax classifier
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))

        # finished constructing architecture for network
        return model


# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize data & labels
print("[INFO] loading images...")
data, labels = [], []

# grab image paths & shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over input images & append them to data
for imagePath in imagePaths:
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        data.append(image)
        label = imagePath.split("\\")[1]
        print("[INFO] appending label", label)
        labels.append(label)
    except:
        print("[ERROR] got a blank image...")

# scale pixels to be in [0, 1] by division in 255
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition data into training/testing as 75%/25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels to vectors, "one-hot" methodology
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct image generator for keras data augmentation. this allows us to avoid overfitting & generalize well.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize our VGG style convolutional network
model = BananaVGGNet.build(width=64, height=64, depth=3,
                           classes=len(lb.classes_))

# parameters for training model
LR = 0.01
epochs = 75
batch = 32

# compile model
print("[INFO] training CNN...")
opt = SGD(lr=LR, decay=LR / epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch,
                        epochs=epochs)

# save model & label binarizer to disk
print("[INFO] saving work...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
