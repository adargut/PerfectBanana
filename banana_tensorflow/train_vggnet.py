import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2

from banana_tensorflow.banana_vggnet import BananaVGGNet

matplotlib.use("Agg")

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
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)

    label = imagePath.split("\\")[1]
    print("[INFO] appending label", label, "...")
    labels.append(label)

# scale pixels to be in [0, 1] by division in 255
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition data into training/testing as 75%/25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels to vectors, "one-hot" methodology
lb = LabelBinarizer()
trainY = lb.fit(trainY)
testY = lb.transform(testY)

# construct image generator for keras data augmentation. this allows us to avoid overfitting & generalize well.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize our VGG style convolutional network
model = BananaVGGNet.build(width=64, height=64, depth=3,
                           classes=len(lb.classes_))

# parameters for training model
LR = 0.01  # todo maybe tweak this?
epochs = 75
batch = 32

# compile model
print("[INFO] training CNN...")  # todo maybe change to radam optimizer?
opt = SGD(lr=LR, decay=LR / epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch),
                        validation_data=(testX, testY), steps_per_epoch=(trainX) // batch,
                        epochs=epochs)

# evaluate the network we trained
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot loss and accuracy histogram
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save model & label binarizer to disk
print("[INFO] saving work...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
