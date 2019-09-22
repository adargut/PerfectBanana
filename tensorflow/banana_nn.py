# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import warnings
from tqdm import tqdm

# ignore tf.compat errors
warnings.filterwarnings("ignore", category=DeprecationWarning)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path for input dataset")
parser.add_argument("-m", "--model", required=True,
                    help="path to output trained model")
parser.add_argument("-l", "--label-bin", required=True,
                    help="path to output label binarizer")
parser.add_argument("-p", "--plot", required=True,
                    help="path to output plotted data")
args = vars(parser.parse_args())

# initialize data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab image paths & shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over images and append to data
for imagePath in tqdm(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    # extract label from image name and append to labels
    label = imagePath.split("\\")[1]
    labels.append(label)

# scale pixels in range [0, 1] and convert to NumPy array
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition data into test/train
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# transform labels to ohl vector format
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# set up model architecture
model = Sequential()
# input layer and first hidden layer, note input shape is tensor of dimensions 32x32x3 = 3072
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
# consecutive hidden layer and output layer, consisting of #labels nodes
model.add(Dense(512, activation="sigmoid"))
print("[INFO] output layer has", len(lb.classes_), "nodes...")
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize learning rate and #epochs for training
lr = 0.01
epochs = 15
opt = SGD(lr=lr)

# compile model using stochastic gradient descent optimizer and categorical crossentropy
print("[INFO] training neural network...")
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# train model in batches of 32
Hist = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=32)

# evaluate how well the network makes predictions
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, Hist.history["loss"], label="train_loss")
plt.plot(N, Hist.history["val_loss"], label="val_loss")
plt.plot(N, Hist.history["accuracy"], label="train_acc")
plt.plot(N, Hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Banana NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save model & label binarizer
print("[INFO] saving model...")
model.save(args["model"])
print("args for label:", args["label-bin"])
file = open(args["label-bin"], "wb")
file.write(pickle.dump(lb))
file.close()
