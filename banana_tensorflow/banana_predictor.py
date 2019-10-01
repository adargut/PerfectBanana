# import some packages
from keras.models import load_model
import argparse
import pickle
import cv2

# construct argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
                    help="path to input image we are going to classify")
parser.add_argument("-m", "--model", required=True,
                    help="path to trained Keras model")
parser.add_argument("-l", "--label-bin", required=True,
                    help="path to label binarizer")
parser.add_argument("-w", "--width", type=int, default=32,
                    help="target spatial dimension width")
parser.add_argument("-he", "--height", type=int, default=32,
                    help="target spatial dimension height")
parser.add_argument("-f", "--flatten", type=int, default=-1,
                    help="whether or not we should flatten the image")
args = vars(parser.parse_args())

# load input image and resize it
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# scale pixels to be in [0, 1]
image = image.astype("float") / 255.0

# check if flattening is needed
if args["flatten"] > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
# CNNs require no flattening
else:
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# load model and label binarizer
print("[INFO] loading model and binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# make a predict for image
prediction = model.predict(image)

# find out prediction label
idx = prediction.argmax(axis=1)[0]
label = lb.classes_[idx]

# draw prediction on output image
text = "{}: {:.2f}%".format(label, prediction[0][idx] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
