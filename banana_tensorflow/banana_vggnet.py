from keras.models import Sequential
from keras import layers
from keras import backend as bk


class BananaVGGNet:
    def __init__(self):
        pass

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
        model.add(layers.Conv2D(32, (3, 3), padding="same", inputShape=inputShape))
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
