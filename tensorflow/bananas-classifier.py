# Generate our custom dataset
from keras.preprocessing.image import ImageDataGenerator
# Use a linear stack of layers for our model
from keras.models import Sequential
# Our model consists of input, hidden & output layers
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# Used for some tweaking with data
from keras import backend as kback

# Dimension for banana photos
img_width, img_height = 150, 150  # TODO: maybe use other dimensions in accordance to accuracy?

train_data_dir = 'bananas/train'  # Use to train NN, bananas with index 1-80
validation_data_dir = 'bananas/test'  # Use to validate NN's accuracy, bananas with indexes 81-100

# Parameters for Naive Bayes classifier
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# Determine input shape
if kback.image_data_format() == 'channels first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Build model: we are using a CNN with ReLU activation function alongside sigmoid
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Augment train data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Augment test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Tensor image data generator from batch
train_generator = train_datagen.flow_from_directory(  # Train data
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(  # Test data
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Train model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Save our model TODO freeze model to put in app's asset folder
model.save('bananas/')
