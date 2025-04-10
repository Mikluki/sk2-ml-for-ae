import keras
from keras import layers, models

# Load the MNIST dataset, which contains handwritten digits
# The dataset is split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

# Preprocess the image data:
# 1. Reshape the images to have shape (samples, height, width, channels)
# 2. Convert the data type to float32
# 3. Normalize pixel values to be between 0 and 1 by dividing by 255
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# Convert the labels to categorical one-hot encoding
# This transforms the label integers (0-9) into binary class vectors
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

########### MODEL ARCHITECTURE ###########
# Create a Sequential model - this is a linear stack of layers
model = models.Sequential()

# Add the first convolutional layer with 32 filters, 3x3 kernel size
# ReLU activation function, and specifying the input shape
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))

# Add max pooling layer to reduce spatial dimensions by taking maximum value in 2x2 windows
model.add(layers.MaxPooling2D((2, 2)))

# Add second convolutional layer with 64 filters and 3x3 kernel size
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Add another max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Add a third convolutional layer with 64 filters
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Flatten the 3D output to 1D to connect to dense layers
model.add(layers.Flatten())

# Add a dense (fully connected) layer with 64 neurons and ReLU activation
model.add(layers.Dense(64, activation="relu"))

# Add the output layer with 10 neurons (one for each digit class)
# Softmax activation for multi-class classification
model.add(layers.Dense(10, activation="softmax"))

########### COMPILE THE MODEL ###########
# Compile the model:
# - Using Adam optimizer
# - Categorical cross-entropy loss function (standard for multi-class classification)
# - Tracking accuracy metric during training
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

########### TRAIN THE MODEL ###########
# Train the model:
# - For 5 epochs (complete passes through the training dataset)
# - With batch size of 64 (number of samples processed before model update)
# - Using 10% of training data as validation set
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
model.save("model.keras")
