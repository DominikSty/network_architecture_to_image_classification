# Design from the ground up of network architecture to simple image classification

### Introduction

The theme of the project is to design from scratch a network architecture for simple image classification. This means implementing and programming a working program that will be based on artificial intelligence. The network design will involve the design of layers of the model, which will later be taught on the learners’ data and a qualitative check on the test data. The final stage of the project will be to test the model on data and evaluate its effectiveness. The most important thing when creating a project will be to select the appropriate layers of the model and determine their learning parameters in order to achieve the best possible results. Most of the parameters will depend on the selected database and its subject.

### 1. Artificial Intelligence Network Architecture
Machine learning network architecture is a structure or arrangement of layers and connections in a neural network that is used to solve a specific problem. Machine learning networks are built of units called neurons that process input and generate responses based on patterns and rules contained in the training set provided.
The basic element of the network architecture is the neuron, which receives the input data, performs certain operations on that data and generates the result. Neurons are organized into layers, and connections between neurons are determined by scales that determine the effect of a given neuron on other neurons in the network.

### 2. CNN Network Architecture
CNN (Convolutional Neural Network) architecture is a type of deep neural network used for data processing and analysis, especially in the field of image processing. CNN is widely used in applications related to image recognition, object detection, image classification, image segmentation and many other video processing tasks.

### 3. DNN Network Architecture
Deep Neural Network (DNN) architecture is a type of neural network with many hidden layers that enable a model to learn complex data representations. DNN is also known as Multilayer Perceptron (MLP).

### 4. Combining CNN and DNN architecture
The combination of DNN and CNN network architectures in one model allows for efficient use of the advantages of both types of networks. CNN can be used as part of the initial feature extraction, transforming the input images into more condensed representations. These representations can then be passed to the DNN network, which learns the relationships between these features and performs more complex tasks such as image classification, object detection, and image segmentation.

## NETWORK ARCHITECTURE DESIGN FOR IMAGE CLASSIFICATION
The project presenting the theme of designing from the basics of network architecture to simple image classification was implemented in the Python programming language due to the large volume of libraries supporting artificial intelligence modeling. A set of handwritten digits (0-9) with a size of 28x28 pixels was adopted as the base. An example of such a set can be found in the tensorflow library package named mnist. The exact path is: ```tensorflow.keras.datasets.mnist```

![image](https://github.com/DominikSty/network_architecture_to_image_classification/assets/101213292/e8ff2c30-abdf-4b39-930b-20965840433c)

The first step in designing an architecture is to create a model implementation, because of the combined DNN and CNN models, they must be implemented first.

### MODEL CNN

```
# Create model CNN
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(20, (5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(numClass, activation='softmax')
])
```

• The input layer has a shape (28, 28, 1) corresponding to the dimensions of the input images.

• Then there is a convolution layer (tf. keras. layers. Conv2D) with 20 filters of size (5, 5). These filters are used to extract features from images.

• Convulsion is followed by batch normalization (tf. keras. layers. BatchNormalization()), which regulates the input statistics to the next layer.

• ReLU activation (tf. keras. layers. Activation('relu')) is used to activate the results of the previous layer.

• The Flatten() layer converts data from a multidimensional tensor to a one-dimensional tensor, preparing it for dense layers.

• Dense layer (tf. keras. layers. Dense) of numClass size (10 in this case) with softmax activation, which generates probabilities of belonging to particular classes.

### MODEL DNN

```
# Create model DNN
dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((numClass,), input_shape=(numClass,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(numClass, activation='softmax')
])
```

• The Reshape() input layer converts the input from a tensor of numClass size to a one-dimensional tensor of the same size.

• The next three dense layers (tf. keras. layers. Dense) have 256, 128 and 64 units respectively, benefit from ReLU activation.

• The last thick layer is numClass size with softmax activation.

### COMBINED MODEL DNN & CNN

A combined model is then created, which consists of an input layer with dimensions (28, 28, 1), a CNN model and a DNN model. This is done using tf. keras. models. Sequential(). The input passes through the CNN model, and then the output passes through the DNN model. Once a linked model is created, it is compiled using the compile() function. Adam is used as the optimizer and the loss function is sparse_categorical_crossentropy. The metric used to evaluate model performance is accuracy.

```
# Creating a Connected Model
combined_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    cnn_model,
    dnn_model
])
# Model Compilation
combined_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
```

### Model training
In the next step, the model designed in this way needs to be taught on training data and to check its progress using test data.

```
# Loading training and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Division of training data into training and validation set
val_split = 0.1
val_samples = int(len(x_train) * val_split)
x_val = x_train[:val_samples]
y_val = y_train[:val_samples]
x_train = x_train[val_samples:]
y_train = y_train[val_samples:]

# Preparation of training data
x_train = x_train.reshape(-1, 28, 28)  # Removing dimension for channel
x_train = x_train / 255.0  # Normalize pixel values to 0-1 range
y_train = y_train.astype(int)

# Preparation of validation data
x_val = x_val.reshape(-1, 28, 28)  # Removing dimension for channel
x_val = x_val / 255.0  # Normalize pixel values to 0-1 range
y_val = y_val.astype(int)
```

Training data consists of image pairs (x_train) and corresponding tags (y_train), as well as test data (x_test and y_test). The val_split parameter specifies the proportion of data that will be used as the validation set (in this case 0.1, i.e. 10%). It then determines the number of validation examples (val_samples) based on this ratio and separates these examples from the training data. 
To prepare training and test data, reshape()) to (-1, 28, 28) to remove the dimension for the channel (data is grayscale, so there are no color channels), and then normalize the pixel values to the range 0-1 dividing by 255.0.
The most important part is to train the model through the fit() function, which is written to the history variable for later analysis of the resulting process. Training data and training labels as arguments, as well as validation data and validation labels, are given by the parameter validation_data and the number of training epochs (in this case 15) is also specified.

```
# Model training with validation data
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15)
```

The model learned in this way is saved to the file ‘model.h5’ for later use.

```
# Writing the model to a file
model.save('src/model/model.h5')
```

The parameters stored in ‘history’ are the history of the learning process of the model, which can be easily visualized using the matplotlib library using the pyplot function.

```
# Learning Process Charts
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

The last element is to display a summary of the whole process by evaluating the accuracy of the model on the test data.

```
# Preparation of test data
x_test = x_test.reshape(-1, 28, 28)  # Removing dimension for channel
x_test = x_test / 255.0  # Normalize pixel values to 0-1 range
y_test = y_test.astype(int)

# Assessment of model accuracy on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Model accuracy:', test_acc)
```

## RUN THE MODEL CREATION SCRIPT

The process of learning the model over 15 epochs, where the average time of one epoch is 20 seconds. Each of the eras contains 1,688 steps of learning. As you go through each epoch, you can see how much the ratio of loss value and model accuracy changes.

```
Epoch 1/15
1688/1688 [==============================] - 20s 11ms/step - loss: 1.1210 - accuracy: 0.4787 - val_loss: 1.0058 - val_accuracy: 0.5047
Epoch 2/15
1688/1688 [==============================] - 19s 11ms/step - loss: 0.8360 - accuracy: 0.5838 - val_loss: 0.3763 - val_accuracy: 0.8620
Epoch 3/15
1688/1688 [==============================] - 19s 11ms/step - loss: 0.1440 - accuracy: 0.9596 - val_loss: 0.1060 - val_accuracy: 0.9688
...
Epoch 13/15
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0109 - accuracy: 0.9966 - val_loss: 0.0814 - val_accuracy: 0.9832
Epoch 14/15
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0105 - accuracy: 0.9969 - val_loss: 0.0853 - val_accuracy: 0.9813
Epoch 15/15
1688/1688 [==============================] - 19s 11ms/step - loss: 0.0108 - accuracy: 0.9969 - val_loss: 0.0884 - val_accuracy: 0.9825
```

The visualization of the learning process is contained in two graphs. The first includes the quality of the ‘loss’ function in relation to the number of eras, while the second value of the ‘accuracy’ function of the model. The whole code implementation can be found in the file ‘src/create_model.py’.

![image](https://github.com/DominikSty/network_architecture_to_image_classification/assets/101213292/ff7af813-2c17-4089-bbb1-929372a641ba)

The end result is a description of the values by testing the model on test data. It returns the qualitative accuracy value of the model.

```
313/313 - 1s - loss: 0.0878 - accuracy: 0.9833 - 1s/epoch - 4ms/step
Model accuracy: 0.983299970626831
```

## TESTING MODEL

A GUI has been created to show how the model functions. It is a simple window application containing a short description of the operation and a canvas designed to create your own graphics with a resolution of 28x28 pixels. For self-creation of digits to test the model. It contains a canvas class from the tkinter library.

```
def draw_pixel(self, event):
    x = event.x // PIXEL_SIZE
    y = event.y // PIXEL_SIZE
    self.draw.point((x, y), fill='white')
    self.canvas.create_rectangle(x * PIXEL_SIZE, 
                                 y * PIXEL_SIZE,
                                 (x + 1) * PIXEL_SIZE, 
                                 (y + 1) * PIXEL_SIZE,
                                 fill='white')
```

Drawing by pressing PPM and activating the “Check” button will cause the drawn image to be tested on the model and return the expected number (the one with the highest result in the prediction). The function of saving the image from the canvas and using it in the comparison function is responsible for this. The complete implementation of the program can be found in the file ‘src/test_model. py’.

```
file_path = "buffor"
self.image.save(file_path, "JPEG")
# Model test
self.model = tf.keras.models.load_model("src/model/model.h5")
image_a = Image.open(file_path).convert('L')  # Convert image to grayscale
image_a = image_a.resize((28, 28))            # Resize image to 28x28 pixels
image_a = np.array(image_a) / 255.0           # Normalize pixel values to 0-1 range
image_a = np.expand_dims(image_a, axis=0)     # Adding an extra dimension for the batch
predictions = self.model.predict(image_a)
predicted_class = np.argmax(predictions)
print("Score:", predicted_class)
self.result_label.config(text="Anticipated number: " + str(predicted_class))
os.remove(file_path)
```

### EXAMPLE OF RUN "TEST MODEL"

![image](https://github.com/DominikSty/network_architecture_to_image_classification/assets/101213292/e48a068d-850d-46bf-8f07-2c37351fbc1b)

```
1/1 [==============================] - 0s 156ms/step
Score: 2
```

![image](https://github.com/DominikSty/network_architecture_to_image_classification/assets/101213292/1ea94f64-9be3-4351-b74f-3f588c8d225a)

```
1/1 [==============================] - 0s 78ms/step
Score: 8
```

![image](https://github.com/DominikSty/network_architecture_to_image_classification/assets/101213292/d278b5f0-8399-441d-a86b-17e13f9762ad)

```
1/1 [==============================] - 0s 87ms/step
Score: 0
```
