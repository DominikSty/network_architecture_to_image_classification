import tensorflow as tf
import matplotlib.pyplot as plt


def create_combined_model():
    numClass = 10  # Number of Classes to Classify

    # Create model CNN
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (5, 5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(numClass, activation='softmax')
    ])

    # Create model DNN
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((numClass,), input_shape=(numClass,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(numClass, activation='softmax')
    ])

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

    return combined_model


# Creating a Model Instance
model = create_combined_model()

# Wyświetlanie macierzy wag przed treningiem

for layer in model.layers:
    if len(layer.get_weights()) > 0:
        weights = layer.get_weights()[0]
        print(f"Initial weights for layer {layer.name}:")
        print(weights)
        print()

plt.plot(weights)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.title('Ogólny pogląd wag przed funkcji uczenia')

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

# Model training with validation data
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15)

# Wyświetlanie macierzy wag po treningu
for layer in model.layers:
    if len(layer.get_weights()) > 0:
        weights = layer.get_weights()[0]
        print(f"Trained weights for layer {layer.name}:")
        print(weights)
        print()


plt.plot(weights)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ogólny pogląd wag po funkcji uczenia')
plt.show()

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

# Preparation of test data
x_test = x_test.reshape(-1, 28, 28)  # Removing dimension for channel
x_test = x_test / 255.0  # Normalize pixel values to 0-1 range
y_test = y_test.astype(int)

# Assessment of model accuracy on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Model accuracy:', test_acc)



# Writing the model to a file
model.save('src/model/model.h5')
