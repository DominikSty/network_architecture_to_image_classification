import tensorflow as tf
from PIL import Image

# Wczytanie danych MNIST
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Wybór przykładowego obrazu numeru
example_image = x_train[0]

# Konwersja obrazu na obiekt typu Image z użyciem biblioteki PIL
image = Image.fromarray(example_image)

# Zapisanie obrazu do pliku JPEG
image.save("image.jpg")