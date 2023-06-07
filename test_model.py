import tensorflow as tf
import numpy as np
from PIL import Image

def load_model_from_file(file_path):
    model = tf.keras.models.load_model(file_path)
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Konwersja obrazu do odcieni szarości
    image = image.resize((28, 28))  # Zmiana rozmiaru obrazu na 28x28 pikseli
    image = np.array(image) / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
    image = np.expand_dims(image, axis=0)  # Dodanie dodatkowego wymiaru dla wsadu (batch)
    return image

# Wczytanie modelu z pliku
loaded_model = load_model_from_file("model.h5")

# Przygotowanie obrazu testowego
image_path = "image_for_test/image_0.jpg"
preprocessed_image = preprocess_image(image_path)

# Klasyfikacja obrazu
predictions = loaded_model.predict(preprocessed_image)
predicted_class = np.argmax(predictions)

# Wyświetlenie wyników
print("Przewidziana klasa:", predicted_class)