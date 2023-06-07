# Temat projektu: Projektowanie od podstaw architektury sieci do prostej klasyfikacji obrazów

import tensorflow as tf

# Definicja architektury sieci DNN
def create_dnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Warstwa wejściowa - spłaszczenie obrazu 28x28 pikseli
        tf.keras.layers.Dense(256, activation='relu'),  # Warstwa ukryta z 256 neuronami i funkcją aktywacji ReLU
        tf.keras.layers.Dense(128, activation='relu'),  # Dodatkowa warstwa ukryta z 128 neuronami i funkcją aktywacji ReLU
        tf.keras.layers.Dense(64, activation='relu'),   # Dodatkowa warstwa ukryta z 64 neuronami i funkcją aktywacji ReLU
        tf.keras.layers.Dense(10, activation='softmax')  # Warstwa wyjściowa z 10 neuronami (klasyfikacja na 10 klas) i funkcją aktywacji softmax
    ])
    
    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Tworzenie instancji modelu
model = create_dnn_model()

# Wczytanie danych treningowych i testowych
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Przygotowanie danych treningowych
x_train = x_train / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
y_train = y_train.astype(int)

# Trenowanie modelu
model.fit(x_train, y_train, epochs=15)

# Przygotowanie danych testowych
x_test = x_test / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
y_test = y_test.astype(int)

# Ocena dokładności modelu na danych testowych
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Dokładność modelu:', test_acc)

# Zapis modelu do pliku
model.save('model.h5')