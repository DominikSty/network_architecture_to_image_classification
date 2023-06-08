# Temat projektu: Projektowanie od podstaw architektury sieci do prostej klasyfikacji obrazów

import tensorflow as tf
import matplotlib.pyplot as plt


def create_combined_model():
    numClass = 10  # Liczba klas do sklasyfikowania

    # Tworzenie modelu CNN
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (5, 5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(numClass, activation='softmax')
    ])

    # Tworzenie modelu DNN
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((numClass,), input_shape=(numClass,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(numClass, activation='softmax')
    ])

    # Tworzenie połączonego modelu
    combined_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        cnn_model,
        dnn_model
    ])

    # Kompilacja modelu
    combined_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    return combined_model


# Tworzenie instancji modelu
model = create_combined_model()

# Wczytanie danych treningowych i testowych
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Podział danych treningowych na zbiór treningowy i walidacyjny
val_split = 0.1
val_samples = int(len(x_train) * val_split)
x_val = x_train[:val_samples]
y_val = y_train[:val_samples]
x_train = x_train[val_samples:]
y_train = y_train[val_samples:]

# Przygotowanie danych treningowych
x_train = x_train.reshape(-1, 28, 28)  # Usunięcie wymiaru dla kanału
x_train = x_train / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
y_train = y_train.astype(int)

# Przygotowanie danych walidacyjnych
x_val = x_val.reshape(-1, 28, 28)  # Usunięcie wymiaru dla kanału
x_val = x_val / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
y_val = y_val.astype(int)

# Trenowanie modelu z danymi walidacyjnymi
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15)

# Wykresy procesu uczenia się
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.show()

# Przygotowanie danych testowych
x_test = x_test.reshape(-1, 28, 28)  # Usunięcie wymiaru dla kanału
x_test = x_test / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
y_test = y_test.astype(int)

# Ocena dokładności modelu na danych testowych
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Dokładność modelu:', test_acc)

# Zapis modelu do pliku
model.save('src/model/model.h5')



# import tensorflow as tf

# # Definicja architektury sieci CNN
# def create_cnn_model():
#     numClass = 10  # Liczba klas do sklasyfikowania
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Input(shape=(28, 28, 1)),  # Warstwa wejściowa - obraz o rozmiarze 28x28 pikseli i 1 kanał
#         tf.keras.layers.Conv2D(20, (5, 5)),  # Warstwa konwolucyjna z 20 filtrami o rozmiarze 5x5
#         tf.keras.layers.BatchNormalization(),  # Warstwa normalizacji wsadowej
#         tf.keras.layers.Activation('relu'),  # Warstwa aktywacji ReLU
#         tf.keras.layers.Flatten(),  # Warstwa spłaszczająca
#         tf.keras.layers.Dense(numClass, activation='softmax')  # Warstwa gęsta z funkcją aktywacji softmax
#     ])

#     # Kompilacja modelu
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

#     return model

# # Tworzenie instancji modelu
# model = create_cnn_model()

# # Wczytanie danych treningowych i testowych
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Przygotowanie danych treningowych
# x_train = x_train.reshape(-1, 28, 28, 1)  # Dodanie wymiaru dla kanału
# x_train = x_train / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
# y_train = y_train.astype(int)

# # Trenowanie modelu
# model.fit(x_train, y_train, epochs=15)

# # Przygotowanie danych testowych
# x_test = x_test.reshape(-1, 28, 28, 1)  # Dodanie wymiaru dla kanału
# x_test = x_test / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
# y_test = y_test.astype(int)

# # Ocena dokładności modelu na danych testowych
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print('Dokładność modelu:', test_acc)

# # Zapis modelu do pliku
# model.save('model.h5')


# import tensorflow as tf

# # Definicja architektury sieci DNN
# def create_dnn_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28)),  # Warstwa wejściowa - spłaszczenie obrazu 28x28 pikseli
#         tf.keras.layers.Dense(256, activation='relu'),  # Warstwa ukryta z 256 neuronami i funkcją aktywacji ReLU
#         tf.keras.layers.Dense(128, activation='relu'),  # Dodatkowa warstwa ukryta z 128 neuronami i funkcją aktywacji ReLU
#         tf.keras.layers.Dense(64, activation='relu'),   # Dodatkowa warstwa ukryta z 64 neuronami i funkcją aktywacji ReLU
#         tf.keras.layers.Dense(10, activation='softmax')  # Warstwa wyjściowa z 10 neuronami (klasyfikacja na 10 klas) i funkcją aktywacji softmax
#     ])
    
#     # Kompilacja modelu
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Tworzenie instancji modelu
# model = create_dnn_model()

# # Wczytanie danych treningowych i testowych
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Przygotowanie danych treningowych
# x_train = x_train / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
# y_train = y_train.astype(int)

# # Trenowanie modelu
# model.fit(x_train, y_train, epochs=15)

# # Przygotowanie danych testowych
# x_test = x_test / 255.0  # Normalizacja wartości pikseli do zakresu 0-1
# y_test = y_test.astype(int)

# # Ocena dokładności modelu na danych testowych
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print('Dokładność modelu:', test_acc)

# # Zapis modelu do pliku
# model.save('model.h5')