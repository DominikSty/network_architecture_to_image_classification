import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import os

# Rozmiar pojedynczego piksela
PIXEL_SIZE = 10
# Rozmiar planszy (w pikselach)
BOARD_SIZE = 28

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Test modelu")
        window_width = 600
        window_height = 400
        # Ustaw szerokość i wysokość okna
        self.root.geometry(f"{window_width}x{window_height}")

        self.canvas = tk.Canvas(self.root, 
                                width=BOARD_SIZE * PIXEL_SIZE, 
                                height=BOARD_SIZE * PIXEL_SIZE, 
                                bg='black')
        self.canvas.pack()
        self.canvas.place(x=20, y=100)
        self.canvas.bind('<B1-Motion>', self.draw_pixel)

        self.button_save = tk.Button(self.root, text='Sprawdź', command=self.save_image)
        self.button_save.place(x=340, y=120)

        self.clear_button = tk.Button(self.root, text="Wyczyść", command=self.clear_canvas)
        self.clear_button.place(x=340, y=160)

        self.result_label = tk.Label(self.root, text="", fg="black", font=("Arial", 12))
        self.result_label.place(x=340, y=260)

        self.info_label1 = tk.Label(self.root, 
                                    text="Program przeznaczony do testowania modelu sieci do rozpoznawania obrazów", 
                                    fg="black", font=("Arial", 11))
        self.info_label1.place(x=10, y=10)
        self.info_label2 = tk.Label(self.root, 
                                    text="liczb, czarno-biały obraz w rozdzielczości 28x28 pikseli poddawany jest predykcji", 
                                    fg="black", font=("Arial", 11))
        self.info_label2.place(x=10, y=30)
        self.info_label3 = tk.Label(self.root, 
                                    text="na wyuczonym modelu i zwraca przewidywaną wartość. Poniżej znaduje się płutno, ", 
                                    fg="black", font=("Arial", 11))
        self.info_label3.place(x=10, y=50)
        self.info_label4 = tk.Label(self.root, 
                                    text="na którym PPM można narysować własną liczbę do sprawdzenia.", 
                                    fg="black", font=("Arial", 11))
        self.info_label4.place(x=10, y=70)

        self.image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def draw_pixel(self, event):
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE
        self.draw.point((x, y), fill='white')
        self.canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE,
                                     (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
                                     fill='white')
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def save_image(self):
        file_path = "buffor"
        self.image.save(file_path, "JPEG")
        # Test modelu
        self.model = tf.keras.models.load_model("src/model/model.h5")
        image_a = Image.open(file_path).convert('L')  # Konwersja obrazu do odcieni szarości
        image_a = image_a.resize((28, 28))            # Zmiana rozmiaru obrazu na 28x28 pikseli
        image_a = np.array(image_a) / 255.0           # Normalizacja wartości pikseli do zakresu 0-1
        image_a = np.expand_dims(image_a, axis=0)     # Dodanie dodatkowego wymiaru dla wsadu (batch)
        predictions = self.model.predict(image_a)
        predicted_class = np.argmax(predictions)
        print("Wynik:", predicted_class)
        self.result_label.config(text="Przewidywana liczba: " + str(predicted_class))
        os.remove(file_path)


root = tk.Tk()
app = DrawingApp(root)
root.mainloop()