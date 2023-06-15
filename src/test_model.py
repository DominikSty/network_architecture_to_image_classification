import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import os


# Single pixel size
PIXEL_SIZE = 10
# Board size (pixels)
BOARD_SIZE = 28

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Test")
        window_width = 600
        window_height = 400
        # Set the width and height of the window
        self.root.geometry(f"{window_width}x{window_height}")

        self.canvas = tk.Canvas(self.root, 
                                width=BOARD_SIZE * PIXEL_SIZE, 
                                height=BOARD_SIZE * PIXEL_SIZE, 
                                bg='black')
        self.canvas.pack()
        self.canvas.place(x=20, y=100)
        self.canvas.bind('<B1-Motion>', self.draw_pixel)

        self.button_save = tk.Button(self.root, text='Check', command=self.save_image)
        self.button_save.place(x=340, y=120)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.place(x=340, y=160)

        self.result_label = tk.Label(self.root, text="", fg="black", font=("Arial", 12))
        self.result_label.place(x=340, y=260)

        self.info_label1 = tk.Label(self.root, 
                                    text="Program designed to test network model for image recognition numbers,", 
                                    fg="black", font=("Arial", 11))
        self.info_label1.place(x=10, y=10)
        self.info_label2 = tk.Label(self.root, 
                                    text="a black and white image with a resolution of 28x28 pixels is subject to prediction", 
                                    fg="black", font=("Arial", 11))
        self.info_label2.place(x=10, y=30)
        self.info_label3 = tk.Label(self.root, 
                                    text="on the learned model and returns the predicted value. Below is a canvas,", 
                                    fg="black", font=("Arial", 11))
        self.info_label3.place(x=10, y=50)
        self.info_label4 = tk.Label(self.root, 
                                    text="on which PPM can draw its own number to check.", 
                                    fg="black", font=("Arial", 11))
        self.info_label4.place(x=10, y=70)

        self.image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)

    # def draw_pixel(self, event):
    #     x = event.x // PIXEL_SIZE
    #     y = event.y // PIXEL_SIZE
    #     self.draw.point((x, y), fill='white')
    #     self.canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE,
    #                                  (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
    #                                  fill='white')

    def draw_pixel(self, event):
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE

        self.draw.point((x, y), fill='white')
        self.canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE,
                                     (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
                                     fill='white')
        # Odczytanie koloru piksela z oryginalnego obrazu
        pixel_color = self.image.getpixel((x, y))

        if pixel_color == (255, 255, 255):
           # Rysowanie piksela o odcieniu szarości
            self.draw.point((x, y), fill='gray')
            radius =  event.x // PIXEL_SIZE //event.y # Promień okręgu
            self.canvas.create_rectangle(
                x * PIXEL_SIZE - radius,
                y * PIXEL_SIZE - radius,
                x * PIXEL_SIZE + radius,
                y * PIXEL_SIZE + radius,
                fill='gray'
            )

        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def save_image(self):
        file_path = "buffor.jpg"
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
       # os.remove(file_path)

    

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()