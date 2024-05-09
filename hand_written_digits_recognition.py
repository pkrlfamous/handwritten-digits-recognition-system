import os.path
import tensorflow as tf
import numpy as np
# for Deeplearning architecture sequential connects different deep learning layers
# the deep learning layers are dense, dropout, activation and so on
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
import tkinter as tk
from tkinter import *
import win32gui
from PIL import ImageGrab

trainModel = True
if trainModel:
    if os.path.isfile('trainedModel.h5'):
        print('trainedModel.h5 already exists, no need to train the model. Try after deleting trainedModel.h5')
    
    else:
        #loading dataset
        mnistDataset = tf.keras.datasets.mnist

        #dividing the data into training and testing datasets
        (x_train, y_train), (x_test, y_test) = mnistDataset.load_data()

        # values are between 0 to 255 cause it is a grayscale image so normalizing it

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        
        # this will print values between 0 and 1 cause values are normalized
        # print(x_train[0])

        # resizing the image to fit into for CNN
        image_size = 28
        # increasing one dimension for kernel (filter) operation
        x_trainResized = np.array(x_train).reshape(-1, image_size, image_size, 1)
        x_testResized = np.array(x_test).reshape(-1, image_size, image_size, 1)

        # creating neural network
        model = Sequential()

        # first convolutional layer
        model.add(Conv2D(64, (3,3), input_shape = x_trainResized.shape[1:])) # only for first convolution layer to mention input layer size
        model.add(Activation("relu")) # activation function to make it non-linear, i.e <0 remove and >0 forward
        model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling size of 2x2

        # second layer
        model.add(Conv2D(64, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # third layer
        model.add(Conv2D(64, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # first fully connected convolutional layer
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))

        # second fully connected layer
        model.add(Dense(32))
        model.add(Activation("relu"))

        # final fully connected layer, must be equal to number of classes or classifications
        model.add(Dense(10))
        model.add(Activation('softmax')) # activation function is changed to Softmax (class probabilities)

        model.summary()

        # compiling the model
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        # training the model with 5 epochs
        model.fit(x_trainResized, y_train, epochs=5, validation_split = 0.3)

        model.save('trainedModel.h5')
        print('Model is saved')
else:
    print(" trainModel is false, try by changing 'trainModel' to True")

try:
    model = load_model('trainedModel.h5')
except:
    print("trainModel.h5 is not present in the directory, try by creating trainModel.h5 file")
    exit()

def predict(image): # takes image and uses model to predict
    image = image.resize((28,28))

    #converting the image to grayscale
    image = image.convert('L')
    image = np.array(image)

    #reshaping the image to feed our model
    image = image.reshape(1,28,28,1)
    image = image/255.0

    prediction = model.predict([image])[0]
    return np.argmax(prediction), max(prediction)

class GUI(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)
        self.title('Handwritten digit recognition system')
        self.x = self.y = 0
        self.resizable(0,0)


        # Initializing the elements and the canvas
        self.canvas = tk.Canvas(self, height = 350, width = 350, cursor = "cross",  bg = "black")
        self.clear_button = tk.Button(self, text = "Clear All", command = self.delete_drawing, bg='red')
        self.categorize_button = tk.Button(self, text = "Predict", command = self.categorize_digits, bg='green')
        self.label = tk.Label(self, text="Write a digit", bg='skyblue', font=("Arial", 10))
       
        # Placing the labels and buttons in the grid
        self.canvas.grid(row=0, column=0, pady=2)
        self.clear_button.grid(row=1, column=0, padx=2, pady=2)
        self.categorize_button.grid(row=1, column=1, pady=2, padx=2)
        self.label.grid(row=0, column=1, padx=2)
        
        # binding methods and functions to the events like moving cursor and clicking
        self.canvas.bind("<B1-Motion>", self.write)
    

    def delete_drawing(self):
        self.canvas.delete("all")
        self.label.pack_forget()
        
    def categorize_digits(self):
        # getting the handle of the canvas
        handle = self.canvas.winfo_id()
        # getting the coordinate of the canvas
        coordinates = win32gui.GetWindowRect(handle)

        p, q, r, s = coordinates
        coordinates = ( p+4, q+4, r-4, s-4)
        drawn_image = ImageGrab.grab(coordinates) # copies contents of the screen to PIL memory

        digit, accuracy = predict(drawn_image)
        self.label.configure(text='The digit probably be:'+ str(digit) +', and the accuracy is '+ str(int(accuracy*100))+'%')

    def write(self, event):
        self.x = event.x
        self.y = event.y
        radius = 9
        self.canvas.create_oval(self.x - radius + 1, self.y - radius, self.x + radius, self.y + radius, fill = 'white')
       
gui = GUI()
mainloop() # event loops and listens for button click or keypress events
