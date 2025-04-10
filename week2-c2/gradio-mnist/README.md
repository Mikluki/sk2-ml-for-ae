# Handwritten Digit Recognition with CNN

A simple web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Structure

```
.
├── app.py           # Gradio web interface for digit recognition
├── model
│   └── model.keras  # Trained CNN model
├── README.md        # This file
└── training.py      # Script to train the CNN model
```

## Overview

This project implements a handwritten digit recognition system with two main components:

1. A CNN model trained on the MNIST dataset (`training.py`)
2. A web interface built with Gradio that allows users to draw digits and get real-time predictions (`app.py`)

## Model Architecture

The CNN architecture consists of:

- 3 convolutional layers with ReLU activation
- 2 max pooling layers
- Fully connected layers with 64 neurons and a 10-neuron output layer (softmax)

## Usage

1. Train the model (there is also a pretrained model saved @ `model/model.keras`):

   ```
   python training.py
   ```

2. Launch the web application:

   ```
   python app.py
   ```

3. Draw a digit on the canvas and submit to see the prediction.

## Dependencies

- Keras
- Gradio
- NumPy
- scikit-image
