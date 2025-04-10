import os

import gradio as gr
import keras
import numpy as np
from skimage.transform import resize

# Force CPU usage to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the pre-trained model
model = keras.models.load_model("model/model.keras")


def recognize_digit(image_dict):
    print(f"Input received: type={type(image_dict)}")

    # Check if input is None
    if image_dict is None:
        print("Received None input")
        return {str(i): 0.0 for i in range(10)}

    try:
        # Extract the image data from the dictionary
        if isinstance(image_dict, dict):
            print(f"Dictionary keys: {list(image_dict.keys())}")

            # For debugging, print a summary of each key's content
            for key in image_dict.keys():
                value = image_dict[key]
                if isinstance(value, np.ndarray):
                    print(
                        f"Key: {key}, Type: {type(value)}, Shape: {value.shape}, Min: {value.min()}, Max: {value.max()}"
                    )
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    print(f"Key: {key}, Type: list of dicts, Length: {len(value)}")
                else:
                    print(f"Key: {key}, Type: {type(value)}")

            # Try to use the composite image (this contains the drawing)
            if "composite" in image_dict and isinstance(
                image_dict["composite"], np.ndarray
            ):
                image = image_dict["composite"]
                print(f"Using composite image with shape {image.shape}")
            # Or try the layers if available
            elif (
                "layers" in image_dict
                and isinstance(image_dict["layers"], list)
                and image_dict["layers"]
            ):
                # Combine all layers
                layers = []
                for layer_dict in image_dict["layers"]:
                    if isinstance(layer_dict, dict) and "data" in layer_dict:
                        layer_data = layer_dict["data"]
                        if isinstance(layer_data, np.ndarray):
                            layers.append(layer_data)

                if layers:
                    # Combine layers (simple max value across layers)
                    image = layers[0].copy()
                    for layer in layers[1:]:
                        if layer.shape == image.shape:
                            image = np.maximum(image, layer)
                    print(
                        f"Combined {len(layers)} layers into image with shape {image.shape}"
                    )
                else:
                    # Fallback to background
                    image = image_dict["background"]
                    print(
                        f"No layer data found, using background with shape {image.shape}"
                    )
            else:
                # Fallback to background
                image = image_dict["background"]
                print(f"Using background image with shape {image.shape}")
        else:
            # If directly passed a numpy array (unlikely but possible)
            image = image_dict
            print(f"Input is directly a numpy array with shape {image.shape}")

        # Convert RGB to grayscale if needed
        if len(image.shape) == 3:
            print(
                f"Processing image with shape {image.shape}, min={image.min()}, max={image.max()}"
            )
            if image.shape[2] == 4:  # RGBA
                # Use alpha channel as mask if it exists and has variation
                alpha = image[:, :, 3]
                if alpha.min() != alpha.max():
                    print("Using alpha channel as mask")
                    image = alpha
                else:
                    image = np.mean(image[:, :, :3], axis=2)
                    print("Converted RGBA to grayscale (using RGB channels)")
            elif image.shape[2] == 3:  # RGB
                image = np.mean(image[:, :, :3], axis=2)
                print("Converted RGB to grayscale")
            else:
                image = image[:, :, 0]
                print(f"Used first channel of {image.shape[2]}-channel image")

        print(
            f"Before thresholding: min={image.min()}, max={image.max()}, mean={image.mean()}"
        )

        # Normalize to 0-1 range if needed
        if image.max() > 1.0:
            image = image / 255.0
            print(f"Normalized to 0-1 range: min={image.min()}, max={image.max()}")

        # Invert if needed - for Gradio's Sketchpad, black drawing on white background is common
        # So we actually want to invert ONLY if the background is black (mean close to 0)
        mean_val = np.mean(image)
        if mean_val < 0.5:
            # This means we have white digits on black background (already correct for MNIST)
            print(
                f"Image appears to have white digits on black background (mean={mean_val}), keeping as is"
            )
        else:
            # This means we have black digits on white background (need to invert for MNIST)
            image = 1.0 - image
            print(
                f"Inverted image from black-on-white to white-on-black (mean before={mean_val}, after={np.mean(image)})"
            )

        # Apply thresholding to clean up the image
        if image.max() > image.min():  # Only if there's variation in the image
            threshold = 0.3
            image = (image > threshold).astype(float)
            print(
                f"Applied threshold {threshold}, values after: min={image.min()}, max={image.max()}"
            )
        else:
            print("Skipping threshold as image has no variation")

        # Resize to 28x28 (MNIST format)
        image = resize(image, (28, 28), anti_aliasing=True)
        print(
            f"Resized to 28x28, values: min={image.min()}, max={image.max()}, mean={image.mean()}"
        )

        # Prepare for model input
        model_input = image.reshape(1, 28, 28, 1).astype("float32")

        # Get prediction from model
        prediction = model.predict(model_input, verbose=0)
        print(f"Raw prediction: {prediction[0]}")
        print(f"Sum of predictions: {np.sum(prediction[0])}")
        print(f"Most likely digit: {np.argmax(prediction[0])}")

        # Create result dictionary
        result = {str(i): float(prediction[0][i]) for i in range(10)}
        print(f"Result: {result}")

        return result

    except Exception as e:
        print(f"Error processing input: {str(e)}")
        import traceback

        traceback.print_exc()
        return {str(i): 0.0 for i in range(10)}


# Define a test function to verify model is working
def test_model():
    print("Testing model with sample data...")
    # Create a simple test image (a vertical line like in your screenshot)
    test_img = np.zeros((28, 28))
    test_img[:, 14] = 1.0  # Vertical line in the middle

    # Reshape for model input
    test_input = test_img.reshape(1, 28, 28, 1).astype("float32")

    # Get prediction
    try:
        test_pred = model.predict(test_input, verbose=0)
        print(f"Test prediction: {test_pred[0]}")
        print(f"Most likely digit: {np.argmax(test_pred[0])}")
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        import traceback

        traceback.print_exc()


# Create Gradio interface
iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(
        type="numpy",
        image_mode="L",  # Grayscale mode
        height=280,
        width=280,
        brush=gr.Brush(default_size=20, colors=["#000000"], color_mode="fixed"),
        label="Draw a digit here (0-9)",
    ),
    outputs=gr.Label(num_top_classes=10),
    live=False,  # Set to True for real-time prediction
    title="Digit Recognition",
    description="Draw a digit (0-9) on the canvas, and the model will predict what it is.",
    examples=[],  # You could add example inputs here
    cache_examples=False,
)

# Launch the interface
if __name__ == "__main__":
    # Test the model first
    test_model()

    # Launch the Gradio interface with debug mode
    iface.launch(debug=True)
