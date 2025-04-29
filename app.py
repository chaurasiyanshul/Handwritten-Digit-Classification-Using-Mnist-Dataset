import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow INFO/WARNING logs

from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import pickle
import io

app = Flask(__name__)

# Load the pre-trained pickled Keras model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Reshape to (1, 28, 28) to match model input
    image_array = image_array.reshape(1, 28, 28)
    return image_array

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No image uploaded')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No image selected')
        
        try:
            # Read and preprocess the image
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image.reshape(1,28,28)).argmax(axis=1)              
            predicted_digit = np.argmax(prediction)
            
            return render_template('index.html', prediction=predicted_digit)
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)