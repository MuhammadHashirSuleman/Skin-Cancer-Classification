from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import logging
import pickle

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load model and label map
model = tf.keras.models.load_model('../models/skin_cancer_cnn_model.keras')
label_map_path = '../models/label_map.pkl'
if not os.path.exists(label_map_path):
    raise FileNotFoundError("label_map.pkl not found. Ensure it exists in the directory.")
with open(label_map_path, 'rb') as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}
dangerous_classes = ['mel', 'bcc', 'akiec']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (128, 128)  # Matches your model's input size

def predict_image(filepath, model, label_map):
    """Predict the class of a single image."""
    img = Image.open(filepath).resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    if img_array.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
        raise ValueError("Invalid image format or size")
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    pred_class = np.argmax(prediction)
    confidence = prediction[pred_class]
    return pred_class, confidence

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_3', pred_index=None):
    """Generate Grad-CAM heatmap for a given image."""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logging.error(f"Error in make_gradcam_heatmap: {str(e)}")
        raise

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skin Cancer Classification</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-2xl">
        <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">Skin Lesion Classifier</h1>
        <form method="post" enctype="multipart/form-data" class="space-y-4">
            <div class="flex items-center justify-center">
                <label class="w-full flex flex-col items-center px-4 py-6 bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:bg-gray-100">
                    <svg class="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V8m0 0l-4 4m4-4l4 4m6-4v8m-8 0h12"></path>
                    </svg>
                    <span class="mt-2 text-sm text-gray-600">Upload a PNG, JPG, or JPEG image</span>
                    <input type="file" name="file" accept=".png,.jpg,.jpeg" class="hidden">
                </label>
            </div>
            <div class="text-center">
                <button type="submit" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition">Analyze Image</button>
            </div>
        </form>
        {% if error %}
            <p class="mt-4 text-red-600 text-center font-medium">{{ error }}</p>
        {% endif %}
        {% if prediction %}
            <div class="mt-6 p-4 bg-gray-50 rounded-lg">
                <h2 class="text-xl font-semibold text-gray-800">Prediction: {{ prediction }}</h2>
                <h3 class="text-lg text-gray-600 mt-2">Confidence: {{ confidence }}%</h3>
                {% if warning %}
                    <p class="mt-3 text-red-600 font-bold">Warning: Dangerous skin cancer detected!</p>
                {% endif %}
                {% if gradcam_img %}
                    <h3 class="text-lg font-semibold text-gray-800 mt-4">Grad-CAM Heatmap:</h3>
                    <img src="data:image/png;base64,{{ gradcam_img }}" alt="Grad-CAM Heatmap" class="mt-2 max-w-xs mx-auto rounded-lg shadow-md">
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    confidence = None
    warning = False
    gradcam_img = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded"
            logging.error("No file uploaded")
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected"
                logging.error("No file selected")
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload a PNG, JPG, or JPEG image."
                logging.error(f"Invalid file type: {file.filename}")
            else:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join('Uploads', filename)
                    os.makedirs('Uploads', exist_ok=True)
                    file.save(filepath)
                    pred_class, conf = predict_image(filepath, model, label_map)
                    prediction = inv_label_map[pred_class]
                    confidence = round(conf * 100, 2)
                    logging.info(f"Prediction: {prediction}, Confidence: {confidence}, File: {filename}")
                    if prediction in dangerous_classes:
                        warning = True

                    # Grad-CAM
                    img = Image.open(filepath).resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0
                    if img_array.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
                        error = "Invalid image format or size"
                        logging.error(f"Invalid image format: {filename}")
                    else:
                        img_array = np.expand_dims(img_array, axis=0)
                        heatmap = make_gradcam_heatmap(img_array, model)
                        plt.figure(figsize=(3, 3))
                        plt.imshow(img)
                        plt.imshow(heatmap, cmap='jet', alpha=0.5)
                        buf = io.BytesIO()
                        plt.axis('off')
                        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close()
                        buf.seek(0)
                        gradcam_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    logging.error(f"Error processing image: {str(e)}, File: {filename}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence, 
                                 warning=warning, gradcam_img=gradcam_img, error=error)

if __name__ == '__main__':
    app.run(debug=False, port=5000)