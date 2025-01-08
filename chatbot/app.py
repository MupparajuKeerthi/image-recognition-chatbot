import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from PIL import Image
import numpy as np
import cv2
import onnxruntime
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from werkzeug.utils import secure_filename
import gdown

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Folder configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['UPSCALED_FOLDER'] = 'static/upscaled'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPSCALED_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model paths
models_folder = "models"
os.makedirs(models_folder, exist_ok=True)
modelx2_file_path = os.path.join(models_folder, "modelx2.ort")

# BLIP models for captioning and VQA
caption_processor = BlipProcessor.from_pretrained("models/blip-captioning")
caption_model = BlipForConditionalGeneration.from_pretrained("models/blip-captioning")

vqa_processor = BlipProcessor.from_pretrained("models/blip-vqa")
vqa_model = BlipForQuestionAnswering.from_pretrained("models/blip-vqa")

# Download ONNX model
modelx2_file_id = "1Hvt3_t8S2W5CNYUCFgd2L_KitedAJEmH"
if not os.path.exists(modelx2_file_path):
    url = f"https://drive.google.com/uc?export=download&id={modelx2_file_id}"
    gdown.download(url, modelx2_file_path, quiet=False)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_pil_to_cv2(image):
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def pre_process(img: np.array) -> np.array:
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def post_process(img: np.array) -> np.array:
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
    return img

def inference(model_path: str, img_array: np.array) -> np.array:
    options = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(model_path, options)
    ort_inputs = {ort_session.get_inputs()[0].name: img_array}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def upscale(image_path: str):
    pil_image = Image.open(image_path)
    img = convert_pil_to_cv2(pil_image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image_output = post_process(inference(modelx2_file_path, pre_process(img)))
    image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)
    return image_output

def generate_caption(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(images=image, return_tensors="pt")
    output = caption_model.generate(**inputs)
    return caption_processor.decode(output[0], skip_special_tokens=True)

def answer_question(image_path: str, question: str):
    image = Image.open(image_path).convert("RGB")
    inputs = vqa_processor(images=image, text=question, return_tensors="pt")
    output = vqa_model.generate(**inputs)
    return vqa_processor.decode(output[0], skip_special_tokens=True)

def find_similar_images(image_path: str):
    """
    Placeholder for image similarity search.
    Replace with your implementation (e.g., feature extraction + similarity scoring).
    """
    return [
        "https://via.placeholder.com/150/0000FF",  # Replace with real URLs
        "https://via.placeholder.com/150/FF0000",
        "https://via.placeholder.com/150/00FF00",
    ]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('process_image', filename=filename))
    else:
        flash('Invalid file type. Please upload PNG, JPG, or JPEG.')
        return redirect(url_for('index'))

@app.route('/process/<filename>')
def process_image(filename):
    return render_template('process.html', filename=filename)

@app.route('/enhance/<filename>')
def enhance_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    upscaled_image = upscale(file_path)
    upscaled_filename = f'upscaled_{filename}'
    upscaled_path = os.path.join(app.config['UPSCALED_FOLDER'], upscaled_filename)
    cv2.imwrite(upscaled_path, upscaled_image)
    return render_template('enhance.html', original_filename=filename, upscaled_filename=upscaled_filename)

@app.route('/caption/<filename>')
def caption_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    caption = generate_caption(file_path)
    return render_template('caption.html', filename=filename, caption=caption)

@app.route('/vqa/<filename>', methods=['GET', 'POST'])
def vqa_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = answer_question(file_path, question)
            return render_template('vqa.html', filename=filename, question=question, answer=answer)
        else:
            flash('Please enter a question.')
    return render_template('vqa.html', filename=filename)

@app.route('/similarity/<filename>')
def similarity_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    similar_images = find_similar_images(file_path)
    return render_template('similarity.html', filename=filename, similar_images=similar_images)

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
