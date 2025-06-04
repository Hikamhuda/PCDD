import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Fungsi untuk memeriksa apakah ekstensi file diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi (simulasi) untuk mendeteksi tangan "hi five" (perlu implementasi ML/CV)
def detect_hi_five(frame):
    # Implementasi deteksi tangan "hi five" menggunakan OpenCV atau model deteksi objek
    # Contoh sederhana: mendeteksi area dengan banyak "kulit" dan bentuk tertentu
    # Ini adalah placeholder dan memerlukan implementasi yang lebih canggih
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ... logika deteksi ...
    # Jika terdeteksi, kembalikan True
    return False  # Sementara selalu kembalikan False

# Fungsi untuk melakukan penyesuaian gambar
def adjust_image(image_path, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, noise_reduction=0):
    try:
        img = Image.open(image_path)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

        if noise_reduction > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=noise_reduction))

        img.save(image_path)
        return True
    except Exception as e:
        print(f"Error adjusting image: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    # Ini akan dipicu oleh JavaScript saat tombol capture ditekan
    # Implementasi pengambilan gambar dari kamera ada di JavaScript (script.js)
    # Di sini kita hanya menerima nama file yang telah disimpan
    filename = request.form.get('filename')
    if filename:
        return redirect(url_for('edit', filename=filename))
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('edit', filename=filename))
    return redirect(request.url)

@app.route('/edit/<filename>')
def edit(filename):
    return render_template('editor.html', filename=filename)

@app.route('/adjust/<filename>', methods=['POST'])
def adjust(filename):
    brightness = float(request.form.get('brightness', 1.0))
    contrast = float(request.form.get('contrast', 1.0))
    saturation = float(request.form.get('saturation', 1.0))
    sharpness = float(request.form.get('sharpness', 1.0))
    noise = int(request.form.get('noise', 0))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if adjust_image(image_path, brightness, contrast, saturation, sharpness, noise):
        return redirect(url_for('display', filename=filename))
    else:
        return "Gagal menyesuaikan gambar."

@app.route('/display/<filename>')
def display(filename):
    return render_template('display.html', filename=filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)