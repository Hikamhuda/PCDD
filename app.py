import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import mediapipe as mp
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['SECRET_KEY'] = 'your_secret_key_here'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect hi-five gesture (all fingers open)
def detect_hi_five(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Finger tip and pip (knuckle) landmarks
            finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
            finger_pips = [3, 6, 10, 14, 18]
            
            fingers_open = []
            
            # Check thumb (different logic for thumb)
            if landmarks[4].x < landmarks[3].x:  # Right hand thumb open
                fingers_open.append(True)
            elif landmarks[4].x > landmarks[3].x:  # Left hand thumb open
                fingers_open.append(True)
            else:
                fingers_open.append(False)
            
            # Check other fingers
            for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
                if landmarks[tip].y < landmarks[pip].y:  # Finger is open
                    fingers_open.append(True)
                else:
                    fingers_open.append(False)
            
            # If all fingers are open
            if all(fingers_open):
                return True
    return False

# Function to extract features from the image
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale for some features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        # Edge detection (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_pixels / total_pixels
        
        # Color histogram features
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        features = {
            'mean': mean_val,
            'std_dev': std_val,
            'edge_ratio': edge_ratio,
            'histogram': {
                'blue': hist_b.tolist(),
                'green': hist_g.tolist(),
                'red': hist_r.tolist()
            }
        }
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to adjust image quality
def adjust_image(image_path, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, noise_reduction=0):
    try:
        img = Image.open(image_path)
        
        # Apply adjustments
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)
        
        if noise_reduction > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=noise_reduction))
        
        # Save processed image with timestamp
        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(image_path)}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        img.save(processed_path)
        
        return processed_filename
    except Exception as e:
        print(f"Error adjusting image: {e}")
        return None

# Global variable for hi-five status
hi_five_detected = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_page')
def capture_page():
    return render_template('capture.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global hi_five_detected
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)  # Mirror effect
            frame_with_detection = frame.copy()
            
            # Detect hi-five
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5) as hands:
                
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_with_detection,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                        
                        if detect_hi_five(frame):
                            hi_five_detected = True
                            cv2.putText(frame_with_detection, "HI-FIVE DETECTED!", (50, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            hi_five_detected = False
            
            ret, buffer = cv2.imencode('.jpg', frame_with_detection)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/check_hi_five')
def check_hi_five():
    global hi_five_detected
    return {'hi_five': hi_five_detected}

@app.route('/capture', methods=['POST'])
def capture():
    global hi_five_detected
    hi_five_detected = False
    
    cap = cv2.VideoCapture(0)
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Get last frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.imwrite(filepath, frame)
    
    cap.release()
    return redirect(url_for('edit', filename=filename))

@app.route('/realtime_adjust', methods=['POST'])
def realtime_adjust():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    # Get adjustment parameters
    brightness = float(request.form.get('brightness', 1.0))
    contrast = float(request.form.get('contrast', 1.0))
    saturation = float(request.form.get('saturation', 1.0))
    sharpness = float(request.form.get('sharpness', 1.0))
    noise_reduction = float(request.form.get('noise_reduction', 0))
    
    # Read image from file upload
    img = Image.open(file.stream)
    
    # Process adjustments
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
    
    # Save to buffer
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Return as base64
    import base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {'image': img_base64}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features from uploaded image
        features = extract_features(filepath)
        if features:
            print("Extracted features:", features)
        
        return redirect(url_for('edit', filename=filename))
    
    flash('Invalid file type. Allowed types: png, jpg, jpeg, webp', 'error')
    return redirect(request.url)

@app.route('/edit/<filename>')
def edit(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Image not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('editor.html', filename=filename)

@app.route('/process/<filename>', methods=['POST'])
def process_image(filename):
    try:
        brightness = float(request.form.get('brightness', 1.0))
        contrast = float(request.form.get('contrast', 1.0))
        saturation = float(request.form.get('saturation', 1.0))
        sharpness = float(request.form.get('sharpness', 1.0))
        noise_reduction = float(request.form.get('noise_reduction', 0))
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_filename = adjust_image(
            original_path,
            brightness,
            contrast,
            saturation,
            sharpness,
            noise_reduction
        )
        
        if processed_filename:
            return redirect(url_for('result', filename=processed_filename))
        else:
            flash('Error processing image', 'error')
            return redirect(url_for('edit', filename=filename))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('edit', filename=filename))

@app.route('/result/<filename>')
def result(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path):
        flash('Processed image not found', 'error')
        return redirect(url_for('index'))
    
    # Extract features from processed image
    features = extract_features(processed_path)
    
    return render_template('result.html', 
                         filename=filename,
                         features=features)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)