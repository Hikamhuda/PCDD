import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import mediapipe as mp
from datetime import datetime
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['SECRET_KEY'] = 'your_secret_key_here' # For production, use environment variables
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize MediaPipe Hands (ONCE globally for efficiency)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # Process continuously for video stream
    max_num_hands=2,                # Detect up to two hands
    min_detection_confidence=0.7,   # Higher confidence for initial detection
    min_tracking_confidence=0.5     # Lower confidence for tracking after detection
)
mp_drawing = mp.solutions.drawing_utils # Utility for drawing landmarks

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optimized function to detect hi-five gesture (all fingers open)
# Takes processed hand landmarks and handedness as input
def detect_hi_five_optimized(hand_landmarks_obj, handedness_label):
    if not hand_landmarks_obj:
        return False
        
    landmarks = hand_landmarks_obj.landmark # Access the list of landmark objects

    # MediaPipe HandLandmark enum for clarity
    lm = mp_hands.HandLandmark

    finger_open_status = []

    # Thumb: Check if it's extended.
    # Compare Y of THUMB_TIP and THUMB_IP (tip should be higher/smaller Y).
    # For side extension, compare X of THUMB_TIP and THUMB_MCP based on handedness (mirrored view).
    thumb_tip_y = landmarks[lm.THUMB_TIP].y
    thumb_ip_y = landmarks[lm.THUMB_IP].y
    thumb_tip_x = landmarks[lm.THUMB_TIP].x
    thumb_mcp_x = landmarks[lm.THUMB_MCP].x # Metacarpophalangeal joint (base of thumb)

    # Primary check: thumb pointing upwards (tip higher than IP joint)
    thumb_vertically_open = thumb_tip_y < thumb_ip_y
    
    # Secondary check: thumb horizontally extended from palm
    thumb_horizontally_open = False
    if handedness_label == "Right": # In mirrored view, right hand thumb tip is to the left of MCP when extended
        thumb_horizontally_open = thumb_tip_x < thumb_mcp_x 
    elif handedness_label == "Left": # In mirrored view, left hand thumb tip is to the right of MCP when extended
        thumb_horizontally_open = thumb_tip_x > thumb_mcp_x
    
    # Thumb is considered open if either vertically or significantly horizontally open.
    # A simple OR condition can work here. More complex logic might involve angles or distances.
    if thumb_vertically_open or thumb_horizontally_open:
         finger_open_status.append(True)
    else:
         finger_open_status.append(False)

    # Other fingers: Index, Middle, Ring, Pinky
    # Compare Y of TIP and PIP. Tip should be higher (smaller Y) than PIP.
    # Also ensure PIP is higher than MCP to confirm extension, not just curled up.
    other_fingers_tips_indices = [lm.INDEX_FINGER_TIP, lm.MIDDLE_FINGER_TIP, lm.RING_FINGER_TIP, lm.PINKY_TIP]
    other_fingers_pips_indices = [lm.INDEX_FINGER_PIP, lm.MIDDLE_FINGER_PIP, lm.RING_FINGER_PIP, lm.PINKY_PIP]
    other_fingers_mcps_indices = [lm.INDEX_FINGER_MCP, lm.MIDDLE_FINGER_MCP, lm.RING_FINGER_MCP, lm.PINKY_MCP]


    for i in range(len(other_fingers_tips_indices)):
        tip_y = landmarks[other_fingers_tips_indices[i]].y
        pip_y = landmarks[other_fingers_pips_indices[i]].y
        mcp_y = landmarks[other_fingers_mcps_indices[i]].y
        
        # Finger is open if tip is above PIP, and PIP is above MCP (ensures extension)
        if tip_y < pip_y and pip_y < mcp_y:
            finger_open_status.append(True)
        else:
            finger_open_status.append(False)
            
    return all(finger_open_status)


# Function to extract features from the image
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image at {image_path} for feature extraction.")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
        
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        features = {
            'mean': float(mean_val), # Ensure JSON serializable
            'std_dev': float(std_val),
            'edge_ratio': float(edge_ratio),
            'histogram': {
                'blue': hist_b.flatten().tolist(), # Flatten and convert to list
                'green': hist_g.flatten().tolist(),
                'red': hist_r.flatten().tolist()
            }
        }
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

# Function to adjust image quality
def adjust_image(image_path, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, noise_reduction=0):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB") # Ensure it's in RGB for consistent processing
        
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
            # GaussianBlur radius in Pillow is different from OpenCV's kernel size.
            # Keep it simple; users can experiment with values.
            img = img.filter(ImageFilter.GaussianBlur(radius=noise_reduction))
        
        base_name = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(base_name)
        # Ensure processed filename doesn't grow indefinitely with "processed_" prefix
        if name_part.startswith("processed_"):
            name_part = name_part[len("processed_"):]
            # Remove old timestamp if present
            if len(name_part) > 15 and name_part[8] == '_' and name_part[:8].isdigit() and name_part[9:15].isdigit():
                 name_part = name_part[16:]


        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name_part}{ext_part}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        img.save(processed_path)
        
        return processed_filename
    except Exception as e:
        print(f"Error adjusting image {image_path}: {e}")
        return None

# Global variable for hi-five status (used by /check_hi_five endpoint)
hi_five_detected_globally = False

@app.route('/')
def index():
    # Ensure template exists or this will error
    return render_template('index.html') 

@app.route('/capture_page')
def capture_page():
    # Ensure template exists
    return render_template('capture.html')

def generate_frames():
    global hi_five_detected_globally # Use the global 'hands' instance initialized outside
    
    cap = cv2.VideoCapture(0) # Or specific camera index
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        # Yield an error message or image if you want to show it on the client
        # For now, just stops generation.
        return 

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame. Skipping.")
            time.sleep(0.1) # Wait a bit before trying again
            continue

        frame = cv2.flip(frame, 1)  # Mirror effect for intuitive interaction
        
        # Convert the BGR image to RGB for MediaPipe.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands. This is the main detection step.
        results = hands.process(frame_rgb) # Use the global 'hands' instance
        
        current_frame_hi_five = False # Reset for current frame, for any hand
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the original BGR frame.
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # Landmark style
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connection style
                )
                
                handedness_label = "Unknown"
                if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                    handedness = results.multi_handedness[hand_idx]
                    handedness_label = handedness.classification[0].label # 'Left' or 'Right'
                    score = handedness.classification[0].score
                    
                    # Display handedness
                    cv2.putText(frame, f"{handedness_label} Hand ({score:.2f})", 
                                (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1] - 30), 
                                 int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0] - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

                # Perform hi-five detection using the optimized function
                if detect_hi_five_optimized(hand_landmarks, handedness_label):
                    current_frame_hi_five = True
                    # Display "HI-FIVE!" near the detected hand
                    cv2.putText(frame, "HI-FIVE!", 
                                (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), 
                                 int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0] - 60)),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Update the global hi-five status
        hi_five_detected_globally = current_frame_hi_five
        
        if hi_five_detected_globally: # Optional: General message if any hand is a hi-five
             cv2.putText(frame, "HI-FIVE DETECTED!", (20, 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Encode the frame in JPEG format.
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame to JPEG.")
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the multipart response.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    print("Releasing video capture device.")
    cap.release()
    # cv2.destroyAllWindows() # Usually not needed/problematic in web server threads

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_hi_five')
def check_hi_five():
    global hi_five_detected_globally
    return {'hi_five': hi_five_detected_globally}

@app.route('/capture', methods=['POST'])
def capture():
    global hi_five_detected_globally
    hi_five_detected_globally = False # Reset status on new capture
    
    # Consider if video_feed's camera should be paused/released here
    # For simplicity, we open a new instance. This might conflict if /video_feed is active.
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        flash('Could not access the camera.', 'error')
        return redirect(url_for('capture_page')) # Or 'index'

    ret, frame = cap.read()
    cap.release() # Release immediately after capture

    if ret:
        frame = cv2.flip(frame, 1) # Mirror effect consistent with video_feed
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"Image captured and saved to {filepath}")
            return redirect(url_for('edit', filename=filename))
        except Exception as e:
            print(f"Error saving captured image: {e}")
            flash('Error saving captured image.', 'error')
            return redirect(url_for('capture_page'))
    else:
        flash('Failed to capture image from camera.', 'error')
        return redirect(url_for('capture_page'))


@app.route('/realtime_adjust', methods=['POST'])
def realtime_adjust():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    
    try:
        brightness = float(request.form.get('brightness', 1.0))
        contrast = float(request.form.get('contrast', 1.0))
        saturation = float(request.form.get('saturation', 1.0))
        sharpness = float(request.form.get('sharpness', 1.0))
        noise_reduction = float(request.form.get('noise_reduction', 0))
        
        img = Image.open(file.stream) # Read directly from stream
        img = img.convert("RGB")

        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        if noise_reduction > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=noise_reduction))
        
        buffer = BytesIO()
        img.save(buffer, format="JPEG") # Or PNG if preferred
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {'image': img_base64}

    except Exception as e:
        print(f"Error in realtime_adjust: {e}")
        return {'error': f'Error processing image: {str(e)}'}, 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(request.url) # Redirect to the same page (likely index with upload form)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading.', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"File uploaded successfully to {filepath}")
            
            # Optional: Extract features from uploaded image immediately
            # features = extract_features(filepath)
            # if features:
            #     print("Extracted features from uploaded image:", features)
            
            return redirect(url_for('edit', filename=filename))
        except Exception as e:
            print(f"Error saving uploaded file {filename}: {e}")
            flash(f'Error saving file: {str(e)}', 'error')
            return redirect(request.url)
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, webp.', 'error')
        return redirect(request.url)

@app.route('/edit/<filename>')
def edit(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Image not found. It might have been moved or deleted.', 'error')
        return redirect(url_for('index'))
    
    # Ensure template exists
    return render_template('editor.html', filename=filename)

@app.route('/process/<filename>', methods=['POST'])
def process_image(filename):
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        flash('Original image not found for processing.', 'error')
        return redirect(url_for('edit', filename=filename))
        
    try:
        brightness = float(request.form.get('brightness', 1.0))
        contrast = float(request.form.get('contrast', 1.0))
        saturation = float(request.form.get('saturation', 1.0))
        sharpness = float(request.form.get('sharpness', 1.0))
        noise_reduction = float(request.form.get('noise_reduction', 0)) # Ensure this is float or int as needed
        
        processed_filename = adjust_image(
            original_path,
            brightness, contrast, saturation, sharpness, noise_reduction
        )
        
        if processed_filename:
            return redirect(url_for('result', filename=processed_filename))
        else:
            flash('Error processing image. Adjustments might have failed.', 'error')
            return redirect(url_for('edit', filename=filename))
            
    except ValueError:
        flash('Invalid input for adjustment parameters. Please enter numbers only.', 'error')
        return redirect(url_for('edit', filename=filename))
    except Exception as e:
        print(f"Error in process_image for {filename}: {e}")
        flash(f'An unexpected error occurred: {str(e)}', 'error')
        return redirect(url_for('edit', filename=filename))

@app.route('/result/<filename>')
def result(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path):
        flash('Processed image not found. It might have been deleted or an error occurred.', 'error')
        return redirect(url_for('index'))
    
    features = extract_features(processed_path) # Extract features from the processed image
    
    # Ensure template exists
    return render_template('result.html', 
                           filename=filename,
                           features=features if features else {}) # Pass empty dict if features are None

if __name__ == '__main__':
    # Create upload and processed directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # For development: app.run(debug=True)
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True, host='0.0.0.0', port=5000) # host='0.0.0.0' makes it accessible on network
