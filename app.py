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
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['SECRET_KEY'] = 'ganti_dengan_kunci_rahasia_yang_sangat_kuat_dan_unik' # Ganti ini!
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize MediaPipe Hands (ONCE globally for efficiency)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # Process video frames
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_hi_five_optimized(hand_landmarks_obj, handedness_label):
    if not hand_landmarks_obj:
        return False
    landmarks = hand_landmarks_obj.landmark
    lm = mp_hands.HandLandmark
    
    finger_open_status = []

    # Thumb: Check if tip is above IP joint (vertically open) or horizontally away from MCP (horizontally open)
    thumb_tip_y = landmarks[lm.THUMB_TIP].y
    thumb_ip_y = landmarks[lm.THUMB_IP].y
    thumb_tip_x = landmarks[lm.THUMB_TIP].x
    thumb_mcp_x = landmarks[lm.THUMB_MCP].x

    thumb_vertically_open = thumb_tip_y < thumb_ip_y
    thumb_horizontally_open = False
    if handedness_label == "Right":
        thumb_horizontally_open = thumb_tip_x < thumb_mcp_x
    elif handedness_label == "Left":
        thumb_horizontally_open = thumb_tip_x > thumb_mcp_x
    
    if thumb_vertically_open or thumb_horizontally_open:
        finger_open_status.append(True)
    else:
        finger_open_status.append(False)

    other_fingers_tips_indices = [lm.INDEX_FINGER_TIP, lm.MIDDLE_FINGER_TIP, lm.RING_FINGER_TIP, lm.PINKY_TIP]
    other_fingers_pips_indices = [lm.INDEX_FINGER_PIP, lm.MIDDLE_FINGER_PIP, lm.RING_FINGER_PIP, lm.PINKY_PIP]
    other_fingers_mcps_indices = [lm.INDEX_FINGER_MCP, lm.MIDDLE_FINGER_MCP, lm.RING_FINGER_MCP, lm.PINKY_MCP]

    for i in range(len(other_fingers_tips_indices)):
        tip_y = landmarks[other_fingers_tips_indices[i]].y
        pip_y = landmarks[other_fingers_pips_indices[i]].y
        mcp_y = landmarks[other_fingers_mcps_indices[i]].y
        if tip_y < pip_y and pip_y < mcp_y: # Finger is extended
            finger_open_status.append(True)
        else:
            finger_open_status.append(False)
            
    return all(finger_open_status)

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: 
            print(f"Warning: Could not read image at {image_path} for feature extraction.")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(img); std_val = np.std(img)
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        return {'mean': float(mean_val), 'std_dev': float(std_val), 'edge_ratio': float(edge_ratio),
                'histogram': {'blue': hist_b.flatten().tolist(), 'green': hist_g.flatten().tolist(), 'red': hist_r.flatten().tolist()}}
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def _apply_hsl_adjustments_to_pil_image(img_pil, params):
    hsv_img = img_pil.convert('HSV')
    hsv_np = np.array(hsv_img, dtype=np.float32)
    h_channel, s_channel, v_channel = hsv_np[:,:,0], hsv_np[:,:,1], hsv_np[:,:,2]

    S_THRESH = params.get('s_thresh', 85) 
    V_THRESH = params.get('v_thresh', 65) 

    color_ranges = {
        "r": ([0, S_THRESH, V_THRESH], [10, 255, 255], [245, S_THRESH, V_THRESH], [255, 255, 255]),
        "o": ([11, S_THRESH, V_THRESH], [28, 255, 255]),
        "y": ([29, S_THRESH, V_THRESH], [48, 255, 255]),
        "g": ([49, S_THRESH, V_THRESH], [95, 255, 255]),
        "a": ([114, S_THRESH, V_THRESH], [140, 255, 255]),
        "b": ([153, S_THRESH, V_THRESH], [185, 255, 255]),
        "p": ([189, S_THRESH, V_THRESH], [220, 255, 255]),
        "m": ([221, S_THRESH, V_THRESH], [244, 255, 255])
    }

    for color_key_short in ["r", "o", "y", "g", "a", "b", "p", "m"]:
        hue_delta = params.get(f'hue_{color_key_short}_delta', 0)
        sat_factor = params.get(f'sat_{color_key_short}_factor', 1.0)
        light_factor = params.get(f'light_{color_key_short}_factor', 1.0)

        if not (hue_delta == 0 and sat_factor == 1.0 and light_factor == 1.0):
            ranges_for_color = color_ranges[color_key_short]
            current_mask = np.zeros_like(h_channel, dtype=bool)

            if color_key_short == "r": 
                lr1, ur1, lr2, ur2 = ranges_for_color
                mask1 = (h_channel >= lr1[0]) & (h_channel <= ur1[0]) & \
                        (s_channel >= lr1[1]) & (s_channel <= ur1[1]) & \
                        (v_channel >= lr1[2]) & (v_channel <= ur1[2])
                mask2 = (h_channel >= lr2[0]) & (h_channel <= ur2[0]) & \
                        (s_channel >= lr2[1]) & (s_channel <= ur2[1]) & \
                        (v_channel >= lr2[2]) & (v_channel <= ur2[2])
                current_mask = mask1 | mask2
            else:
                lr, ur = ranges_for_color
                current_mask = (h_channel >= lr[0]) & (h_channel <= ur[0]) & \
                               (s_channel >= lr[1]) & (s_channel <= ur[1]) & \
                               (v_channel >= lr[2]) & (v_channel <= ur[2])

            if np.any(current_mask):
                hue_adjustment_value_pil = (hue_delta / 360.0) * 255.0
                h_channel[current_mask] = (h_channel[current_mask] + hue_adjustment_value_pil) % 256
                s_channel[current_mask] = np.clip(s_channel[current_mask] * sat_factor, 0, 255)
                v_channel[current_mask] = np.clip(v_channel[current_mask] * light_factor, 0, 255)

    hsv_np[:,:,0] = h_channel
    hsv_np[:,:,1] = s_channel
    hsv_np[:,:,2] = v_channel
    
    adjusted_hsv_img = Image.fromarray(hsv_np.astype(np.uint8), 'HSV')
    return adjusted_hsv_img.convert("RGB")

def _apply_white_balance(img_pil, wb_temp_param):
    """Applies white balance adjustment to a PIL image."""
    if wb_temp_param == 0.0: # No change needed
        return img_pil

    img_np = np.array(img_pil, dtype=np.float32) # Use float for calculations
    r_channel, g_channel, b_channel = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    # Define how much the temperature slider affects the channels.
    # wb_temp_param is expected to be from -100 (cool) to 100 (warm).
    # adjustment_strength scales this effect. Tune this for sensitivity.
    adjustment_strength = 0.3 # Max change of 30 units if slider is at 100 or -100

    if wb_temp_param > 0: # Warmer: Increase Red, Decrease Blue
        r_channel += wb_temp_param * adjustment_strength
        b_channel -= wb_temp_param * adjustment_strength * 0.7 # Blue reduction factor
    elif wb_temp_param < 0: # Cooler: Decrease Red, Increase Blue
        # wb_temp_param is negative here
        r_channel += wb_temp_param * adjustment_strength * 0.7 # Red reduction factor (less aggressive)
        b_channel -= wb_temp_param * adjustment_strength # Effective addition due to double negative

    img_np[:,:,0] = np.clip(r_channel, 0, 255)
    # Green channel is typically not directly adjusted for simple warm/cool,
    # but more complex WB might affect it.
    img_np[:,:,1] = np.clip(g_channel, 0, 255) 
    img_np[:,:,2] = np.clip(b_channel, 0, 255)

    return Image.fromarray(img_np.astype(np.uint8), 'RGB')


def adjust_image(image_path, params):
    try:
        img_pil = Image.open(image_path).convert("RGB")
        
        # 1. General Adjustments
        if params.get('brightness', 1.0) != 1.0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(params['brightness'])
        if params.get('contrast', 1.0) != 1.0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])
        if params.get('saturation', 1.0) != 1.0: # Global saturation
            img_pil = ImageEnhance.Color(img_pil).enhance(params['saturation'])
        if params.get('sharpness', 1.0) != 1.0:
            img_pil = ImageEnhance.Sharpness(img_pil).enhance(params['sharpness'])
        if params.get('noise_reduction', 0) > 0:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=params['noise_reduction']))

        # 2. White Balance Adjustment
        white_balance_temp = params.get('white_balance_temp', 0.0)
        if white_balance_temp != 0.0:
            img_pil = _apply_white_balance(img_pil, white_balance_temp)

        # 3. HSL Adjustments (per color)
        img_pil = _apply_hsl_adjustments_to_pil_image(img_pil, params)
        
        base_name = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(base_name)
        if name_part.startswith("processed_"):
            name_part = name_part[len("processed_"):]
            if len(name_part) > 15 and name_part[8] == '_' and name_part[:8].isdigit() and name_part[9:15].isdigit():
                name_part = name_part[16:]

        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name_part}{ext_part}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        img_pil.save(processed_path)
        
        return processed_filename
    except Exception as e:
        print(f"Error adjusting image {image_path}: {e}")
        traceback.print_exc()
        return None

hi_five_detected_globally = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_page')
def capture_page():
    return render_template('capture.html')

def generate_frames():
    global hi_five_detected_globally; cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open video capture device."); return
    while True:
        success, frame = cap.read()
        if not success: print("Error: Failed to capture frame."); time.sleep(0.1); continue
        frame = cv2.flip(frame, 1); frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb); current_frame_hi_five = False
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=2))
                handedness_label = "Unknown"
                if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                    handedness = results.multi_handedness[hand_idx]; handedness_label = handedness.classification[0].label; score = handedness.classification[0].score
                    cv2.putText(frame, f"{handedness_label} ({score:.2f})", (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*frame.shape[1]-30), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y*frame.shape[0]-30)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2,cv2.LINE_AA)
                if detect_hi_five_optimized(hand_landmarks, handedness_label):
                    current_frame_hi_five = True
                    cv2.putText(frame, "HI-FIVE!", (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y*frame.shape[0]-60)), cv2.FONT_HERSHEY_TRIPLEX,1.2,(0,255,0),3,cv2.LINE_AA)
        hi_five_detected_globally = current_frame_hi_five
        if hi_five_detected_globally: cv2.putText(frame, "HI-FIVE DETECTED!", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: print("Error: Failed to encode frame."); continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_hi_five')
def check_hi_five():
    return {'hi_five': hi_five_detected_globally}

@app.route('/capture', methods=['POST'])
def capture():
    global hi_five_detected_globally; hi_five_detected_globally = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): flash('Tidak dapat mengakses kamera.', 'error'); return redirect(url_for('capture_page'))
    ret, frame = cap.read(); cap.release()
    if ret:
        frame = cv2.flip(frame, 1)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: cv2.imwrite(filepath, frame); return redirect(url_for('edit', filename=filename))
        except Exception as e: print(f"Error saving captured: {e}"); flash('Gagal menyimpan gambar.', 'error'); return redirect(url_for('capture_page'))
    else: flash('Gagal mengambil gambar dari kamera.', 'error'); return redirect(url_for('capture_page'))

def _parse_adjustment_params_from_form(form_data):
    params = {
        'brightness': float(form_data.get('brightness', 1.0)),
        'contrast': float(form_data.get('contrast', 1.0)),
        'saturation': float(form_data.get('saturation', 1.0)), 
        'sharpness': float(form_data.get('sharpness', 1.0)),
        'noise_reduction': float(form_data.get('noise_reduction', 0)),
        'white_balance_temp': float(form_data.get('white_balance_temp', 0.0)), # New white balance param
    }
    for color_key_short in ["r", "o", "y", "g", "a", "b", "p", "m"]:
        params[f'hue_{color_key_short}_delta'] = int(form_data.get(f'hue_{color_key_short}', 0))
        params[f'sat_{color_key_short}_factor'] = float(form_data.get(f'sat_{color_key_short}', 1.0))
        params[f'light_{color_key_short}_factor'] = float(form_data.get(f'light_{color_key_short}', 1.0))
    return params

@app.route('/realtime_adjust', methods=['POST'])
def realtime_adjust():
    if 'file' not in request.files:
        return {'error': 'File tidak disediakan'}, 400
    file = request.files['file']
    if file.filename == '':
        return {'error': 'File tidak dipilih'}, 400
    
    try:
        params = _parse_adjustment_params_from_form(request.form)
        img_pil = Image.open(file.stream).convert("RGB")

        # 1. General Adjustments
        if params.get('brightness', 1.0) != 1.0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(params['brightness'])
        if params.get('contrast', 1.0) != 1.0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(params['contrast'])
        if params.get('saturation', 1.0) != 1.0: 
            img_pil = ImageEnhance.Color(img_pil).enhance(params['saturation'])
        if params.get('sharpness', 1.0) != 1.0:
            img_pil = ImageEnhance.Sharpness(img_pil).enhance(params['sharpness'])
        if params.get('noise_reduction', 0) > 0:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=params['noise_reduction']))
        
        # 2. White Balance Adjustment
        white_balance_temp = params.get('white_balance_temp', 0.0)
        if white_balance_temp != 0.0:
             img_pil = _apply_white_balance(img_pil, white_balance_temp)
        
        # 3. HSL Adjustments per color
        img_pil = _apply_hsl_adjustments_to_pil_image(img_pil, params)
        
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG") 
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {'image': img_base64}

    except Exception as e:
        print(f"Error in realtime_adjust: {e}")
        traceback.print_exc()
        return {'error': f'Gagal memproses gambar: {str(e)}'}, 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: flash('Tidak ada bagian file.', 'error'); return redirect(request.url)
    file = request.files['file']
    if file.filename == '': flash('Tidak ada file dipilih.', 'error'); return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try: file.save(filepath); return redirect(url_for('edit', filename=filename))
        except Exception as e: print(f"Error saving file: {e}"); flash(f'Gagal menyimpan file: {str(e)}', 'error'); return redirect(request.url)
    else: flash('Tipe file tidak valid.', 'error'); return redirect(request.url)

@app.route('/edit/<filename>')
def edit(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(processed_filepath):
            filepath = processed_filepath 
        else:
            flash('Gambar tidak ditemukan di folder upload maupun processed.', 'error')
            return redirect(url_for('index'))
            
    return render_template('editor.html', filename=filename, original_folder=app.config['UPLOAD_FOLDER'])


@app.route('/process/<filename>', methods=['POST'])
def process_image(filename):
    original_path_in_uploads = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    original_path_in_processed = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    
    original_path = None
    if os.path.exists(original_path_in_uploads):
        original_path = original_path_in_uploads
    elif os.path.exists(original_path_in_processed):
        original_path = original_path_in_processed
        
    if not original_path:
        flash('Gambar asli tidak ditemukan untuk diproses.', 'error')
        return redirect(url_for('edit', filename=filename)) 
        
    try:
        params = _parse_adjustment_params_from_form(request.form)
        processed_filename = adjust_image(original_path, params)
        
        if processed_filename:
            return redirect(url_for('result', filename=processed_filename))
        else:
            flash('Gagal memproses gambar. Penyesuaian mungkin gagal atau tidak menghasilkan file.', 'error')
            return redirect(url_for('edit', filename=filename))
            
    except ValueError: 
        flash('Input tidak valid untuk parameter penyesuaian. Harap masukkan angka yang benar.', 'error')
        return redirect(url_for('edit', filename=filename))
    except Exception as e:
        print(f"Error in process_image for {filename}: {e}")
        traceback.print_exc()
        flash(f'Terjadi kesalahan tak terduga saat memproses gambar: {str(e)}', 'error')
        return redirect(url_for('edit', filename=filename))

@app.route('/result/<filename>')
def result(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path): flash('Gambar yang diproses tidak ditemukan.', 'error'); return redirect(url_for('index'))
    features = extract_features(processed_path)
    return render_template('result.html', filename=filename, features=features if features else {})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
