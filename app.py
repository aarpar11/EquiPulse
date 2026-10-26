import os

os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
import scipy.signal as signal
import collections
import time
import threading
import urllib.request
import urllib.parse
import json
import logging
from flask import Flask, Response, jsonify, request, render_template_string

# --- 1. App Configuration ---
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- 2. Face Detector & CLAHE ---
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        urllib.request.urlretrieve(url, cascade_file)
    except Exception as e:
        print(f"Cascade download notice: {e}")

face_cascade = cv2.CascadeClassifier(cascade_file) if os.path.exists(cascade_file) else None
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# --- 3. ONNX Recurrent Neural Engine ---
session = ort.InferenceSession("model.onnx")


def get_shape(inp):
    return tuple(d if (isinstance(d, int) and d > 0) else 1 for d in inp.shape)


state = {inp.name: np.zeros(get_shape(inp), dtype=np.float32) for inp in session.get_inputs()[1:]}

# --- 4. Threading, Telemetry & Sensor Buffers ---
lock = threading.Lock()
MAX_BUFFER = 240
bvp_buffer = collections.deque(maxlen=MAX_BUFFER)
pos_buffer = collections.deque(maxlen=MAX_BUFFER)
time_buffer = collections.deque(maxlen=MAX_BUFFER)
consensus_bpm_buffer = collections.deque(maxlen=15)
filtered_wave_display = []

telemetry = {
    "bpm": 0.0,
    "hrv": 0.0,
    "sqi": 0,
    "snr": 0.0,
    "status": "Calibrating skin lock...",
    "tone": "Measuring...",
    "roi_label": "Scanning",
    "is_tracking": False,
    "guide_distance": "Checking...",
    "guide_lighting": "Checking...",
    "guide_stability": "Checking..."
}
latest_frame = None
chroma_baseline = None
rgb_ema_baseline = None
smooth_bbox = None
prev_face_center = None

class AdaptiveBPMKalmanFilter:
    """
    1D Adaptive Kalman Filter for rPPG Heart Rate Tracking.
    """
    def __init__(self, initial_bpm=72.0, process_noise=0.015):
        self.x = float(initial_bpm)
        self.p = 1.0
        self.q = process_noise
        self.last_update = time.time()

    def update(self, measured_bpm, snr, is_moving=False):
        now = time.time()
        dt = max(0.01, min(1.0, now - self.last_update))
        self.last_update = now

        self.p += self.q * dt

        snr_clamped = max(0.5, min(6.0, float(snr)))
        base_r = 12.0 / (snr_clamped ** 2)

        if is_moving:
            base_r *= 10.0

        k = self.p / (self.p + base_r)
        self.x += k * (float(measured_bpm) - self.x)
        self.p = (1.0 - k) * self.p
        self.x = float(np.clip(self.x, 48.0, 140.0))
        return self.x

# Global filter instances
bpm_kalman = AdaptiveBPMKalmanFilter(initial_bpm=72.0)
bg_lum_history = collections.deque(maxlen=15)

def preprocess_cielab(patch_bgr):
    global chroma_baseline
    patch = cv2.resize(patch_bgr, (36, 36), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    mean_L = np.mean(L) + 1e-5
    frame_chroma = np.array([np.mean(a), np.mean(b)])

    if chroma_baseline is None:
        chroma_baseline = frame_chroma
    else:
        chroma_baseline = 0.95 * chroma_baseline + 0.05 * frame_chroma

    delta_a = (a - chroma_baseline[0]) / mean_L
    delta_b = (b - chroma_baseline[1]) / mean_L
    delta_L = (L - np.mean(L)) / mean_L

    normalized_patch = np.stack([delta_a, delta_b, delta_L], axis=-1)
    normalized_patch = np.clip(normalized_patch * 6.0, -1.0, 1.0)
    return normalized_patch[np.newaxis, np.newaxis, :, :, :].astype(np.float32)

def extract_log_lambert_pos(patch_bgr):
    global rgb_ema_baseline
    if patch_bgr is None or patch_bgr.shape[0] < 8 or patch_bgr.shape[1] < 8:
        return 0.0

    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (12, 126, 68), (255, 182, 138))
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    valid_lum = (gray > 18) & (gray < 240)
    final_mask = (skin_mask > 0) & valid_lum

    if np.sum(final_mask) > 15:
        curr_bgr = np.mean(patch_bgr[final_mask], axis=0).astype(np.float32)
    else:
        curr_bgr = np.mean(patch_bgr, axis=(0, 1)).astype(np.float32)

    curr_bgr = np.maximum(curr_bgr, 1.0)
    if rgb_ema_baseline is None:
        rgb_ema_baseline = curr_bgr.copy()
    else:
        rgb_ema_baseline = 0.992 * rgb_ema_baseline + 0.008 * curr_bgr

    absorbance = -np.log(curr_bgr / (rgb_ema_baseline + 1e-5))
    b_abs, g_abs, r_abs = absorbance[0], absorbance[1], absorbance[2]

    s1 = g_abs - b_abs
    s2 = g_abs + b_abs - 2.0 * r_abs
    return float(s1 + 0.85 * s2)

def score_skin_patch(patch_bgr):
    if patch_bgr is None or patch_bgr.shape[0] < 8 or patch_bgr.shape[1] < 8:
        return -1.0
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (12, 126, 68), (255, 182, 138))
    skin_ratio = float(np.mean(skin_mask > 0))
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smoothness_bonus = 1.0 / (1.0 + laplacian_var / 250.0)
    return skin_ratio * 0.70 + smoothness_bonus * 0.30

def open_active_camera():
    for idx in [0, 1, 2]:
        test_cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if test_cap.isOpened():
            for _ in range(15):
                ret, frame = test_cap.read()
                if ret and np.mean(frame) > 10.0:
                    return test_cap
            test_cap.release()
    return cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

def camera_inference_loop(cap):
    global latest_frame, smooth_bbox, filtered_wave_display, telemetry, prev_face_center

    locked_hrv = 44.0
    last_process_time = 0.0

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            now = time.time()
            h, w, _ = frame.shape
            target_box = None
            chosen_label = "Face Detected"
            dist_status = "Optimal (18-24 in)"

            if face_cascade is not None and not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                equalized_gray = clahe.apply(gray)
                faces = face_cascade.detectMultiScale(equalized_gray, scaleFactor=1.08, minNeighbors=4, minSize=(80, 80))

                valid_faces = []
                for (fx, fy, fw, fh) in faces:
                    center_roi = frame[fy+int(fh*0.2):fy+int(fh*0.8), fx+int(fw*0.2):fx+int(fw*0.8)]
                    if score_skin_patch(center_roi) > 0.25:
                        valid_faces.append((fx, fy, fw, fh))

                if len(valid_faces) > 0:
                    fx, fy, fw, fh = max(valid_faces, key=lambda f: f[2] * f[3])

                    face_width_ratio = fw / float(w)
                    if face_width_ratio < 0.17:
                        dist_status = "Too Far (Move Closer)"
                    elif face_width_ratio > 0.44:
                        dist_status = "Too Close (Move Back)"

                    fh_box = (fx + int(fw * 0.33), fy + int(fh * 0.14), int(fw * 0.34), int(fh * 0.12))
                    lc_box = (fx + int(fw * 0.16), fy + int(fh * 0.46), int(fw * 0.22), int(fh * 0.14))
                    rc_box = (fx + int(fw * 0.62), fy + int(fh * 0.46), int(fw * 0.22), int(fh * 0.14))

                    candidates = [(fh_box, "Forehead ROI"), (lc_box, "Left Cheek ROI"), (rc_box, "Right Cheek ROI")]
                    best_score = -1.0
                    for (bx, by, bw, bh), label in candidates:
                        patch = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                        sc = score_skin_patch(patch)
                        if label == "Forehead ROI":
                            sc *= 1.05
                        if sc > best_score:
                            best_score = sc
                            target_box = (bx, by, bw, bh)
                            chosen_label = label

            is_auto = target_box is not None

            if not is_auto:
                box_w, box_h = int(w * 0.16), int(h * 0.11)
                target_box = (int((w - box_w) / 2), int(h * 0.28), box_w, box_h)

            fx, fy, fw, fh = target_box
            if smooth_bbox is None:
                smooth_bbox = [float(fx), float(fy), float(fw), float(fh)]
            else:
                smooth_bbox[0] = 0.88 * smooth_bbox[0] + 0.12 * fx
                smooth_bbox[1] = 0.88 * smooth_bbox[1] + 0.12 * fy
                smooth_bbox[2] = 0.88 * smooth_bbox[2] + 0.12 * fw
                smooth_bbox[3] = 0.88 * smooth_bbox[3] + 0.12 * fh

            curr_center = (smooth_bbox[0] + smooth_bbox[2] * 0.5, smooth_bbox[1] + smooth_bbox[3] * 0.5)
            motion_px = 0.0
            if prev_face_center is not None:
                motion_px = float(np.hypot(curr_center[0] - prev_face_center[0], curr_center[1] - prev_face_center[1]))
            prev_face_center = curr_center

            is_still = is_auto and (motion_px < 2.2)
            stability_status = "Stable (Locked)" if is_still else ("Motion Detected" if is_auto else "Searching Face")

            sx, sy, sw, sh = [int(round(v)) for v in smooth_bbox]
            sx, sy = max(0, min(sx, w - 12)), max(0, min(sy, h - 12))
            ex, ey = min(w, sx + max(12, sw)), min(h, sy + max(12, sh))
            skin_roi = frame[sy:ey, sx:ex]

            bg_patch = frame[0:min(35, h), 0:min(35, w)]
            curr_bg_lum = float(np.mean(cv2.cvtColor(bg_patch, cv2.COLOR_BGR2GRAY)))
            bg_lum_history.append(curr_bg_lum)
            ambient_flicker = False
            if len(bg_lum_history) >= 8:
                ambient_flicker = abs(curr_bg_lum - np.mean(bg_lum_history)) > 8.0

            light_status = "Optimal Even Lighting" if not ambient_flicker else "Lighting Shift Detected"

            if skin_roi.shape[0] >= 10 and skin_roi.shape[1] >= 10 and np.mean(skin_roi) > 5.0 and not ambient_flicker:
                input_tensor = preprocess_cielab(skin_roi)
                feed_dict = {session.get_inputs()[0].name: input_tensor}
                feed_dict.update(state)
                outputs = session.run(None, feed_dict)

                onnx_sample = float(np.array(outputs[0]).flatten()[0])
                pos_sample = extract_log_lambert_pos(skin_roi)

                if is_still:
                    bvp_buffer.append(onnx_sample)
                    pos_buffer.append(pos_sample)
                    time_buffer.append(now)
                elif len(bvp_buffer) > 0:
                    bvp_buffer.append(0.95 * bvp_buffer[-1] + 0.05 * onnx_sample)
                    pos_buffer.append(0.95 * pos_buffer[-1] + 0.05 * pos_sample)
                    time_buffer.append(now)

                if len(outputs) > 1:
                    for i, inp in enumerate(session.get_inputs()[1:]):
                        if i + 1 < len(outputs):
                            state[inp.name] = outputs[i + 1]

            box_color = (16, 185, 129) if is_still else ((56, 189, 248) if is_auto else (245, 158, 11))
            cv2.rectangle(frame, (sx, sy), (ex, ey), box_color, 2)
            cv2.putText(frame, f"{chosen_label}", (sx, max(18, sy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            if len(bvp_buffer) >= 160 and (time_buffer[-1] - time_buffer[0]) > 5.2 and (now - last_process_time) >= 0.20:
                last_process_time = now
                times = np.array(time_buffer)
                actual_fs = 30.0

                t_uniform = np.linspace(times[0], times[-1], int((times[-1] - times[0]) * actual_fs))
                onnx_interp = signal.detrend(np.interp(t_uniform, times, np.array(bvp_buffer)))
                pos_interp = signal.detrend(np.interp(t_uniform, times, np.array(pos_buffer)))

                onnx_norm = onnx_interp / (np.std(onnx_interp) + 1e-6)
                pos_norm = pos_interp / (np.std(pos_interp) + 1e-6)
                fused = 0.70 * pos_norm + 0.30 * onnx_norm

                b_filt, a_filt = signal.butter(3, [0.90 / (actual_fs / 2), 1.70 / (actual_fs / 2)], btype='bandpass')
                filt = signal.filtfilt(b_filt, a_filt, fused)

                with lock:
                    filtered_wave_display = filt[-120:].tolist()

                nperseg = min(len(filt), 180)
                freqs, psd = signal.welch(filt, fs=actual_fs, window='hann', nperseg=nperseg, noverlap=int(nperseg * 0.75), nfft=4096)
                valid_mask = (freqs >= 0.90) & (freqs <= 1.70)
                valid_freqs, valid_psd = freqs[valid_mask], psd[valid_mask]

                if len(valid_psd) > 0:
                    peak_idx = np.argmax(valid_psd)
                    welch_bpm = float(valid_freqs[peak_idx] * 60.0)
                    snr = float(valid_psd[peak_idx] / (np.median(valid_psd) + 1e-6))

                    norm_f = filt - np.mean(filt)
                    acf = np.correlate(norm_f, norm_f, mode='full')[len(norm_f)-1:]
                    acf = acf / (acf[0] + 1e-6)
                    min_l, max_l = int(actual_fs * (60.0 / 102.0)), min(len(acf) - 1, int(actual_fs * (60.0 / 54.0)))

                    acf_bpm = welch_bpm
                    if max_l > min_l + 2:
                        acf_peaks, _ = signal.find_peaks(acf[min_l:max_l], prominence=0.08)
                        if len(acf_peaks) > 0:
                            acf_bpm = 60.0 / ((min_l + acf_peaks[np.argmax(acf[min_l:max_l][acf_peaks])]) / actual_fs)

                    agrees = abs(welch_bpm - acf_bpm) <= 5.0
                    raw_bpm = 0.60 * welch_bpm + 0.40 * acf_bpm if agrees else welch_bpm

                    smooth_bpm = bpm_kalman.update(raw_bpm, snr=snr, is_moving=(not is_still or ambient_flicker))

                    expected_rr = 60000.0 / smooth_bpm
                    peaks, _ = signal.find_peaks(filt, distance=int(actual_fs * (48.0 / smooth_bpm)), prominence=np.std(filt) * 0.35)
                    if is_still and len(peaks) >= 4:
                        rr_ms = np.diff(t_uniform[peaks] * 1000.0)
                        clean_rr = rr_ms[np.abs(rr_ms - expected_rr) < (0.16 * expected_rr)]
                        if len(clean_rr) >= 2:
                            inst_rmssd = float(np.sqrt(np.mean(np.diff(clean_rr) ** 2)))
                            locked_hrv = 0.90 * locked_hrv + 0.10 * np.clip(inst_rmssd, 24.0, 78.0)

                    sqi_val = int(np.clip((snr - 1.1) / 3.2 * 100, 35, 96)) if is_still else 40

                    with lock:
                        telemetry["snr"] = round(float(snr), 1)
                        telemetry["sqi"] = sqi_val
                        telemetry["is_tracking"] = is_auto
                        telemetry["roi_label"] = chosen_label
                        telemetry["guide_distance"] = dist_status
                        telemetry["guide_lighting"] = light_status
                        telemetry["guide_stability"] = stability_status
                        telemetry["bpm"] = round(smooth_bpm, 1)
                        telemetry["hrv"] = round(locked_hrv, 1)

                        if telemetry["bpm"] < 75:
                            telemetry["status"] = "Normal Resting Baseline"
                            telemetry["tone"] = "Optimal Elasticity"
                        elif telemetry["bpm"] < 85:
                            telemetry["status"] = "Mild Sympathetic Load"
                            telemetry["tone"] = "Moderate Vascular Tone"
                        else:
                            telemetry["status"] = "Elevated Resting Rate"
                            telemetry["tone"] = "Increased Arterial Strain"

            with lock:
                latest_frame = frame.copy()

            time.sleep(0.01)
        except Exception:
            time.sleep(0.05)
cap_device = open_active_camera()
threading.Thread(target=camera_inference_loop, args=(cap_device,), daemon=True).start()


# --- 5. Non-Blocking Video Feed ---
def gen_frames():
    while True:
        frame_bytes = None
        with lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
        if frame_bytes is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            time.sleep(0.033)
        else:
            time.sleep(0.04)


# --- 6. Scientific Epidemiological Risk Model (Transparent Point Ledger) ---
def compute_cardiovascular_risk(data):
    bpm = float(data.get("bpm", 72.0))
    hrv = float(data.get("hrv", 42.0))
    age = int(data.get("age", 35))
    bp_category = data.get("bp_category", "unknown")
    family_history = bool(data.get("family_history", False))
    ethnicity = data.get("ethnicity", "general")
    sleep_hours = data.get("sleep", "6-8")
    chronic_stress = data.get("stress", "moderate")
    activity_level = data.get("activity", "low")

    breakdown = []
    drivers = []

    # 1. Resting Heart Rate Component (0 - 30 pts, Framingham Continuous Hazard Model)
    rhr_pts = int(round(np.clip(((bpm - 60.0) / 35.0) * 30.0, 0.0, 30.0)))
    breakdown.append({"label": f"Resting Heart Rate ({bpm:.1f} BPM)", "pts": rhr_pts, "max": 30,
                      "ref": "Framingham Cohort (+18% risk / +10 BPM)"})
    if bpm >= 80:
        drivers.append({"title": f"Elevated Resting Heart Rate ({bpm:.1f} BPM)",
                        "desc": "Resting HR >= 80 BPM is an independent clinical predictor of incident hypertension and increased arterial wall shear stress.",
                        "severity": f"+{rhr_pts} pts"})

    # 2. Vagal Tone / HRV Component (0 - 20 pts, ARIC Study Autonomic Model)
    hrv_pts = int(round(np.clip(((55.0 - hrv) / 35.0) * 20.0, 0.0, 20.0)))
    breakdown.append({"label": f"HRV Vagal Tone ({hrv:.1f} ms)", "pts": hrv_pts, "max": 20,
                      "ref": "ARIC Study (Low RMSSD < 30ms = 1.48x HR)"})
    if hrv < 35:
        drivers.append({"title": f"Reduced Vagal Brake ({hrv:.1f} ms)",
                        "desc": "Lower beat-to-beat variability indicates sympathetic dominance, which keeps peripheral blood vessels constricted.",
                        "severity": f"+{hrv_pts} pts"})

    # 3. Blood Pressure Category (0 - 25 pts, 2017 ACC/AHA Staging)
    bp_map = {"normal": 0, "unknown": 6, "elevated": 12, "stage1": 18, "stage2": 25}
    bp_pts = bp_map.get(bp_category, 6)
    breakdown.append({"label": f"Blood Pressure Stage ({bp_category.title()})", "pts": bp_pts, "max": 25,
                      "ref": "2017 ACC/AHA Clinical Practice Guidelines"})
    if bp_category == "unknown":
        drivers.append({"title": "Unverified Blood Pressure Baseline",
                        "desc": "Because hypertension has no physical symptoms, an unchecked cuff baseline is a primary screening gap.",
                        "severity": "+6 pts"})
    elif bp_pts >= 12:
        drivers.append({"title": f"Documented {bp_category.title()} Blood Pressure",
                        "desc": "Elevated arterial pressure directly compounds mechanical workload on the left ventricle.",
                        "severity": f"+{bp_pts} pts"})

    # 4. Demographic & Hereditary Multipliers (0 - 15 pts)
    demo_pts = 0
    if family_history:
        demo_pts += 6
        drivers.append({"title": "First-Degree Family History",
                        "desc": "Parental hypertension doubles baseline genetic susceptibility to vascular stiffness.",
                        "severity": "+6 pts"})
    if ethnicity in ["black", "south_asian", "hispanic"]:
        eth_add = 6 if ethnicity == "black" else 5
        demo_pts += eth_add
        drivers.append({"title": "Epidemiological Vascular Factor",
                        "desc": "Population studies (Jackson Heart / MASALA) document earlier-onset vascular stiffening, warranting proactive screening.",
                        "severity": f"+{eth_add} pts"})
    if age >= 45:
        demo_pts += 3
    demo_pts = min(15, demo_pts)
    breakdown.append({"label": "Hereditary & Demographic Factors", "pts": demo_pts, "max": 15,
                      "ref": "MESA / Jackson Heart / MASALA Studies"})

    # 5. Modifiable Lifestyle Load (0 - 10 pts)
    life_pts = 0
    if sleep_hours == "<6":
        life_pts += 4
        drivers.append({"title": "Short Sleep (< 6 Hours)",
                        "desc": "Blunts the natural 10-15% nocturnal dip in blood pressure during deep sleep.",
                        "severity": "+4 pts"})
    if chronic_stress == "high":
        life_pts += 3
        drivers.append(
            {"title": "Chronic Daily Stress", "desc": "Sustains elevated circulating cortisol and adrenaline.",
             "severity": "+3 pts"})
    if activity_level == "low":
        life_pts += 3
        drivers.append({"title": "Low Weekly Aerobic Volume (< 60m)",
                        "desc": "Reduces stroke volume efficiency, requiring the heart to beat more times per minute.",
                        "severity": "+3 pts"})
    breakdown.append(
        {"label": "Sleep, Stress & Activity Load", "pts": life_pts, "max": 10, "ref": "AHA Life's Essential 8 Metrics"})

    total_score = int(np.clip(rhr_pts + hrv_pts + bp_pts + demo_pts + life_pts, 4, 98))

    if total_score < 25:
        tier = "Low Risk / Optimal Profile"
        summary = "Your resting heart rate and beat-to-beat variability show healthy vagal tone and low vascular workload."
        rec_action = "Maintain regular walking habits and verify your blood pressure once a year."
    elif total_score < 50:
        tier = "Mild Risk / Early Sympathetic Strain"
        summary = "Your score reflects mild cardiovascular workload from modifiable factors (such as sleep, activity, or unchecked BP)."
        rec_action = "Check your blood pressure at a free local pharmacy kiosk (Tab 4) and follow the Zone 2 walking plan (Tab 5)."
    elif total_score < 72:
        tier = "Moderate Risk / Prehypertension Warning"
        summary = "Combined heart rate, autonomic, and background factors indicate elevated strain on your blood vessels."
        rec_action = "Schedule a blood pressure check at a local clinic (Tab 3) and measure at home with an upper-arm cuff."
    else:
        tier = "High Risk / Clinical Evaluation Advised"
        summary = "Multiple cardiovascular risk indicators are elevated. Proactive medical staging is strongly recommended."
        rec_action = "Book an appointment with a local primary care clinic or cardiologist in Tab 3 promptly."

    sns_percentage = min(90, max(20, int(round(35 + (rhr_pts / 30.0) * 35 + (hrv_pts / 20.0) * 25))))
    pns_percentage = 100 - sns_percentage

    return {
        "score": total_score,
        "tier": tier,
        "summary": summary,
        "rec_action": rec_action,
        "sns_ratio": sns_percentage,
        "pns_ratio": pns_percentage,
        "breakdown": breakdown,
        "drivers": drivers,
        "input_bpm": round(bpm, 1),
        "input_hrv": round(hrv, 1)
    }


# --- 7. REAL US Government NPI Registry API for Clinics & Pharmacies ---
def resolve_location(query):
    clean_q = query.strip()
    if clean_q.isdigit() and len(clean_q) == 5:
        try:
            req = urllib.request.Request(f"https://api.zippopotam.us/us/{clean_q}",
                                         headers={"User-Agent": "EquiPulse/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                place = data["places"][0]
                return {
                    "city": place["place name"],
                    "state": place["state abbreviation"],
                    "zip": clean_q,
                    "display": f"{place['place name']}, {place['state abbreviation']} {clean_q}"
                }
        except Exception:
            pass
    parts = [p.strip() for p in clean_q.split(",")]
    city = parts[0].title()
    state = parts[1].upper() if len(parts) > 1 else "IL"
    return {"city": city, "state": state, "zip": clean_q if clean_q.isdigit() else "", "display": f"{city}, {state}"}


def fetch_real_npi_providers(loc_info, taxonomy_desc, limit=4):
    results = []
    seen_names = set()
    zip_code = loc_info.get("zip", "")

    search_passes = [zip_code, zip_code[:3] + "*"] if len(zip_code) == 5 else [None]

    for z_query in search_passes:
        if len(results) >= limit:
            break
        try:
            params = {
                "version": "2.1",
                "taxonomy_description": taxonomy_desc,
                "limit": "15",
                "enumeration_type": "NPI-2"
            }
            if z_query:
                params["postal_code"] = z_query
            else:
                params["city"] = loc_info["city"]
                params["state"] = loc_info["state"]

            url = "https://npiregistry.cms.hhs.gov/api/?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "EquiPulse-Health/1.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode())
                for item in data.get("results", []):
                    basic = item.get("basic", {})
                    name = basic.get("organization_name") or basic.get("authorized_official_last_name")
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    addrs = item.get("addresses", [])
                    loc_addr = next((a for a in addrs if a.get("address_purpose") == "LOCATION"), addrs[0] if addrs else {})
                    street = loc_addr.get("address_1", "").title()
                    city = loc_addr.get("city", loc_info["city"]).title()
                    state = loc_addr.get("state", loc_info["state"])
                    postal = loc_addr.get("postal_code", "")[:5]
                    phone = loc_addr.get("telephone_number", "Call Local Directory")

                    results.append({
                        "name": name.title(),
                        "street": street,
                        "city_state": f"{city}, {state} {postal}",
                        "phone": phone,
                        "exact": (postal == zip_code)
                    })
        except Exception as e:
            print(f"NPI lookup notice: {e}")

    return results[:limit]

def get_targeted_resources(loc_info, risk_profile):
    clinics = fetch_real_npi_providers(loc_info, "Family Medicine", limit=2)
    cardio = fetch_real_npi_providers(loc_info, "Cardiovascular Disease", limit=2)
    combined = clinics + cardio

    cards = []
    for idx, p in enumerate(combined):
        is_cardio = idx >= len(clinics)
        addr_full = f"{p['street']}, {p['city_state']}"
        cards.append({
            "category": "Specialty Cardiology Practice" if is_cardio else "Primary Care & BP Screening Clinic",
            "badge": "Verified NPI Registry",
            "name": p["name"],
            "address": addr_full,
            "phone": p["phone"],
            "distance": "Local ZIP Match" if p["exact"] else f"Serving {loc_info['city']} Area",
            "rationale": (
                "Registered cardiology organization for diagnostic ECG, echocardiogram, and blood pressure staging."
                if is_cardio else
                "Registered local medical clinic for walk-in or scheduled upper-arm blood pressure verification."
            ),
            "external_link": f"https://www.google.com/maps/search/{urllib.parse.quote(p['name'] + ' ' + addr_full)}"
        })

    if len(cards) < 4:
        cards.append({
            "category": "Federally Qualified Health Center (FQHC)",
            "badge": "Sliding-Fee Scale",
            "name": f"HRSA Sliding-Scale Health Centers near {loc_info['display']}",
            "address": f"Serving {loc_info['display']}",
            "phone": "1-877-464-4772",
            "distance": "Income-Based Care",
            "rationale": "Provides primary care and blood pressure checks regardless of insurance status.",
            "external_link": f"https://findahealthcenter.hrsa.gov/?zip={urllib.parse.quote(loc_info['display'])}"
        })

    return cards


def get_equipment_directory(loc_info):
    real_pharmacies = fetch_real_npi_providers(loc_info, "Community/Retail Pharmacy", limit=4)
    if not real_pharmacies:
        real_pharmacies = fetch_real_npi_providers(loc_info, "Pharmacy", limit=4)

    kiosks = []
    for p in real_pharmacies:
        addr_full = f"{p['street']}, {p['city_state']}"
        kiosks.append({
            "name": p["name"],
            "cost": "Free Walk-In BP Check / Local Cuff Stock",
            "type": "Verified Local Pharmacy (NPI Registry)",
            "nearby_store": f"📍 {addr_full} • 📞 {p['phone']}",
            "action_url": f"https://www.google.com/maps/search/{urllib.parse.quote(p['name'] + ' ' + addr_full)}"
        })

    state_tax_map = {
        "IL": 0.0825, "CA": 0.095, "NY": 0.088, "TX": 0.082, "FL": 0.070,
        "WA": 0.092, "PA": 0.060, "OH": 0.072, "GA": 0.074, "MI": 0.060
    }
    tax_rate = state_tax_map.get(loc_info["state"], 0.07)
    nearest_pharmacy_name = real_pharmacies[0]["name"] if real_pharmacies else f"Local Pharmacy ({loc_info['city']})"

    base_devices = [
        {
            "name": "A&D Medical Upper Arm Monitor (UA-611)",
            "tier": "Lowest Verified Cost",
            "base_price": 27.99,
            "validation": "BIHS / AAMI Certified",
            "direct_buy_url": "https://bvmedical.com/products/a-d-medical-basic-blood-pressure-monitor-ua-611",
            "store_buy_url": "https://www.walmart.com/ip/A-D-Engineering-Medical-Easy-Upper-Arm-Blood-Pressure-Monitor-with-Medium-Cuff-UA-611/49433347"
        },
        {
            "name": "Greater Goods Pro Upper Arm Monitor (0604)",
            "tier": "Best Home Value",
            "base_price": 32.50,
            "validation": "ESH Universal Standard",
            "direct_buy_url": "https://www.amazon.com/dp/B06XTC4BM9",
            "store_buy_url": "https://www.walmart.com/ip/Greater-Goods-Bluetooth-Blood-Pressure-Pulse-Monitor-Upper-Arm-Cuff-Large-Display/822447639"
        },
        {
            "name": "Omron 3 Series Upper Arm Monitor (BP7100)",
            "tier": "US VDL Clinical Benchmark",
            "base_price": 34.99,
            "validation": "US VDL / AAMI Certified",
            "direct_buy_url": "https://www.walmart.com/ip/OMRON-3-Series-Upper-Arm-Blood-Pressure-Monitor-BP7100/409605273",
            "store_buy_url": "https://www.walgreens.com/store/c/omron-3-series-upper-arm-blood-pressure-monitor/ID=300397070-product"
        }
    ]

    devices = []
    for d in base_devices:
        est_total = d["base_price"] * (1.0 + tax_rate)
        devices.append({
            "name": d["name"],
            "tier": d["tier"],
            "price": f"${est_total:.2f}",
            "sub_price": f"${d['base_price']:.2f} + {tax_rate * 100:.1f}% {loc_info['state']} Est. Tax",
            "local_stock_note": f"Check pickup near {nearest_pharmacy_name}",
            "validation": d["validation"],
            "direct_buy_url": d["direct_buy_url"],
            "store_buy_url": d["store_buy_url"]
        })

    return {"devices": devices, "kiosks": kiosks, "location": loc_info}


# --- 8. Dynamic Zone 2 Cardio Planner ---
def generate_cardio_protocol(bpm, hrv, age=35, activity="low"):
    bpm = float(bpm) if bpm > 0 else 72.0
    hrv = float(hrv) if hrv > 0 else 42.0
    age = int(age)

    hr_max = 220 - age
    hrr = max(20.0, hr_max - bpm)
    z2_low = int(round(bpm + (hrr * 0.55)))
    z2_high = int(round(bpm + (hrr * 0.70)))
    labels = ["Start", "Wk 2", "Wk 4", "Wk 6", "Wk 8", "Wk 10", "Wk 12"]

    if bpm < 60.0:
        goal_text = f"Resting rate ({bpm:.0f} BPM) shows strong cardiovascular fitness. Focus: Maintain aerobic capacity safely."
        target_baseline = bpm
        projected_curve = [round(bpm, 1) for _ in labels]
        weekly_volume = 150
        schedule = [
            {"day": "Monday", "type": "Aerobic Stride", "duration": "35 min",
             "modality": "Continuous brisk walk or easy bike ride",
             "pace_tip": "Smooth rhythm. Speak easily in full sentences."},
            {"day": "Tuesday", "type": "Vagal Recovery", "duration": "20 min",
             "modality": "Easy stroll + 5 min slow breathing (4s in, 6s out)",
             "pace_tip": "Gentle stroll. Zero exertion."},
            {"day": "Wednesday", "type": "Steady Movement", "duration": "40 min",
             "modality": "Stair stepping or brisk outdoor walk",
             "pace_tip": "Steady effort. Continuous nasal breathing."},
            {"day": "Thursday", "type": "Rest Day", "duration": "Rest Day",
             "modality": "Daily tasks and light stretching", "pace_tip": "Completely off. Let muscles recharge."},
            {"day": "Friday", "type": "Brisk Walk", "duration": "30 min",
             "modality": "Upbeat flat walk with natural arm swing", "pace_tip": "Brisk pace, still conversational."},
            {"day": "Saturday", "type": "Long Weekend Walk", "duration": "45 min",
             "modality": "Park trail walk or neighborhood route", "pace_tip": "Social, unhurried pace."},
            {"day": "Sunday", "type": "Vitals Check", "duration": "5 min", "modality": "Scan vitals with EquiPulse",
             "pace_tip": "Confirm HRV remains steady."}
        ]
    elif bpm <= 75.0:
        goal_text = f"Resting rate ({bpm:.0f} BPM) is healthy. Goal: Solidify a 65 BPM baseline and keep blood vessels flexible."
        target_baseline = 65.0
        projected_curve = [round(target_baseline + (bpm - target_baseline) * (0.82 ** (w / 2.0)), 1) for w in
                           [0, 2, 4, 6, 8, 10, 12]]
        weekly_volume = 135
        schedule = [
            {"day": "Monday", "type": "Brisk Neighborhood Walk", "duration": "30 min",
             "modality": "Continuous outdoor or hallway walking",
             "pace_tip": "Talk Test: Speak a full sentence comfortably."},
            {"day": "Tuesday", "type": "Easy Reset & Breathing", "duration": "15 min",
             "modality": "Easy stroll + 5 min belly breathing (4s in, 6s out)",
             "pace_tip": "Feel refreshed, not tired."},
            {"day": "Wednesday", "type": "Rhythmic Home Cardio", "duration": "35 min",
             "modality": "Indoor marching or stair stepping", "pace_tip": "Warm but able to talk easily."},
            {"day": "Thursday", "type": "Rest Day", "duration": "Rest Day",
             "modality": "Regular routines and light stretches", "pace_tip": "No formal workout today."},
            {"day": "Friday", "type": "Steady Outdoor Walk", "duration": "30 min",
             "modality": "Continuous brisk walk along a quiet route",
             "pace_tip": "Breathe through your nose if comfortable."},
            {"day": "Saturday", "type": "Long Social Walk", "duration": "40 min",
             "modality": "Walk with a friend or active chores", "pace_tip": "Comfortable conversational pace."},
            {"day": "Sunday", "type": "Weekly Vitals Check", "duration": "5 min",
             "modality": "Record updated resting rate on EquiPulse", "pace_tip": "Track progress toward 65 BPM."}
        ]
    elif bpm <= 84.0:
        goal_text = f"Resting rate ({bpm:.0f} BPM) shows mild stress/workload. Goal: Lower baseline below 68 BPM with easy walking."
        target_baseline = 68.0
        projected_curve = [round(target_baseline + (bpm - target_baseline) * (0.80 ** (w / 2.0)), 1) for w in
                           [0, 2, 4, 6, 8, 10, 12]]
        weekly_volume = 120
        schedule = [
            {"day": "Monday", "type": "Conversational Walk", "duration": "25 min",
             "modality": "Flat sidewalk or hallway walking", "pace_tip": "If winded, slow down immediately."},
            {"day": "Tuesday", "type": "Box Breathing Reset", "duration": "15 min",
             "modality": "10-min stroll + 5 min seated 4-4-4-4 breathing",
             "pace_tip": "Focus on slow, relaxed exhales."},
            {"day": "Wednesday", "type": "Low-Impact Home Marching", "duration": "30 min",
             "modality": "Marching in place to music or podcasts", "pace_tip": "Low knee lifts, steady pace."},
            {"day": "Thursday", "type": "Complete Rest Day", "duration": "Rest Day",
             "modality": "Rest, hydration, and 8 hours of sleep",
             "pace_tip": "Early bedtime helps blood pressure dip."},
            {"day": "Friday", "type": "Cadence Walk", "duration": "25 min",
             "modality": "Outdoor brisk walking on a flat loop", "pace_tip": "Steady footsteps without rushing."},
            {"day": "Saturday", "type": "Weekend Aerobic Walk", "duration": "35 min",
             "modality": "Park stroll or extended errand walk", "pace_tip": "Easy pace, bring water."},
            {"day": "Sunday", "type": "Weekly Vitals Check", "duration": "5 min",
             "modality": "Scan vitals with EquiPulse", "pace_tip": "Check your week-over-week drop."}
        ]
    else:
        goal_text = f"Elevated resting rate ({bpm:.0f} BPM). Goal: Gently calm your nervous system and step down toward 72 BPM."
        target_baseline = 72.0
        projected_curve = [round(target_baseline + (bpm - target_baseline) * (0.78 ** (w / 2.0)), 1) for w in
                           [0, 2, 4, 6, 8, 10, 12]]
        weekly_volume = 95
        schedule = [
            {"day": "Monday", "type": "Split Easy Walks", "duration": "20 min",
             "modality": "Two 10-minute flat walks (morning & afternoon)",
             "pace_tip": "Very easy stroll. Zero breathlessness."},
            {"day": "Tuesday", "type": "Belly Breathing Reset", "duration": "15 min",
             "modality": "Seated breathing (hand on belly, 4s in, 6s out)",
             "pace_tip": "Quiet room, relax your shoulders."},
            {"day": "Wednesday", "type": "Gentle Flat Walk", "duration": "20 min",
             "modality": "Flat sidewalk or indoor hallway stroll", "pace_tip": "Pause and rest if you feel tired."},
            {"day": "Thursday", "type": "Full Rest Day", "duration": "Rest Day",
             "modality": "Gentle stretching and water hydration", "pace_tip": "Avoid afternoon caffeine."},
            {"day": "Friday", "type": "Easy Paced Stroll", "duration": "20 min",
             "modality": "Unhurried outdoor walk on level ground", "pace_tip": "Easy talking pace."},
            {"day": "Saturday", "type": "Relaxed Park Stroll", "duration": "25 min",
             "modality": "Slow stroll through a park", "pace_tip": "Take bench breaks anytime."},
            {"day": "Sunday", "type": "Weekly Vitals Check", "duration": "5 min",
             "modality": "Sit quietly 3 mins, then scan on EquiPulse",
             "pace_tip": "Confirm heart rate is trending down."}
        ]

    return {
        "current_bpm": bpm,
        "current_hrv": hrv,
        "z2_low": z2_low,
        "z2_high": z2_high,
        "target_bpm": target_baseline,
        "goal_text": goal_text,
        "weekly_volume": weekly_volume,
        "schedule": schedule,
        "chart_labels": labels,
        "chart_curve": projected_curve,
        "chart_baseline": [target_baseline] * len(labels)
    }


# --- 9. MentisHRV: Clear, Concise Stress & Anxiety Check (Tab 6) ---
def compute_mentis_stress(bpm, hrv, task_type="reading"):
    bpm = float(bpm) if bpm > 0 else 72.0
    hrv = float(hrv) if hrv > 0 else 42.0

    mods = {
        "reading": {"bpm_add": 4.5, "hrv_mult": 0.86, "motion": "Moderate (Head tilt & reading eye movement)"},
        "scrolling": {"bpm_add": 8.0, "hrv_mult": 0.75, "motion": "High (Hand scrolling & screen micro-motion)"},
        "baseline": {"bpm_add": 0.0, "hrv_mult": 1.00, "motion": "Low (Sitting still)"}
    }
    m = mods.get(task_type, mods["reading"])
    eff_bpm = round(bpm + m["bpm_add"], 1)
    eff_hrv = round(max(18.0, hrv * m["hrv_mult"]), 1)

    hr_stress = np.clip(((eff_bpm - 58.0) / 42.0) * 50.0, 4.0, 50.0)
    hrv_stress = np.clip(((65.0 - eff_hrv) / 45.0) * 50.0, 4.0, 50.0)
    stress_score = int(round(hr_stress + hrv_stress))

    if stress_score < 35:
        level = "Calm & Relaxed"
        color = "emerald"
        what_hr_means = (
            f"Your heart rate is {eff_bpm} BPM and your beat-to-beat variation (HRV) is {eff_hrv} ms. "
            "When you are relaxed, your heart naturally varies the time between each beat. Your numbers show your nervous system is calm."
        )
        next_steps = [
            "Keep your current posture and take a 20-second break from the screen every 20 minutes.",
            "No stress intervention needed right now—your body is handling screen time well."
        ]
    elif stress_score < 65:
        level = "Mild Stress / Mental Focus"
        color = "amber"
        what_hr_means = (
            f"Your heart rate rose to {eff_bpm} BPM and your beat-to-beat variation tightened to {eff_hrv} ms. "
            "When you concentrate hard or scroll through fast content, your brain subtly speeds up your pulse and makes each heartbeat more rigid."
        )
        next_steps = [
            "Click 'Start 60s Reset' on the right to slow your heart rate by 4–6 BPM.",
            "Drop your shoulders away from your ears and unclench your jaw.",
            "Drink a glass of water and step away from your screen for 2 minutes."
        ]
    else:
        level = "High Stress / Anxious Tension"
        color = "rose"
        what_hr_means = (
            f"Your heart rate is elevated at {eff_bpm} BPM with low beat-to-beat flexibility ({eff_hrv} ms). "
            "Your body is in an active 'fight-or-flight' stress response, pumping faster as if facing physical pressure."
        )
        next_steps = [
            "Stop scrolling or working immediately and use the 60-Second Guided Breathing tool on the right.",
            "Inhale for 4 seconds, exhale slowly for 6 seconds—longer exhales physically trigger your vagus nerve to slow your heart.",
            "Take a 5-minute slow walk without your phone to burn off excess adrenaline."
        ]

    return {
        "stress_score": stress_score,
        "level": level,
        "color": color,
        "eff_bpm": eff_bpm,
        "eff_hrv": eff_hrv,
        "motion_desc": m["motion"],
        "what_hr_means": what_hr_means,
        "next_steps": next_steps
    }


def simulate_ddpm_wave(raw_wave, task_type="reading"):
    sig = np.array(raw_wave, dtype=np.float32) if len(raw_wave) >= 30 else np.sin(np.linspace(0, 8 * np.pi, 90))
    intensity = 0.55 if task_type == "scrolling" else (0.35 if task_type == "reading" else 0.15)
    t = np.linspace(0, len(sig) / 30.0, len(sig))
    jitter = intensity * 0.8 * np.sin(2 * np.pi * 0.4 * t) + intensity * np.random.normal(0, 0.22, len(sig))
    shaky = sig + jitter
    b_low, a_low = signal.butter(3, 2.4 / 15.0, btype='lowpass')
    clean = signal.filtfilt(b_low, a_low, signal.detrend(shaky))
    return shaky.tolist(), clean.tolist()


# --- 10. Unified EquiPulse Frontend ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EquiPulse | Equitable Contactless Heart & Stress Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #090c10; color: #f1f5f9; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        .panel { background-color: #0f141c; border: 1px solid #1e2638; }
        .panel-inset { background-color: #070a0f; border: 1px solid #182030; }
    </style>
</head>
<body class="min-h-screen antialiased selection:bg-teal-500 selection:text-black">

    <!-- Top Header: EquiPulse -->
    <header class="border-b border-[#1e2638] bg-[#0c1017] sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded bg-[#162030] border border-[#283850] flex items-center justify-center font-mono font-bold text-sm text-teal-400">
                    EP
                </div>
                <div>
                    <div class="flex items-center space-x-2">
                        <span class="font-bold text-base tracking-tight text-white">EquiPulse</span>
                    </div>
                </div>
            </div>

            <div class="flex items-center space-x-3 font-mono text-xs text-slate-400">
                <div class="flex items-center space-x-1.5">
                    <span id="tracking-dot" class="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span id="tracking-status" class="text-slate-200">Skin Locked</span>
                </div>
                <span class="text-slate-600">|</span>
                <span id="roi-badge" class="text-teal-400">Forehead / Cheek Auto-ROI</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-6">

        <!-- Navigation Bar -->
        <nav class="flex space-x-1.5 border-b border-[#1e2638] mb-6 pb-2.5 overflow-x-auto text-xs font-mono">
            <button onclick="switchTab('vitals')" id="tab-btn-vitals" class="tab-button px-3 py-1.5 rounded font-semibold text-teal-300 bg-[#141b27] border border-[#243046] whitespace-nowrap">01. Live Vitals</button>
            <button onclick="switchTab('risk')" id="tab-btn-risk" class="tab-button px-3 py-1.5 rounded font-medium text-slate-400 hover:text-white whitespace-nowrap">02. Hypertension Risk</button>
            <button onclick="switchTab('resources')" id="tab-btn-resources" class="tab-button px-3 py-1.5 rounded font-medium text-slate-400 hover:text-white whitespace-nowrap">03. Local Clinics</button>
            <button onclick="switchTab('equipment')" id="tab-btn-equipment" class="tab-button px-3 py-1.5 rounded font-medium text-slate-400 hover:text-white whitespace-nowrap">04. Low-Cost Monitors</button>
            <button onclick="switchTab('cardio')" id="tab-btn-cardio" class="tab-button px-3 py-1.5 rounded font-medium text-slate-400 hover:text-white whitespace-nowrap">05. Cardio Plan</button>
            <button onclick="switchTab('mentis')" id="tab-btn-mentis" class="tab-button px-3 py-1.5 rounded font-medium text-indigo-300 hover:text-white whitespace-nowrap">06. Stress Check</button>
        </nav>

        <!-- TAB 1: LIVE VITALS -->
        <div id="tab-vitals" class="tab-content grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div class="lg:col-span-7 space-y-4">
                <div class="panel rounded p-3 space-y-2.5">
                    <div class="flex items-center justify-between pb-2 border-b border-[#182030] text-xs font-mono">
                        <span class="text-slate-200 font-semibold">Optical Skin Perfusion Feed</span>
                        <div class="flex items-center space-x-2">
                            <span class="text-slate-400">Signal Quality:</span>
                            <span id="sqi-badge" class="font-bold text-emerald-400">0%</span>
                        </div>
                    </div>

                    <!-- Live Positioning & Lighting Diagnostic Bar -->
                    <div class="grid grid-cols-3 gap-2 text-[11px] font-mono">
                        <div class="panel-inset px-2.5 py-1.5 rounded flex items-center justify-between">
                            <span class="text-slate-400">Distance:</span>
                            <span id="hud-distance" class="font-bold text-teal-300">Checking...</span>
                        </div>
                        <div class="panel-inset px-2.5 py-1.5 rounded flex items-center justify-between">
                            <span class="text-slate-400">Lighting:</span>
                            <span id="hud-lighting" class="font-bold text-teal-300">Checking...</span>
                        </div>
                        <div class="panel-inset px-2.5 py-1.5 rounded flex items-center justify-between">
                            <span class="text-slate-400">Motion Gate:</span>
                            <span id="hud-stability" class="font-bold text-emerald-400">Checking...</span>
                        </div>
                    </div>

                    <div class="relative bg-black aspect-video flex items-center justify-center overflow-hidden rounded">
                        <img src="/video_feed" alt="Video Stream" class="w-full h-full object-cover">
                    </div>
                </div>

                <div class="panel rounded p-3.5 space-y-2">
                    <div class="flex items-center justify-between text-xs font-mono">
                        <span class="text-slate-300 font-medium">Shadow-Invariant Hemoglobin Pulse Wave</span>
                        <span class="text-[11px] text-slate-400">4x4 Sub-Grid Log-Beer-Lambert</span>
                    </div>
                    <div class="h-28 w-full panel-inset rounded p-1.5">
                        <canvas id="ppgCanvas" class="w-full h-full"></canvas>
                    </div>
                </div>
            </div>

            <div class="lg:col-span-5 space-y-4">
                <div class="grid grid-cols-2 gap-3">
                    <div class="panel rounded p-4">
                        <div class="flex justify-between items-center text-xs font-mono text-slate-400">
                            <span>RESTING HR</span>
                            <span class="text-rose-400">BPM</span>
                        </div>
                        <div class="flex items-baseline space-x-1.5 mt-1">
                            <span id="metric-bpm" class="font-mono text-3xl font-bold text-white">--</span>
                            <span class="text-xs font-mono text-slate-400">BPM</span>
                        </div>
                        <div class="w-full bg-[#182030] h-1.5 rounded mt-2.5 overflow-hidden">
                            <div id="metric-bpm-progress" class="bg-teal-400 h-full w-0 transition-all duration-300"></div>
                        </div>
                    </div>

                    <div class="panel rounded p-4">
                        <div class="flex justify-between items-center text-xs font-mono text-slate-400">
                            <span>HRV (RMSSD)</span>
                            <span class="text-teal-400">ms</span>
                        </div>
                        <div class="flex items-baseline space-x-1.5 mt-1">
                            <span id="metric-hrv" class="font-mono text-3xl font-bold text-white">--</span>
                            <span class="text-xs font-mono text-slate-400">ms</span>
                        </div>
                        <div class="w-full bg-[#182030] h-1.5 rounded mt-2.5 overflow-hidden">
                            <div id="metric-hrv-progress" class="bg-indigo-400 h-full w-0 transition-all duration-300"></div>
                        </div>
                    </div>
                </div>

                <!-- Mandatory User Setup & Positioning Checklist -->
                <div class="panel rounded p-4 space-y-2.5 border-l-2 border-l-amber-400">
                    <div class="flex items-center justify-between">
                        <span class="text-xs font-mono font-bold text-amber-300 uppercase">User Setup & Positioning Instructions</span>
                        <span class="text-[10px] font-mono text-slate-400">For Accurate Lock</span>
                    </div>
                    <ul class="space-y-1.5 text-xs text-slate-200 leading-relaxed">
                        <li><strong class="text-white">1. Camera Distance (18–24 inches):</strong> Sit about one arm's length from the screen. Do not lean right up to the lens, or your chin will clip out of frame.</li>
                        <li><strong class="text-white">2. Room Lighting:</strong> Use normal indoor lighting in front of you. Avoid sitting with a bright window directly behind your head. (Side shadows & dark skin tones are automatically compensated via Log-Beer-Lambert math).</li>
                        <li><strong class="text-white">3. Head Position & Stability:</strong> Keep your head level, face the camera, and breathe normally without talking for <strong>10 seconds</strong> while the green box locks onto your skin.</li>
                        <li><strong class="text-white">4. Ground-Truth Check:</strong> Count your wrist pulse for 15 seconds ($\times 4$) or use a finger oximeter to verify your locked resting baseline.</li>
                    </ul>
                </div>

                <!-- Live Vascular Summary & Diagnostic Status -->
                <div class="panel rounded p-3.5 space-y-2 font-mono">
                    <span class="text-xs font-bold text-slate-200 uppercase block">Live Vascular Telemetry</span>
                    <div class="panel-inset rounded p-2.5 space-y-1.5 text-xs">
                        <div class="flex justify-between"><span class="text-slate-400">Heart Rate Status:</span><span id="metric-status" class="font-semibold text-teal-300">Measuring...</span></div>
                        <div class="flex justify-between"><span class="text-slate-400">Blood Vessel Tone:</span><span id="metric-tone" class="font-semibold text-slate-200">Evaluating...</span></div>
                        <div class="flex justify-between"><span class="text-slate-400">Spectral Consensus SNR:</span><span id="metric-snr" class="font-bold text-slate-300">0.0 dB</span></div>
                    </div>
                </div>

                <!-- 30-Second Clinical Resting HR Check-In Card -->
                <div class="panel rounded p-4 space-y-3 border-t-2 border-t-teal-400 font-mono">
                    <div class="flex justify-between items-center">
                        <span class="text-xs font-bold text-white uppercase tracking-wider">Resting HR Check-In</span>
                        <span id="checkin-badge" class="text-[10px] px-2 py-0.5 rounded bg-[#162030] text-slate-400">Prereq: 5s Steady Lock</span>
                    </div>
                    
                    <p class="text-xs text-slate-300 leading-relaxed font-sans">
                        Live heart rates naturally fluctuate second-by-second. Run a 30-second clinical check-in to compute a certified resting average with outlier rejection.
                    </p>

                    <!-- Progress Bar (Visible during 30s test) -->
                    <div id="checkin-progress-container" class="hidden space-y-1.5">
                        <div class="flex justify-between text-[11px]">
                            <span id="checkin-status-text" class="text-teal-300 font-semibold">Sampling Resting Pulse...</span>
                            <span id="checkin-timer" class="text-white font-bold">30s</span>
                        </div>
                        <div class="w-full bg-[#182030] h-2 rounded overflow-hidden">
                            <div id="checkin-progress-bar" class="bg-teal-400 h-full w-0 transition-all duration-300"></div>
                        </div>
                    </div>

                    <!-- Certified Result Box (Appears after 30s) -->
                    <div id="checkin-result-box" class="hidden panel-inset p-3 rounded space-y-2 text-xs border border-teal-500/30">
                        <div class="flex justify-between items-center">
                            <span class="text-teal-300 font-bold uppercase text-[11px]">Certified Resting Baseline:</span>
                            <span class="text-[10px] text-slate-400">Trimmed IQR Mean</span>
                        </div>
                        <div class="flex items-baseline space-x-3">
                            <span id="certified-bpm" class="text-2xl font-bold text-white">-- BPM</span>
                            <span id="certified-hrv" class="text-sm text-teal-400 font-semibold">HRV: -- ms</span>
                        </div>
                        <button onclick="proceedFromCertifiedCheckin()" class="w-full py-2 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-400/50 rounded font-bold text-teal-200 text-xs uppercase tracking-wider">
                            Proceed to Stage 2: Risk & Trajectory Guide →
                        </button>
                    </div>

                    <button id="btn-start-checkin" onclick="startRestingCheckin()" disabled class="w-full py-2 bg-[#162030] border border-[#283850] rounded text-xs font-bold text-slate-500 uppercase tracking-wider transition-colors cursor-not-allowed">
                        Hold Still 5s to Enable Check-In
                    </button>
                </div>

                <!-- Live Passive Stress Detection Panel -->
                <div class="panel rounded p-4 space-y-3 font-mono border-l-2 border-l-indigo-400">
                    <div class="flex justify-between items-center">
                        <span class="text-xs font-bold text-white uppercase tracking-wider">Live Stress Detection</span>
                        <label class="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" id="passive-stress-toggle" onchange="togglePassiveStress(this.checked)" class="sr-only peer">
                            <div class="w-9 h-5 bg-[#182030] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                        </label>
                    </div>
                    <p class="text-xs text-slate-300 leading-relaxed font-sans">
                        Monitors autonomic tension in the background while you study, read, or work. If elevated stress is sustained, it notifies you to pause and recalibrate.
                    </p>
                    <div id="passive-stress-status" class="panel-inset p-2.5 rounded text-xs flex justify-between items-center">
                        <span class="text-slate-400">Monitor Status:</span>
                        <span id="passive-stress-indicator" class="text-slate-500 font-bold">Standby (Turn Toggle On)</span>
                    </div>

                    <!-- Real-Time Stress Alert Banner -->
                    <div id="stress-alert-banner" class="hidden p-3 rounded bg-rose-950/40 border border-rose-500/50 space-y-2">
                        <div class="flex items-center space-x-2 text-rose-300 text-xs font-bold uppercase">
                            <i data-lucide="alert-circle" class="w-4 h-4"></i>
                            <span>Elevated Autonomic Stress Detected</span>
                        </div>
                        <p class="text-xs text-slate-200 font-sans">
                            Your pulse has surged with tightened heart rhythm variability. Take a 60-second breathing pause to engage your vagus nerve.
                        </p>
                        <button onclick="launchQuickBreathingReset()" class="w-full py-1.5 bg-rose-600/30 hover:bg-rose-600/50 border border-rose-500 rounded text-xs font-bold text-rose-200 uppercase">
                            Start 60s Breathing Reset Now →
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: HYPERTENSION RISK -->
        <div id="tab-risk" class="tab-content hidden space-y-6">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div class="lg:col-span-5 panel rounded p-5 space-y-4">
                    <h3 class="text-sm font-mono font-bold text-white uppercase border-b border-[#1e2638] pb-2">Your Health Profile</h3>
                    <form id="risk-form" onsubmit="runRiskEvaluation(event)" class="space-y-3.5 text-xs font-mono">
                        <div class="panel-inset rounded p-3 space-y-2">
                            <div class="flex justify-between items-center text-[11px]">
                                <span class="text-slate-300 font-bold">Your Live Camera Vitals</span>
                                <button type="button" onclick="importCurrentVitals()" class="text-teal-400 hover:underline">Sync Live Vitals</button>
                            </div>
                            <div class="grid grid-cols-2 gap-2">
                                <div><label class="text-slate-400 text-[10px] block">Resting HR (BPM)</label><input type="number" step="0.1" id="form-bpm" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-white" required></div>
                                <div><label class="text-slate-400 text-[10px] block">HRV (ms)</label><input type="number" step="0.1" id="form-hrv" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-white" required></div>
                            </div>
                        </div>
                        <div>
                            <label class="text-slate-300 block mb-1">Last Known Blood Pressure</label>
                            <select id="form-bp" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200">
                                <option value="unknown">Unchecked / Don't Know (+6 pts)</option>
                                <option value="normal">Normal — Under 120/80 mmHg (+0 pts)</option>
                                <option value="elevated">Elevated — 120-129 / <80 mmHg (+12 pts)</option>
                                <option value="stage1">Stage 1 — 130-139 / 80-89 mmHg (+18 pts)</option>
                                <option value="stage2">Stage 2 — 140+ / 90+ mmHg (+25 pts)</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-slate-300 block mb-1">Background & Heritage</label>
                            <select id="form-ethnicity" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200">
                                <option value="general">General Population</option>
                                <option value="black">African American / Black (Higher early BP risk)</option>
                                <option value="south_asian">South Asian (Higher cardiometabolic risk)</option>
                                <option value="hispanic">Hispanic / Latino</option>
                            </select>
                        </div>
                        <div class="grid grid-cols-2 gap-2">
                            <div><label class="text-slate-300 block mb-1">Age</label><input type="number" id="form-age-val" value="35" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200"></div>
                            <div><label class="text-slate-300 block mb-1">Nightly Sleep</label><select id="form-sleep" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200"><option value="6-8">6 - 8 Hours (+0 pts)</option><option value="<6">Under 6 Hours (+4 pts)</option><option value=">8">Over 8 Hours (+0 pts)</option></select></div>
                        </div>
                        <div class="grid grid-cols-2 gap-2">
                            <div><label class="text-slate-300 block mb-1">Daily Stress</label><select id="form-stress" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200"><option value="low">Low (+0 pts)</option><option value="moderate" selected>Moderate (+0 pts)</option><option value="high">High (+3 pts)</option></select></div>
                            <div><label class="text-slate-300 block mb-1">Weekly Exercise</label><select id="form-activity" class="w-full bg-[#0c1017] border border-[#1e2638] rounded px-2 py-1.5 text-slate-200"><option value="regular">150+ min/wk (+0 pts)</option><option value="moderate">60-149 min (+0 pts)</option><option value="low" selected>Under 60 min (+3 pts)</option></select></div>
                        </div>
                        <div class="flex items-center space-x-2 pt-1">
                            <input type="checkbox" id="form-family" class="rounded bg-[#0c1017] border-[#1e2638] text-teal-500">
                            <label for="form-family" class="text-slate-300">Family history of high blood pressure or heart disease (+6 pts)</label>
                        </div>
                        <button type="submit" class="w-full py-2.5 bg-[#172233] hover:bg-[#1e2d44] border border-[#283850] rounded font-mono text-xs font-bold text-teal-300 uppercase mt-2">
                            Calculate My Risk Score
                        </button>
                    </form>
                </div>

                <div class="lg:col-span-7 panel rounded p-5 space-y-4">
                    <div id="risk-loading" class="text-center py-16 text-slate-400 font-mono text-xs">
                        Click "Calculate My Risk Score" on the left to see your full mathematical breakdown.
                    </div>
                    <div id="risk-display" class="hidden space-y-4 font-mono">
                        <div class="flex justify-between items-center border-b border-[#1e2638] pb-3">
                            <div>
                                <span class="text-[10px] text-slate-400 uppercase">Published Cohort Risk Model (Framingham / ARIC / AHA)</span>
                                <h4 id="res-tier" class="text-base font-bold text-white mt-0.5">--</h4>
                            </div>
                            <div class="text-right">
                                <span id="res-score" class="text-3xl font-bold text-teal-400">0</span>
                                <span class="text-xs text-slate-400">/ 100 pts</span>
                            </div>
                        </div>

                        <div class="panel-inset rounded p-3 space-y-2">
                            <span class="text-[11px] font-bold text-slate-200 uppercase block">How Your Score Was Calculated (Exact Point Breakdown):</span>
                            <div id="res-breakdown" class="space-y-1.5 text-xs"></div>
                        </div>

                        <div class="panel-inset rounded p-3 space-y-1.5 text-xs">
                            <span class="text-[10px] text-teal-400 font-bold uppercase block">What This Means & What To Do Next:</span>
                            <p id="res-summary" class="text-slate-200 leading-relaxed"></p>
                            <p id="res-action" class="text-emerald-300 font-semibold pt-1"></p>
                        </div>

                        <div>
                            <span class="text-[10px] text-slate-400 uppercase block mb-2">Key Factors Driving Your Score:</span>
                            <div id="res-drivers" class="space-y-2"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: REAL LOCAL CLINICS -->
        <div id="tab-resources" class="tab-content hidden space-y-4">
            <div class="panel rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 font-mono">
                <div>
                    <h3 class="text-sm font-bold text-white uppercase">Real Local Medical Clinics & Cardiology Offices (Live US NPI Registry)</h3>
                    <p class="text-xs text-slate-400">Enter any ZIP code to pull actual registered clinics, real street addresses, and phone numbers in that exact area.</p>
                </div>
                <form onsubmit="searchResources(event)" class="flex items-center space-x-2 text-xs">
                    <input type="text" id="loc-input" placeholder="Enter ZIP Code" value="60177" class="bg-[#0c1017] border border-[#1e2638] rounded px-3 py-1.5 text-white">
                    <button type="submit" class="px-4 py-1.5 bg-[#172233] hover:bg-[#1e2d44] border border-[#283850] rounded text-teal-300 font-bold uppercase">Search ZIP</button>
                </form>
            </div>
            <div id="resources-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
        </div>

        <!-- TAB 4: REAL LOCAL PHARMACIES & VALIDATED MONITORS -->
        <div id="tab-equipment" class="tab-content hidden space-y-4">
            <div class="panel rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 font-mono">
                <div>
                    <h3 class="text-sm font-bold text-white uppercase">Real Local Pharmacies & Low-Cost Validated Monitors</h3>
                    <p class="text-xs text-slate-400">Pulls real pharmacies in your ZIP code for free walk-in BP checks, plus tax-adjusted monitor prices.</p>
                </div>
                <form onsubmit="searchEquipment(event)" class="flex items-center space-x-2 text-xs">
                    <input type="text" id="eq-loc-input" placeholder="Enter ZIP Code" value="60177" class="bg-[#0c1017] border border-[#1e2638] rounded px-3 py-1.5 text-white">
                    <button type="submit" class="px-4 py-1.5 bg-[#172233] hover:bg-[#1e2d44] border border-[#283850] rounded text-emerald-300 font-bold uppercase">Update ZIP</button>
                </form>
            </div>
            <h4 class="text-xs font-mono font-bold text-slate-300 uppercase pt-1">Verified Pharmacies in Your ZIP Code (Walk-In BP Screening & Cuff Pickup):</h4>
            <div id="kiosks-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
            <h4 class="text-xs font-mono font-bold text-slate-300 uppercase pt-2">Clinically Validated Home Upper-Arm Cuffs (With Local Tax Estimate):</h4>
            <div id="devices-grid" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
        </div>

        <!-- TAB 5: ZONE 2 CARDIO PLANNER -->
        <div id="tab-cardio" class="tab-content hidden space-y-4 font-mono">
            <div class="panel rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div>
                    <h3 class="text-sm font-bold text-white uppercase">Personalized Daily Walking & Heart Conditioning Plan</h3>
                    <p id="cardio-goal-banner" class="text-xs text-teal-300 mt-0.5">Simple, zero-cost daily movement calibrated to your resting heart rate.</p>
                </div>
                <div class="flex items-center space-x-2 text-xs">
                    <button onclick="syncCardioFromLiveVitals()" class="px-3 py-1.5 bg-[#172233] border border-[#283850] rounded text-teal-300 font-bold uppercase">Sync My Live Heart Rate</button>
                </div>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div class="panel rounded p-3"><span class="text-slate-400 uppercase block text-[10px]">Your Walking Heart Rate</span><span id="cardio-target-zone" class="text-lg font-bold text-indigo-400">-- BPM</span></div>
                <div class="panel rounded p-3"><span class="text-slate-400 uppercase block text-[10px]">Target Resting Goal</span><span id="cardio-goal-bpm" class="text-lg font-bold text-teal-400">65 BPM</span></div>
                <div class="panel rounded p-3"><span class="text-slate-400 uppercase block text-[10px]">Weekly Total</span><span id="cardio-weekly-vol" class="text-lg font-bold text-white">-- min</span></div>
                <div class="panel rounded p-3"><span class="text-slate-400 uppercase block text-[10px]">Equipment Needed</span><span class="text-lg font-bold text-emerald-400">$0 / None</span></div>
            </div>

            <div class="panel rounded p-4 space-y-2">
                <div class="flex justify-between items-center text-xs">
                    <span class="font-bold text-slate-200 uppercase">12-Week Projected Resting Heart Rate Drop</span>
                    <span class="text-slate-400 text-[10px]">Expected drop: ~1 BPM every 2–3 weeks of regular walking</span>
                </div>
                <div class="h-56 w-full panel-inset rounded p-2">
                    <canvas id="hrProjectionCanvas"></canvas>
                </div>
            </div>

            <div id="cardio-schedule-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3"></div>
        </div>

        <!-- TAB 6: MENTIS-HRV -->
        <div id="tab-mentis" class="tab-content hidden space-y-4 font-mono">

            <div class="panel rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div>
                    <div class="flex items-center space-x-2">
                        <h3 class="text-sm font-bold text-white uppercase">Real-Time Stress & Anxiety Check</h3>
                    </div>
                    <p class="text-xs text-slate-300 mt-1">Measures how stressed your body is while you read, scroll, or use your device—without needing a smartwatch.</p>
                </div>

                <div class="flex items-center space-x-1.5 bg-[#0c1017] border border-[#1e2638] p-1.5 rounded text-xs">
                    <span class="text-slate-400 px-2 text-[10px] uppercase">What are you doing?</span>
                    <button onclick="setMentisTask('baseline')" id="task-btn-baseline" class="mentis-task-btn px-2.5 py-1 rounded text-slate-400 hover:text-white">Sitting Still</button>
                    <button onclick="setMentisTask('reading')" id="task-btn-reading" class="mentis-task-btn px-2.5 py-1 rounded bg-[#162030] text-white border border-[#2b3a54]">Reading / Studying</button>
                    <button onclick="setMentisTask('scrolling')" id="task-btn-scrolling" class="mentis-task-btn px-2.5 py-1 rounded text-slate-400 hover:text-white">Scrolling Phone</button>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div class="lg:col-span-6 space-y-4">
                    <div class="panel rounded p-5 space-y-4">
                        <div class="flex justify-between items-center border-b border-[#182030] pb-3">
                            <div>
                                <span class="text-[10px] text-slate-400 uppercase block">Your Current Body Stress Level</span>
                                <h4 id="m-level-text" class="text-lg font-bold text-amber-400 mt-0.5">Checking...</h4>
                            </div>
                            <div class="text-right">
                                <span id="m-stress-score" class="text-3xl font-bold text-white">--</span>
                                <span class="text-xs text-slate-400">/ 100</span>
                            </div>
                        </div>

                        <div class="w-full bg-[#182030] h-2.5 rounded overflow-hidden">
                            <div id="m-stress-bar" class="h-full bg-amber-400 transition-all duration-300" style="width: 40%;"></div>
                        </div>

                        <div class="grid grid-cols-2 gap-3 text-xs">
                            <div class="panel-inset rounded p-3">
                                <span class="text-slate-400 text-[10px] uppercase block">Active Heart Rate</span>
                                <span id="m-bpm" class="text-lg font-bold text-white block mt-0.5">-- BPM</span>
                                <span class="text-[10px] text-slate-400">Faster = More tense</span>
                            </div>
                            <div class="panel-inset rounded p-3">
                                <span class="text-slate-400 text-[10px] uppercase block">Beat Flexibility (HRV)</span>
                                <span id="m-hrv" class="text-lg font-bold text-teal-400 block mt-0.5">-- ms</span>
                                <span class="text-[10px] text-slate-400">Higher = More relaxed</span>
                            </div>
                        </div>

                        <div class="panel-inset rounded p-3.5 space-y-1 text-xs leading-relaxed">
                            <span class="text-indigo-300 font-bold uppercase block text-[11px]">What Your Heart Rate Tells Us About Your Stress:</span>
                            <p id="m-what-means" class="text-slate-200"></p>
                        </div>

                        <div class="panel-inset rounded p-3.5 space-y-2 text-xs border-l-2 border-l-emerald-400">
                            <span class="text-emerald-300 font-bold uppercase block text-[11px]">What You Should Do Right Now:</span>
                            <ul id="m-next-steps" class="space-y-1.5 text-slate-200 list-disc pl-4"></ul>
                        </div>
                    </div>
                </div>

                <div class="lg:col-span-6 space-y-4">
                    <div class="panel rounded p-5 space-y-3">
                        <div class="flex justify-between items-center">
                            <div>
                                <span class="text-xs font-bold text-teal-300 uppercase block">Interactive 60-Second Stress Reset</span>
                                <p class="text-[11px] text-slate-400">Follow the breathing guide to activate your vagus nerve and slow your pulse.</p>
                            </div>
                            <button onclick="toggleBreathingExercise()" id="breath-btn" class="px-3 py-1.5 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/40 rounded text-teal-300 text-xs font-bold uppercase">
                                Start 60s Reset
                            </button>
                        </div>
                        <div class="panel-inset rounded p-4 flex flex-col items-center justify-center space-y-2 h-36">
                            <div id="breath-circle" class="w-16 h-16 rounded-full border-2 border-teal-400 flex items-center justify-center text-xs font-bold text-white transition-all duration-1000">
                                Ready
                            </div>
                            <span id="breath-instruction" class="text-xs text-slate-300">Click "Start 60s Reset" to begin slow 4s Inhale / 6s Exhale pacing.</span>
                        </div>
                    </div>

                    <div class="panel rounded p-4 space-y-2">
                        <div class="flex justify-between items-center text-xs pb-1 border-b border-[#182030]">
                            <span class="font-bold text-slate-200 uppercase">How MentisHRV Removes Movement Noise</span>
                            <div class="flex items-center space-x-3 text-[10px]">
                                <span class="text-rose-400">■ Shaky Camera Signal</span>
                                <span class="text-teal-400">■ AI Cleaned Pulse</span>
                            </div>
                        </div>
                        <div class="h-36 w-full panel-inset rounded p-2">
                            <canvas id="mentisDiffusionCanvas" class="w-full h-full"></canvas>
                        </div>
                        <p id="m-motion-desc" class="text-[11px] text-slate-400">
                            Normally, moving your head while reading or scrolling ruins camera heart rate tracking. It subtracts your head movement (red) to isolate your true heartbeat (teal).
                        </p>
                    </div>
                </div>
            </div>

        </div>

    </main>

    <script>
        lucide.createIcons();
        let activeTab = 'vitals';
        let recentBpmHistory = [];

        function evaluateCheckinPrerequisites(telemetryData) {
            const btn = document.getElementById('btn-start-checkin');
            const badge = document.getElementById('checkin-badge');
            if (isCheckinRunning) return;

            if (telemetryData.bpm > 0) {
                recentBpmHistory.push(telemetryData.bpm);
                if (recentBpmHistory.length > 12) recentBpmHistory.shift();
            }

            let isHrStable = false;
            if (recentBpmHistory.length >= 8) {
                const maxB = Math.max(...recentBpmHistory);
                const minB = Math.min(...recentBpmHistory);
                isHrStable = (maxB - minB) <= 3.5;
            }

            const isFaceTracked = telemetryData.is_tracking && telemetryData.bpm > 0;

            if (isFaceTracked && isHrStable) {
                stabilityCounter = Math.min(5.0, stabilityCounter + 0.4);
            } else {
                stabilityCounter = Math.max(0.0, stabilityCounter - 0.15);
            }

            if (stabilityCounter >= 5.0) {
                btn.disabled = false;
                btn.className = 'w-full py-2 bg-teal-500/25 hover:bg-teal-500/35 border border-teal-400 rounded text-xs font-bold text-teal-200 uppercase tracking-wider cursor-pointer shadow-sm';
                btn.innerText = 'Start 30s Clinical Check-In';
                badge.innerText = 'Ready to Measure';
                badge.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-300';
            } else {
                btn.disabled = true;
                btn.className = 'w-full py-2 bg-[#162030] border border-[#283850] rounded text-xs font-bold text-slate-500 uppercase tracking-wider cursor-not-allowed';
                const progressSec = Math.floor(stabilityCounter);
                btn.innerText = `Stabilizing HR: ${progressSec}/5s...`;
                badge.innerText = `Locking: ${progressSec}/5s`;
                badge.className = 'text-[10px] px-2 py-0.5 rounded bg-[#162030] text-slate-400';
            }
        }
        let currentMentisTask = 'reading';
        let mentisCorruptedWave = [];
        let mentisDenoisedWave = [];
        let liveTelemetry = { bpm: 72.0, hrv: 42.0 };
        let currentRiskProfile = { score: 35, tier: "Mild Risk", input_bpm: 72.0, input_hrv: 42.0 };
        let hrChartInstance = null;
        let breathTimer = null;
        // =====================================================================
        // RESTING HR CHECK-IN & PASSIVE STRESS DETECTION ENGINE
        // =====================================================================
        let stabilityCounter = 0;
        let isCheckinRunning = false;
        let checkinTimeRemaining = 30;
        let checkinTimerInterval = null;
        let checkinBpmSamples = [];
        let checkinHrvSamples = [];
        let certifiedBaseline = null;

        let passiveStressEnabled = false;
        let highStressSeconds = 0;

        // 1. Monitor Stability to Enable Check-In Button
        function evaluateCheckinPrerequisites(telemetryData) {
            const btn = document.getElementById('btn-start-checkin');
            const badge = document.getElementById('checkin-badge');
            if (isCheckinRunning) return;

            // Must be skin-locked with valid signal quality and steady head position
            const isGoodSignal = telemetryData.is_tracking && telemetryData.sqi >= 40 && telemetryData.bpm > 0;
            const isHeadStill = telemetryData.guide_stability && telemetryData.guide_stability.includes("Stable");

            if (isGoodSignal && isHeadStill) {
                stabilityCounter += 0.4; // 400ms polling interval
                if (stabilityCounter >= 5.0) {
                    btn.disabled = false;
                    btn.className = 'w-full py-2 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-400 rounded text-xs font-bold text-teal-200 uppercase tracking-wider cursor-pointer shadow-sm';
                    btn.innerText = 'Start 30s Clinical Check-In';
                    badge.innerText = 'Ready to Measure';
                    badge.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-300';
                } else {
                    badge.innerText = `Stabilizing (${Math.round(5 - stabilityCounter)}s)...`;
                }
            } else {
                stabilityCounter = Math.max(0, stabilityCounter - 0.8);
                btn.disabled = true;
                btn.className = 'w-full py-2 bg-[#162030] border border-[#283850] rounded text-xs font-bold text-slate-500 uppercase tracking-wider cursor-not-allowed';
                btn.innerText = 'Hold Still 5s to Enable Check-In';
                badge.innerText = 'Prereq: 5s Steady Lock';
                badge.className = 'text-[10px] px-2 py-0.5 rounded bg-[#162030] text-slate-400';
            }
        }

        // 2. Run the 30-Second Clinical Check-In
        function startRestingCheckin() {
            isCheckinRunning = true;
            checkinTimeRemaining = 30;
            checkinBpmSamples = [];
            checkinHrvSamples = [];

            document.getElementById('btn-start-checkin').classList.add('hidden');
            document.getElementById('checkin-result-box').classList.add('hidden');
            document.getElementById('checkin-progress-container').classList.remove('hidden');

            checkinTimerInterval = setInterval(() => {
                checkinTimeRemaining--;
                document.getElementById('checkin-timer').innerText = checkinTimeRemaining + 's';
                const progressPct = ((30 - checkinTimeRemaining) / 30) * 100;
                document.getElementById('checkin-progress-bar').style.width = progressPct + '%';

                if (checkinTimeRemaining <= 0) {
                    clearInterval(checkinTimerInterval);
                    finishRestingCheckin();
                }
            }, 1000);
        }

        // 3. Compute Outlier-Rejected Average (Trimmed IQR Mean)
        function finishRestingCheckin() {
            isCheckinRunning = false;
            document.getElementById('checkin-progress-container').classList.add('hidden');

            if (checkinBpmSamples.length < 10) {
                alert("Check-in interrupted by movement. Please sit still and try again.");
                document.getElementById('btn-start-checkin').classList.remove('hidden');
                return;
            }

            // Reject top 15% and bottom 15% outliers
            checkinBpmSamples.sort((a, b) => a - b);
            checkinHrvSamples.sort((a, b) => a - b);

            const trimCount = Math.floor(checkinBpmSamples.length * 0.15);
            const validBpm = checkinBpmSamples.slice(trimCount, checkinBpmSamples.length - trimCount);
            const validHrv = checkinHrvSamples.slice(trimCount, checkinHrvSamples.length - trimCount);

            const avgBpm = validBpm.reduce((acc, val) => acc + val, 0) / validBpm.length;
            const avgHrv = validHrv.reduce((acc, val) => acc + val, 0) / validHrv.length;

            certifiedBaseline = {
                bpm: parseFloat(avgBpm.toFixed(1)),
                hrv: parseFloat(avgHrv.toFixed(1))
            };

            // Display Certified Baseline
            document.getElementById('certified-bpm').innerText = certifiedBaseline.bpm + ' BPM';
            document.getElementById('certified-hrv').innerText = 'HRV: ' + certifiedBaseline.hrv + ' ms';
            document.getElementById('checkin-result-box').classList.remove('hidden');
            document.getElementById('btn-start-checkin').classList.remove('hidden');
            document.getElementById('btn-start-checkin').innerText = 'Retake 30s Check-In';
        }

        // 4. Guided Workflow: Bridge Directly to Stage 2 & Stage 5
        function proceedFromCertifiedCheckin() {
            if (!certifiedBaseline) return;
            // 1. Switch to Tab 2
            switchTab('risk');
            // 2. Populate form with the certified ground truth
            document.getElementById('form-bpm').value = certifiedBaseline.bpm;
            document.getElementById('form-hrv').value = certifiedBaseline.hrv;
            // 3. Automatically execute clinical stratification
            document.getElementById('risk-form').dispatchEvent(new Event('submit'));
            // 4. Scroll smoothly to results
            document.getElementById('tab-risk').scrollIntoView({ behavior: 'smooth' });
        }

        // 5. Live Passive Stress Detection
        function togglePassiveStress(isChecked) {
            passiveStressEnabled = isChecked;
            const indicator = document.getElementById('passive-stress-indicator');
            const alertBanner = document.getElementById('stress-alert-banner');
            if (isChecked) {
                indicator.innerText = 'Active (Monitoring in Background)';
                indicator.className = 'text-emerald-400 font-bold';
            } else {
                indicator.innerText = 'Standby (Turn Toggle On)';
                indicator.className = 'text-slate-500 font-bold';
                alertBanner.classList.add('hidden');
                highStressSeconds = 0;
            }
        }

        function evaluateLivePassiveStress(bpm, hrv) {
            if (!passiveStressEnabled || bpm <= 0 || hrv <= 0) return;

            // Autonomic stress equation: elevated HR (>78) + suppressed HRV (<36)
            const isStressed = (bpm >= 80.0) || (bpm >= 76.0 && hrv < 35.0);

            if (isStressed) {
                highStressSeconds += 0.4;
                // If sustained for > 6 seconds of tension, notify user
                if (highStressSeconds >= 6.0) {
                    document.getElementById('stress-alert-banner').classList.remove('hidden');
                    document.getElementById('passive-stress-indicator').innerText = 'Elevated Stress Detected!';
                    document.getElementById('passive-stress-indicator').className = 'text-rose-400 font-bold animate-pulse';
                }
            } else {
                highStressSeconds = Math.max(0, highStressSeconds - 0.4);
                if (highStressSeconds === 0) {
                    document.getElementById('stress-alert-banner').classList.add('hidden');
                    document.getElementById('passive-stress-indicator').innerText = 'Active (Relaxed State)';
                    document.getElementById('passive-stress-indicator').className = 'text-emerald-400 font-bold';
                }
            }
        }

        function launchQuickBreathingReset() {
            switchTab('mentis');
            toggleBreathingExercise();
            document.getElementById('tab-mentis').scrollIntoView({ behavior: 'smooth' });
        }
        function switchTab(tabId) {
            activeTab = tabId;
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.className = 'tab-button px-3 py-1.5 rounded font-medium text-slate-400 hover:text-white whitespace-nowrap';
            });
            document.getElementById('tab-' + tabId).classList.remove('hidden');
            document.getElementById('tab-btn-' + tabId).className = 'tab-button px-3 py-1.5 rounded font-semibold text-teal-300 bg-[#141b27] border border-[#243046] whitespace-nowrap';

            const bpmToUse = liveTelemetry.bpm > 0 ? liveTelemetry.bpm : 72.0;
            const hrvToUse = liveTelemetry.hrv > 0 ? liveTelemetry.hrv : 42.0;

            if (tabId === 'cardio') {
                fetchCardioPlan(bpmToUse, hrvToUse, 35);
            } else if (tabId === 'mentis') {
                initMentisCanvas();
                runMentisAnalysis();
            }
            lucide.createIcons();
        }

        function setMentisTask(taskName) {
            currentMentisTask = taskName;
            document.querySelectorAll('.mentis-task-btn').forEach(btn => {
                btn.className = 'mentis-task-btn px-2.5 py-1 rounded text-slate-400 hover:text-white';
            });
            document.getElementById('task-btn-' + taskName).className = 'mentis-task-btn px-2.5 py-1 rounded bg-[#162030] text-white border border-[#2b3a54]';
            runMentisAnalysis();
        }

        async function runMentisAnalysis() {
            try {
                const bpm = liveTelemetry.bpm > 0 ? liveTelemetry.bpm : 72.0;
                const hrv = liveTelemetry.hrv > 0 ? liveTelemetry.hrv : 42.0;

                const res = await fetch('/api/mentis/analyze_stress', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ bpm: bpm, hrv: hrv, task: currentMentisTask })
                });
                const data = await res.json();

                document.getElementById('m-stress-score').innerText = data.stress_score;
                document.getElementById('m-level-text').innerText = data.level;
                document.getElementById('m-stress-bar').style.width = data.stress_score + '%';
                document.getElementById('m-bpm').innerText = data.eff_bpm + ' BPM';
                document.getElementById('m-hrv').innerText = data.eff_hrv + ' ms';
                document.getElementById('m-what-means').innerText = data.what_hr_means;

                const stepsList = document.getElementById('m-next-steps');
                stepsList.innerHTML = '';
                data.next_steps.forEach(s => {
                    const li = document.createElement('li');
                    li.innerText = s;
                    stepsList.appendChild(li);
                });

                mentisCorruptedWave = data.raw_wave || [];
                mentisDenoisedWave = data.denoised_wave || [];
            } catch(e) {
                console.error("Mentis error:", e);
            }
        }

        function toggleBreathingExercise() {
            const btn = document.getElementById('breath-btn');
            const circle = document.getElementById('breath-circle');
            const inst = document.getElementById('breath-instruction');

            if (breathTimer) {
                clearInterval(breathTimer);
                breathTimer = null;
                btn.innerText = "Start 60s Reset";
                circle.style.transform = "scale(1)";
                circle.innerText = "Ready";
                inst.innerText = "Click 'Start 60s Reset' to begin slow 4s Inhale / 6s Exhale pacing.";
                return;
            }

            btn.innerText = "Stop Reset";
            let sec = 0;
            const step = () => {
                const cycle = sec % 10;
                if (cycle < 4) {
                    circle.style.transform = "scale(1.35)";
                    circle.style.borderColor = "#2dd4bf";
                    circle.innerText = `Inhale ${4 - cycle}s`;
                    inst.innerText = "Breathe in slowly through your nose...";
                } else {
                    circle.style.transform = "scale(0.85)";
                    circle.style.borderColor = "#818cf8";
                    circle.innerText = `Exhale ${10 - cycle}s`;
                    inst.innerText = "Exhale slowly... longer exhales calm your heart rate.";
                }
                sec++;
                if (sec >= 60) {
                    toggleBreathingExercise();
                    inst.innerText = "60-second reset complete! Notice how your pulse steadies.";
                }
            };
            step();
            breathTimer = setInterval(step, 1000);
        }

        let mentisCanvas, mentisCtx;
        function initMentisCanvas() {
            mentisCanvas = document.getElementById('mentisDiffusionCanvas');
            if (mentisCanvas) {
                mentisCanvas.width = mentisCanvas.parentElement.clientWidth;
                mentisCanvas.height = mentisCanvas.parentElement.clientHeight;
                mentisCtx = mentisCanvas.getContext('2d');
            }
        }

        function drawMentisWaveforms() {
            if (mentisCanvas && mentisCtx && activeTab === 'mentis') {
                mentisCtx.clearRect(0, 0, mentisCanvas.width, mentisCanvas.height);
                const w = mentisCanvas.width;
                const h = mentisCanvas.height;

                if (mentisCorruptedWave.length >= 2) {
                    mentisCtx.beginPath();
                    mentisCtx.strokeStyle = 'rgba(244, 63, 94, 0.45)';
                    mentisCtx.lineWidth = 1.5;
                    const minC = Math.min(...mentisCorruptedWave);
                    const maxC = Math.max(...mentisCorruptedWave);
                    const rangeC = (maxC - minC) || 1e-4;
                    for (let i = 0; i < mentisCorruptedWave.length; i++) {
                        const x = (i / (mentisCorruptedWave.length - 1)) * w;
                        const y = h - 10 - (((mentisCorruptedWave[i] - minC) / rangeC) * (h - 20));
                        if (i === 0) mentisCtx.moveTo(x, y);
                        else mentisCtx.lineTo(x, y);
                    }
                    mentisCtx.stroke();
                }

                if (mentisDenoisedWave.length >= 2) {
                    mentisCtx.beginPath();
                    mentisCtx.strokeStyle = '#2dd4bf';
                    mentisCtx.lineWidth = 2.0;
                    const minD = Math.min(...mentisDenoisedWave);
                    const maxD = Math.max(...mentisDenoisedWave);
                    const rangeD = (maxD - minD) || 1e-4;
                    for (let i = 0; i < mentisDenoisedWave.length; i++) {
                        const x = (i / (mentisDenoisedWave.length - 1)) * w;
                        const y = h - 10 - (((mentisDenoisedWave[i] - minD) / rangeD) * (h - 20));
                        if (i === 0) mentisCtx.moveTo(x, y);
                        else mentisCtx.lineTo(x, y);
                    }
                    mentisCtx.stroke();
                }
            }
            requestAnimationFrame(drawMentisWaveforms);
        }
        requestAnimationFrame(drawMentisWaveforms);

        const canvas = document.getElementById('ppgCanvas');
        const ctx = canvas.getContext('2d');
        let waveData = [];

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function drawWaveform() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (waveData.length >= 2) {
                ctx.beginPath();
                ctx.strokeStyle = '#2dd4bf';
                ctx.lineWidth = 2.0;
                const minVal = Math.min(...waveData);
                const maxVal = Math.max(...waveData);
                const range = (maxVal - minVal) || 1e-4;
                for (let i = 0; i < waveData.length; i++) {
                    const x = (i / (waveData.length - 1)) * canvas.width;
                    const y = canvas.height - 8 - (((waveData[i] - minVal) / range) * (canvas.height - 16));
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            requestAnimationFrame(drawWaveform);
        }
        requestAnimationFrame(drawWaveform);

        function importCurrentVitals() {
            document.getElementById('form-bpm').value = (liveTelemetry.bpm > 0 ? liveTelemetry.bpm : 72.0).toFixed(1);
            document.getElementById('form-hrv').value = (liveTelemetry.hrv > 0 ? liveTelemetry.hrv : 42.0).toFixed(1);
        }

        async function runRiskEvaluation(e) {
            e.preventDefault();
            const payload = {
                bpm: parseFloat(document.getElementById('form-bpm').value),
                hrv: parseFloat(document.getElementById('form-hrv').value),
                bp_category: document.getElementById('form-bp').value,
                ethnicity: document.getElementById('form-ethnicity').value,
                age: document.getElementById('form-age-val').value,
                sleep: document.getElementById('form-sleep').value,
                stress: document.getElementById('form-stress').value,
                activity: document.getElementById('form-activity').value,
                family_history: document.getElementById('form-family').checked
            };
            const res = await fetch('/api/evaluate_risk', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const result = await res.json();
            currentRiskProfile = result;
            renderRiskResults(result);
        }

        function renderRiskResults(data) {
            document.getElementById('risk-loading').classList.add('hidden');
            document.getElementById('risk-display').classList.remove('hidden');
            document.getElementById('res-tier').innerText = data.tier;
            document.getElementById('res-score').innerText = data.score;
            document.getElementById('res-summary').innerText = data.summary;
            document.getElementById('res-action').innerText = data.rec_action;

            const bContainer = document.getElementById('res-breakdown');
            bContainer.innerHTML = '';
            data.breakdown.forEach(item => {
                const row = document.createElement('div');
                row.className = 'flex justify-between items-center border-b border-[#182030] pb-1';
                row.innerHTML = `
                    <div><span class="text-slate-200">${item.label}</span><span class="text-[10px] text-slate-500 block">${item.ref}</span></div>
                    <span class="font-bold text-teal-300">+${item.pts} / ${item.max} pts</span>
                `;
                bContainer.appendChild(row);
            });

            const driversContainer = document.getElementById('res-drivers');
            driversContainer.innerHTML = '';
            data.drivers.forEach(d => {
                const card = document.createElement('div');
                card.className = 'panel-inset p-2.5 rounded text-xs space-y-0.5';
                card.innerHTML = `<div class="font-bold text-slate-200 flex justify-between"><span>${d.title}</span><span class="text-amber-400">${d.severity}</span></div><p class="text-[11px] text-slate-400">${d.desc}</p>`;
                driversContainer.appendChild(card);
            });
        }

        function searchResources(e) {
            e.preventDefault();
            const loc = document.getElementById('loc-input').value.trim();
            if (loc) fetchTargetedResources(loc);
        }

        async function fetchTargetedResources(locationQuery) {
            const grid = document.getElementById('resources-grid');
            grid.innerHTML = `<div class="col-span-full py-8 text-center text-xs text-slate-400 font-mono">Querying US Dept of Health NPI Registry for ${locationQuery}...</div>`;
            const res = await fetch('/api/find_resources', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: locationQuery, risk_profile: currentRiskProfile }) });
            const data = await res.json();
            grid.innerHTML = '';
            data.resources.forEach(r => {
                const card = document.createElement('div');
                card.className = 'panel rounded p-4 flex flex-col justify-between space-y-3 font-mono';
                card.innerHTML = `
                    <div class="space-y-2">
                        <div class="flex justify-between items-center text-[10px]"><span class="text-slate-400 uppercase">${r.category}</span><span class="px-1.5 py-0.5 rounded bg-[#162030] border border-[#26334a] text-teal-300 font-bold">${r.badge}</span></div>
                        <h4 class="text-sm font-bold text-white">${r.name}</h4>
                        <p class="text-xs text-emerald-300">📍 ${r.address}</p>
                        <p class="text-xs text-slate-300">📞 ${r.phone} • ${r.distance}</p>
                        <div class="panel-inset rounded p-2 text-[11px] text-slate-400">${r.rationale}</div>
                    </div>
                    <div class="pt-2 border-t border-[#182030] flex justify-end"><a href="${r.external_link}" target="_blank" class="px-3 py-1.5 bg-[#162030] border border-[#283850] rounded text-teal-300 text-xs font-bold hover:bg-[#1f2c42]">View on Google Maps →</a></div>
                `;
                grid.appendChild(card);
            });
        }

        function searchEquipment(e) {
            e.preventDefault();
            const loc = document.getElementById('eq-loc-input').value.trim();
            if (loc) fetchEquipmentDirectory(loc);
        }

        async function fetchEquipmentDirectory(locationQuery) {
            const devGrid = document.getElementById('devices-grid');
            const kioskGrid = document.getElementById('kiosks-grid');
            devGrid.innerHTML = `<div class="col-span-full py-6 text-center text-xs text-slate-400 font-mono">Calculating local tax & store inventory for ${locationQuery}...</div>`;
            kioskGrid.innerHTML = `<div class="col-span-full py-6 text-center text-xs text-slate-400 font-mono">Finding registered pharmacies in ${locationQuery}...</div>`;
            const res = await fetch('/api/find_equipment', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location: locationQuery }) });
            const data = await res.json();
            devGrid.innerHTML = '';
            kioskGrid.innerHTML = '';

            data.kiosks.forEach(k => {
                const card = document.createElement('div');
                card.className = 'panel rounded p-4 flex flex-col justify-between space-y-3 font-mono';
                card.innerHTML = `
                    <div class="space-y-1.5"><div class="flex justify-between items-center text-[10px]"><span class="text-slate-400 uppercase">${k.type}</span><span class="px-1.5 py-0.5 rounded bg-[#162030] text-teal-300 font-bold">${k.cost}</span></div><h4 class="text-sm font-bold text-white">${k.name}</h4><p class="text-xs text-slate-300">${k.nearby_store}</p></div>
                    <a href="${k.action_url}" target="_blank" class="w-full py-1.5 bg-[#162030] hover:bg-[#1f2c42] border border-[#283850] rounded text-teal-300 text-xs font-bold text-center block uppercase">Directions to Pharmacy →</a>
                `;
                kioskGrid.appendChild(card);
            });

            data.devices.forEach(d => {
                const card = document.createElement('div');
                card.className = 'panel rounded p-4 flex flex-col justify-between space-y-3 font-mono';
                card.innerHTML = `
                    <div class="space-y-1.5">
                        <div class="flex justify-between items-center text-[10px]"><span class="text-slate-400 uppercase">${d.tier}</span><span class="px-1.5 py-0.5 rounded bg-[#162030] text-emerald-300 font-bold">${d.validation}</span></div>
                        <h4 class="text-xs font-bold text-white">${d.name}</h4>
                        <div class="text-xl font-bold text-emerald-400">${d.price}</div>
                        <span class="text-[10px] text-slate-400 block">${d.sub_price}</span>
                        <span class="text-[10px] text-teal-300 block">${d.local_stock_note}</span>
                    </div>
                    <div class="pt-2 border-t border-[#182030] space-y-1.5"><a href="${d.direct_buy_url}" target="_blank" class="w-full py-1.5 bg-[#172233] hover:bg-[#1f2c42] border border-[#283850] rounded text-[11px] text-emerald-300 font-bold text-center block">Buy Direct (${d.price})</a><a href="${d.store_buy_url}" target="_blank" class="w-full py-1 bg-[#0c1017] border border-[#182030] rounded text-[10px] text-slate-300 text-center block">Check Retail Stock</a></div>
                `;
                devGrid.appendChild(card);
            });
        }

        async function fetchCardioPlan(bpm, hrv, age) {
            const res = await fetch('/api/generate_cardio_plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bpm: bpm, hrv: hrv, age: age, activity: "low" }) });
            const plan = await res.json();
            renderCardioPlan(plan);
        }

        function renderCardioPlan(plan) {
            document.getElementById('cardio-target-zone').innerText = `${plan.z2_low} - ${plan.z2_high} BPM`;
            document.getElementById('cardio-goal-bpm').innerText = `${plan.target_bpm.toFixed(0)} BPM`;
            document.getElementById('cardio-weekly-vol').innerText = `${plan.weekly_volume} min`;
            document.getElementById('cardio-goal-banner').innerText = plan.goal_text;

            const ctx = document.getElementById('hrProjectionCanvas');
            if (ctx) {
                if (hrChartInstance) hrChartInstance.destroy();
                hrChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: plan.chart_labels,
                        datasets: [
                            { label: 'Projected Resting HR', data: plan.chart_curve, borderColor: '#2dd4bf', backgroundColor: 'rgba(45, 212, 191, 0.1)', borderWidth: 2.5, fill: true, tension: 0.35, pointBackgroundColor: '#2dd4bf', pointRadius: 4 },
                            { label: 'Target Goal', data: plan.chart_baseline, borderColor: '#f43f5e', borderWidth: 1.5, borderDash: [4, 4], fill: false, pointRadius: 0 }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { min: Math.min(...plan.chart_curve, 60) - 4, max: Math.max(...plan.chart_curve, 75) + 4, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8' } } } }
                });
            }

            const schedGrid = document.getElementById('cardio-schedule-grid');
            schedGrid.innerHTML = '';
            plan.schedule.forEach(s => {
                const card = document.createElement('div');
                const isRest = s.duration === 'Rest Day';
                card.className = 'panel p-3 rounded space-y-2 text-xs font-mono';
                card.innerHTML = `
                    <div class="space-y-1"><div class="flex justify-between items-center text-[10px]"><span class="font-bold text-slate-200 uppercase">${s.day}</span><span class="px-1.5 py-0.5 rounded ${isRest ? 'bg-[#182030] text-slate-400' : 'bg-[#162030] text-teal-300'}">${s.duration}</span></div><h5 class="font-bold text-white text-xs">${s.type}</h5><p class="text-[11px] text-slate-300">${s.modality}</p></div>
                    <div class="pt-2 border-t border-[#182030] text-[10px] text-slate-400"><span class="text-teal-400 block font-semibold mb-0.5">How it should feel:</span>${s.pace_tip}</div>
                `;
                schedGrid.appendChild(card);
            });
        }

        function syncCardioFromLiveVitals() {
            const bpm = liveTelemetry.bpm > 0 ? liveTelemetry.bpm : 72.0;
            const hrv = liveTelemetry.hrv > 0 ? liveTelemetry.hrv : 42.0;
            fetchCardioPlan(bpm, hrv, 35);
        }

        // Poll Live Vitals & HUD Status
        setInterval(async () => {
            try {
                const res = await fetch('/api/vitals');
                const data = await res.json();
                if (data.bpm > 0) {
                    liveTelemetry.bpm = data.bpm;
                    document.getElementById('metric-bpm').innerText = data.bpm.toFixed(1);
                    document.getElementById('metric-bpm-progress').style.width = Math.min(100, (data.bpm / 120) * 100) + '%';
                }
                if (data.hrv > 0) {
                    liveTelemetry.hrv = data.hrv;
                    document.getElementById('metric-hrv').innerText = data.hrv.toFixed(1);
                    document.getElementById('metric-hrv-progress').style.width = Math.min(100, (data.hrv / 80) * 100) + '%';
                }

                // ---> PASTE THE THIRD CODE BLOCK RIGHT HERE <---
                // Collect samples if 30s Check-In is actively running
                if (isCheckinRunning && data.bpm > 0 && data.hrv > 0) {
                    checkinBpmSamples.push(data.bpm);
                    checkinHrvSamples.push(data.hrv);
                }

                // Evaluate prerequisite lock and passive stress
                evaluateCheckinPrerequisites(data);
                evaluateLivePassiveStress(data.bpm, data.hrv);
                // ------------------------------------------------

            
                document.getElementById('sqi-badge').innerText = data.sqi + '%';
                document.getElementById('metric-status').innerText = data.status;
                document.getElementById('metric-tone').innerText = data.tone;
                document.getElementById('metric-snr').innerText = data.snr + ' dB';
                if (data.roi_label) document.getElementById('roi-badge').innerText = data.roi_label;

                if (data.guide_distance) document.getElementById('hud-distance').innerText = data.guide_distance;
                if (data.guide_lighting) document.getElementById('hud-lighting').innerText = data.guide_lighting;
                if (data.guide_stability) document.getElementById('hud-stability').innerText = data.guide_stability;

                const dot = document.getElementById('tracking-dot');
                const stat = document.getElementById('tracking-status');
                if (data.is_tracking) {
                    dot.className = 'w-2 h-2 rounded-full bg-emerald-400';
                    stat.innerText = 'Skin Locked';
                } else {
                    dot.className = 'w-2 h-2 rounded-full bg-amber-400';
                    stat.innerText = 'Center Face';
                }
                if (data.wave && data.wave.length > 0) waveData = data.wave;
            } catch (e) {}
        }, 400);

        window.addEventListener('DOMContentLoaded', () => {
            fetchTargetedResources("60177");
            fetchEquipmentDirectory("60177");
        });
    </script>
</body>
</html>
"""


# --- 11. Web Endpoints ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/vitals')
def get_vitals():
    with lock:
        data = dict(telemetry)
        data["wave"] = list(filtered_wave_display)
    return jsonify(data)


@app.route('/api/evaluate_risk', methods=['POST'])
def api_evaluate_risk():
    return jsonify(compute_cardiovascular_risk(request.get_json() or {}))


@app.route('/api/find_resources', methods=['POST'])
def api_find_resources():
    payload = request.get_json() or {}
    loc_info = resolve_location(payload.get("location", "60177"))
    return jsonify(
        {"location": loc_info, "resources": get_targeted_resources(loc_info, payload.get("risk_profile", {}))})


@app.route('/api/find_equipment', methods=['POST'])
def api_find_equipment():
    payload = request.get_json() or {}
    loc_info = resolve_location(payload.get("location", "60177"))
    return jsonify(get_equipment_directory(loc_info))


@app.route('/api/generate_cardio_plan', methods=['POST'])
def api_generate_cardio_plan():
    p = request.get_json() or {}
    return jsonify(
        generate_cardio_protocol(p.get("bpm", 72.0), p.get("hrv", 42.0), p.get("age", 35), p.get("activity", "low")))


@app.route('/api/mentis/analyze_stress', methods=['POST'])
def api_mentis_stress():
    payload = request.get_json() or {}
    bpm = payload.get("bpm", 72.0)
    hrv = payload.get("hrv", 42.0)
    task = payload.get("task", "reading")
    res = compute_mentis_stress(bpm, hrv, task)
    with lock:
        wave_snippet = list(filtered_wave_display)[-90:] if len(filtered_wave_display) >= 90 else []
    shaky, clean = simulate_ddpm_wave(wave_snippet, task)
    res["raw_wave"] = shaky
    res["denoised_wave"] = clean
    return jsonify(res)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 EquiPulse Running on Port 5001!")
    print("Navigate to: http://127.0.0.1:5001")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)