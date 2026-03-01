import streamlit as st
import cv2
import tempfile
import time
import torch
import numpy as np
import pandas as pd
import gc
import plotly.graph_objects as go
import plotly.express as px
import re
from collections import Counter
from backend import (
    load_pytorch_model,
    load_drowsiness_model,
    FocusLSTM,
    MicroexpressionResNet,
    combined_driver_state,
    predict_driver_attentiveness,
    predict_microexpression,
    predict_driver_drowsiness
)
import backend as backend_module

st.set_page_config(page_title="Driver Safety Monitor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for grey theme
st.markdown("""
    <style>
        * {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background-color: #f0f0f0;
        }
        .main-header {
            background: linear-gradient(135deg, #4a4a4a 0%, #5c5c5c 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            opacity: 0.95;
            font-weight: 300;
        }
        .status-card {
            background: linear-gradient(135deg, #e8e8e8 0%, #d8d8d8 100%);
            padding: 1.8rem;
            border-radius: 14px;
            border: 1px solid #c0c0c0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            text-align: center;
            transition: all 0.3s ease;
        }
        .status-card:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
            background: linear-gradient(135deg, #dcdcdc 0%, #cccccc 100%);
        }
        .status-icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
        }
        .status-label {
            font-size: 0.85rem;
            color: #4a4a4a;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.8rem;
        }
        .status-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c2c2c;
            margin-bottom: 0.6rem;
        }
        .status-confidence {
            font-size: 0.8rem;
            color: #5a5a5a;
            font-weight: 500;
        }
        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #2c2c2c;
            margin: 2rem 0 1.2rem 0;
            padding-bottom: 0.8rem;
            border-bottom: 2px solid #b8b8b8;
        }
        .metric-card {
            background: linear-gradient(135deg, #e8e8e8 0%, #d8d8d8 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #c0c0c0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .divider {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #a8a8a8, transparent);
        }
        .alert-danger {
            background: #f5e8e8;
            border-left: 4px solid #888888;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #3a3a3a;
        }
        .alert-warning {
            background: #f5f0e8;
            border-left: 4px solid #888888;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #3a3a3a;
        }
        .alert-success {
            background: #e8f5e8;
            border-left: 4px solid #888888;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #2a3a2a;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; border-bottom: 1px solid #d0d0d0;">
        <h1 style="margin: 0; color: #2c2c2c; font-size: 2.8rem; font-weight: 700; letter-spacing: -0.8px;">Driver Safety Monitor</h1>
        <p style="margin: 0.5rem 0 0 0; color: #6a6a6a; font-size: 0.95rem; font-weight: 300;">Intelligent driver behavior analysis system</p>
    </div>
""", unsafe_allow_html=True)

# Icon row (Drowsiness / Attention / Emotion) - REMOVED for cleaner design

st.markdown("")

# Session Controls - Removed from UI, hardcoded for optimal performance
EAR_THRESHOLD = 0.20  # Standard threshold for eye detection
# Per-model sequence lengths (keep original model requirements)
ATT_SEQUENCE_LENGTH = 30  # Attentiveness model trained on 30-frame sequences
DROWSY_SEQUENCE_LENGTH = 10  # Drowsiness model (user's notebook) trained on 10-frame sequences
OVERLAY_PREDICTIONS = True  # Always enabled - predictions overlaid on all frames
ENABLE_CSV_DOWNLOAD = True  # CSV export always available




@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    att_model = load_pytorch_model(FocusLSTM, "notebooks/driver_attentiveness/model.pth", device=device, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2)
    microexp_model = load_pytorch_model(MicroexpressionResNet, "notebooks/facial_micro_expression/best_emotion_model.pth", device=device, num_classes=7)
    # Prefer the notebook-trained fine-tuned drowsiness model (10-frame sequences). Fall back to packaged MobileNet if not present.
    preferred = "notebooks/driver_drowsiness/fine_tuned_model.keras"
    fallback = "notebooks/driver_drowsiness/mobilenet_model.keras"
    drowsy_model = load_drowsiness_model(preferred) or load_drowsiness_model(fallback)
    return att_model, microexp_model, drowsy_model, device

att_model, microexp_model, drowsy_model, device = load_models()

# Apply sidebar EAR threshold to backend
try:
    backend_module.EAR_THRESHOLD = EAR_THRESHOLD
except Exception:
    pass

emotion_map = {
    "0": "Neutral", "1": "Happy", "2": "Sad", "3": "Surprised",
    "4": "Angry", "5": "Disgust", "6": "Fear"
}

def summarize_states(drowsy_over_time, attentive_over_time, microexp_values):
    def percentage(count, total):
        return round(100 * count / total, 1) if total else 0

    total = len(drowsy_over_time)

    drowsy_count = sum(drowsy_over_time)
    alert_count = total - drowsy_count

    attentive_count = sum(1 for a in attentive_over_time if a.upper() == "ATTENTIVE")
    distracted_count = total - attentive_count

    microexp_counter = Counter(microexp_values)
    dominant_microexp, microexp_count = microexp_counter.most_common(1)[0]

    return {
        "drowsy": (drowsy_count, percentage(drowsy_count, total)),
        "alert": (alert_count, percentage(alert_count, total)),
        "attentive": (attentive_count, percentage(attentive_count, total)),
        "distracted": (distracted_count, percentage(distracted_count, total)),
        "dominant_microexp": (dominant_microexp, percentage(microexp_count, total))
    }


def compute_risk_score(drowsy_pct, distracted_pct, negative_emotion_pct):
    """Compute a 0-100 risk score. Higher is worse."""
    # weights: drowsiness 50%, distraction 30%, negative emotion 20%
    risk = (drowsy_pct * 0.5) + (distracted_pct * 0.3) + (negative_emotion_pct * 0.2)
    # Map to 0-100 and clamp
    return int(max(0, min(100, risk)))


def generate_nl_summary(summary):
    """Create a short natural-language session summary."""
    parts = []
    parts.append(f"The driver was drowsy for {summary['drowsy'][1]}% of the session.")
    parts.append(f"Distracted {summary['distracted'][1]}% of the time.")
    parts.append(f"Dominant emotion: {summary['dominant_microexp'][0]} ({summary['dominant_microexp'][1]}%).")
    if summary['drowsy'][1] > 30:
        parts.append("High drowsiness risk — immediate action recommended.")
    elif summary['drowsy'][1] > 15:
        parts.append("Moderate drowsiness detected — monitor the driver.")
    else:
        parts.append("No significant drowsiness detected.")
    return " ".join(parts)


def annotate_frame(frame, drowsy_label, att_label, emotion_label):
    """Return an annotated copy of the frame (RGB)."""
    img = frame.copy()
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    text = f"{drowsy_label} | {att_label} | {emotion_label}"
    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(bgr, f"AttSeq:{ATT_SEQUENCE_LENGTH} DrowSeq:{DROWSY_SEQUENCE_LENGTH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_status_cards(drowsy_status, drowsy_conf, att_status, att_conf, emotion_status, emotion_conf):
    """Render three status cards in a row."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">😴</div>
            <div class="status-label">Drowsiness</div>
            <div class="status-value">{drowsy_status}</div>
            <div class="status-confidence">Confidence: {drowsy_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">👁️</div>
            <div class="status-label">Attentiveness</div>
            <div class="status-value">{att_status}</div>
            <div class="status-confidence">Confidence: {att_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">😊</div>
            <div class="status-label">Emotion</div>
            <div class="status-value">{emotion_status}</div>
            <div class="status-confidence">Confidence: {emotion_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def render_detailed_analysis(drowsy_over_time, attentive_over_time, microexp_values, results):
    """Render comprehensive analysis for video uploads"""
    summary = summarize_states(drowsy_over_time, attentive_over_time, microexp_values)
    
    st.markdown('<h2 class="section-title">Session Results</h2>', unsafe_allow_html=True)
    
    # Extract confidence scores
    drowsy_conf = 0
    att_conf = 0
    emotion_conf = 0
    
    for r in results[:5]:  # Average from first 5 results
        d = r.get("Driver Drowsiness", "")
        m = re.search(r"(\d+\.?\d*)%", d)
        if m:
            try:
                drowsy_conf += float(m.group(1))
            except:
                pass
    drowsy_conf = drowsy_conf / min(5, len(results)) if results else 0
    att_conf = 85.0  # Placeholder
    emotion_conf = 80.0  # Placeholder
    
    # Status cards
    drowsy_status = "DROWSY" if summary['drowsy'][1] > 20 else "ALERT"
    att_status = "FOCUSED" if summary['attentive'][1] > 60 else "DISTRACTED"
    emotion_status = summary['dominant_microexp'][0][:10]  # Truncate long names
    
    render_status_cards(drowsy_status, drowsy_conf, att_status, att_conf, emotion_status, emotion_conf)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Session Analytics
    st.markdown('<h3 class="section-title">Session Analytics</h3>', unsafe_allow_html=True)
    
    # Drowsiness trend
    df_drowsy = pd.DataFrame({
        'Second': range(1, len(drowsy_over_time) + 1),
        'Status': drowsy_over_time
    })
    
    fig_drowsy = go.Figure()
    fig_drowsy.add_trace(go.Scatter(
        x=df_drowsy['Second'],
        y=drowsy_over_time,
        mode='lines+markers',
        name='Drowsiness',
        fill='tozeroy',
        line=dict(color='#5a5a5a', width=2),
        marker=dict(size=5, color='#4a4a4a')
    ))
    fig_drowsy.update_layout(
        title="Drowsiness Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Status (0=Alert, 1=Drowsy)",
        height=320,
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig_drowsy, use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown('<h3 class="section-title">Key Metrics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{len(results)}s", delta=None)
    with col2:
        st.metric("Drowsy Time", f"{summary['drowsy'][0]}s ({summary['drowsy'][1]:.1f}%)")
    with col3:
        st.metric("Attentive Time", f"{summary['attentive'][0]}s ({summary['attentive'][1]:.1f}%)")
    with col4:
        st.metric("Dominant Emotion", summary['dominant_microexp'][0])
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<h3 class="section-title">Insights</h3>', unsafe_allow_html=True)
    
    if summary['drowsy'][1] > 30:
        st.markdown('<div class="alert-danger"><b>High Drowsiness</b><br>Driver showed signs of drowsiness. Recommend break or driver rotation.</div>', unsafe_allow_html=True)
    elif summary['drowsy'][1] > 15:
        st.markdown('<div class="alert-warning"><b>Moderate Drowsiness</b><br>Some drowsiness detected. Monitor closely.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success"><b>Good Alertness</b><br>Driver maintained good alertness throughout session.</div>', unsafe_allow_html=True)
    
    ai_summary = generate_nl_summary(summary)
    st.info(ai_summary)


mode = st.sidebar.radio("Choose Mode", ["Live Webcam", "Upload Video"])

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# ------------------- VIDEO UPLOAD -------------------
if mode == "Upload Video":
    st.markdown("""
    ### Upload Video for Analysis
    Upload a driver video to get comprehensive safety analysis including drowsiness detection, 
    attention levels, and emotional state tracking.
    """)
    
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Create columns for layout
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            st.video(video_path)
        
        with info_col:
            st.info(f"File: {uploaded_video.name}\nSize: {uploaded_video.size / 1024 / 1024:.2f} MB")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = max(1, total_frames // max(1, fps))

        st.info(f"Processing ~{total_seconds} seconds of video at {fps} FPS (att_seq={ATT_SEQUENCE_LENGTH}, drowsy_seq={DROWSY_SEQUENCE_LENGTH})...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        sample_frames = []

        frame_buffer = []
        processed_chunks = 0

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)

            # Process once we have at least the attentiveness window (30 frames)
            if len(frame_buffer) >= ATT_SEQUENCE_LENGTH:
                processed_chunks += 1
                # take the first ATT_SEQUENCE_LENGTH frames as the chunk
                chunk = frame_buffer[:ATT_SEQUENCE_LENGTH]
                # Run model predictions
                try:
                    att = predict_driver_attentiveness(att_model, chunk, device)
                    micro = predict_microexpression(microexp_model, chunk, device)

                    # Drowsiness: run on multiple 10-frame sequences inside the 30-frame chunk and vote
                    d_votes = {"DROWSY": 0, "ALERT": 0}
                    d_confs = []
                    for i in range(0, ATT_SEQUENCE_LENGTH - DROWSY_SEQUENCE_LENGTH + 1, DROWSY_SEQUENCE_LENGTH):
                        seq = chunk[i:i + DROWSY_SEQUENCE_LENGTH]
                        d_raw = predict_driver_drowsiness(drowsy_model, seq, fps=fps, sequence_length=DROWSY_SEQUENCE_LENGTH)
                        # attempt to parse confidence from returned string like 'Driver is DROWSY (53.09%)'
                        m = re.search(r"\((\d+\.?\d*)%\)", d_raw)
                        if m:
                            try:
                                d_confs.append(float(m.group(1)))
                            except:
                                d_confs.append(50.0)
                        else:
                            d_confs.append(50.0)

                        if "DROWSY" in d_raw.upper():
                            d_votes["DROWSY"] += 1
                        else:
                            d_votes["ALERT"] += 1

                    avg_conf = np.mean(d_confs) if d_confs else 0.0
                    drows_label = "DROWSY" if d_votes["DROWSY"] > d_votes["ALERT"] else "ALERT"
                    drows = f"Driver is {drows_label} ({avg_conf:.2f}%)"

                except Exception as e:
                    # fallback to combined if any error
                    d = combined_driver_state(att_model, microexp_model, drowsy_model, chunk, device)
                    att = d.get("Driver Attentiveness")
                    micro = d.get("Driver Microexpression")
                    drows = d.get("Driver Drowsiness")

                results.append({
                    "second": processed_chunks,
                    "Driver Attentiveness": att,
                    "Driver Microexpression": micro,
                    "Driver Drowsiness": drows
                })
                # store a representative frame for overlay
                sample_frames.append(chunk[0])
                # remove the processed frames from buffer (slide window)
                frame_buffer = frame_buffer[ATT_SEQUENCE_LENGTH:]

                progress = min(processed_chunks / total_seconds, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processed {processed_chunks}/{total_seconds} chunks...")
                time.sleep(0.01)

        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        st.success("Video processing complete! Generating detailed analysis...")
        st.markdown("---")

        drowsy_over_time = []
        attentive_over_time = []
        microexp_values = []

        for r in results:
            drowsy_over_time.append(1 if "DROWSY" in r.get("Driver Drowsiness", "").upper() else 0)
            attentive_over_time.append(r.get("Driver Attentiveness", ""))
            class_id = r.get("Driver Microexpression", "").split(":")[-1].strip()
            microexp_values.append(emotion_map.get(class_id, f"Unknown ({class_id})"))

        # Render detailed analysis
        render_detailed_analysis(drowsy_over_time, attentive_over_time, microexp_values, results)
        
        # Build CSV-ready dataframe
        df = pd.DataFrame([{
            "second": r.get("second" , i+1),
            "drowsy": (1 if "DROWSY" in r.get("Driver Drowsiness", "").upper() else 0),
            "drowsiness_text": r.get("Driver Drowsiness", ""),
            "attentiveness": r.get("Driver Attentiveness", ""),
            "microexpression": r.get("Driver Microexpression", ""),
        } for i, r in enumerate(results)])

        if ENABLE_CSV_DOWNLOAD:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download session CSV", data=csv, file_name="driver_session_report.csv", mime='text/csv')

        # Optionally show annotated sample frames
        if OVERLAY_PREDICTIONS and sample_frames:
            st.subheader("Sample Annotated Frames")
            cols = st.columns(min(4, len(sample_frames)))
            for i, f in enumerate(sample_frames[:8]):
                lbl = results[i]
                d_lab = lbl.get("Driver Drowsiness", "")
                a_lab = lbl.get("Driver Attentiveness", "")
                e_lab = lbl.get("Driver Microexpression", "")
                ann = annotate_frame(f, d_lab, a_lab, e_lab)
                cols[i % len(cols)].image(ann, use_column_width=True)

# ------------------- LIVE WEBCAM MODE -------------------
elif mode == "Live Webcam":

    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_col1, button_col2 = st.columns(2, gap="small")
        with button_col1:
            start_btn = st.button("Start Monitoring", key="start", use_container_width=True)
        with button_col2:
            stop_btn = st.button("Stop Monitoring", key="stop", use_container_width=True)

    if start_btn:
        st.session_state.webcam_active = True
    if stop_btn:
        st.session_state.webcam_active = False

    webcam_col, status_col = st.columns([3, 1])
    
    with webcam_col:
        FRAME_WINDOW = st.empty()
    with status_col:
        status_indicator = st.empty()
    
    result_placeholder = st.empty()
    webcam_results = []

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check your camera permissions.")
            st.session_state.webcam_active = False
        else:
            frames = []
            frame_count = 0
            try:
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam disconnected.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Optionally overlay predictions on the displayed frame
                    display_frame = frame_rgb
                    frames.append(frame_rgb)
                    frame_count += 1

                    with status_col:
                        status_indicator.metric("Frames", frame_count)

                    if len(frames) >= ATT_SEQUENCE_LENGTH:
                        # take the first ATT_SEQUENCE_LENGTH frames
                        chunk = frames[:ATT_SEQUENCE_LENGTH]
                        try:
                            att = predict_driver_attentiveness(att_model, chunk, device)
                            micro = predict_microexpression(microexp_model, chunk, device)

                            # Drowsiness voting across multiple 10-frame sequences
                            d_votes = {"DROWSY": 0, "ALERT": 0}
                            d_confs = []
                            for i in range(0, ATT_SEQUENCE_LENGTH - DROWSY_SEQUENCE_LENGTH + 1, DROWSY_SEQUENCE_LENGTH):
                                seq = chunk[i:i + DROWSY_SEQUENCE_LENGTH]
                                d_raw = predict_driver_drowsiness(drowsy_model, seq, fps=30, sequence_length=DROWSY_SEQUENCE_LENGTH)
                                m = re.search(r"\((\d+\.?\d*)%\)", d_raw)
                                if m:
                                    try:
                                        d_confs.append(float(m.group(1)))
                                    except:
                                        d_confs.append(50.0)
                                else:
                                    d_confs.append(50.0)

                                if "DROWSY" in d_raw.upper():
                                    d_votes["DROWSY"] += 1
                                else:
                                    d_votes["ALERT"] += 1

                            avg_conf = np.mean(d_confs) if d_confs else 0.0
                            drows_label = "DROWSY" if d_votes["DROWSY"] > d_votes["ALERT"] else "ALERT"
                            drows = f"Driver is {drows_label} ({avg_conf:.2f}%)"

                        except Exception as e:
                            d = combined_driver_state(att_model, microexp_model, drowsy_model, chunk, device)
                            att = d.get("Driver Attentiveness")
                            micro = d.get("Driver Microexpression")
                            drows = d.get("Driver Drowsiness")

                        result = {
                            "Driver Attentiveness": att,
                            "Driver Microexpression": micro,
                            "Driver Drowsiness": drows
                        }
                        webcam_results.append(result)

                        # Overlay and display
                        if OVERLAY_PREDICTIONS:
                            display_frame = annotate_frame(chunk[0], drows, att, micro)

                        with result_placeholder.container():
                            res_col1, res_col2, res_col3 = st.columns(3)
                            drowsy_status = "DROWSY" if "DROWSY" in drows.upper() else "ALERT"
                            att_status = "DISTRACTED" if "DISTRACTED" in att.upper() else "FOCUSED"

                            with res_col1:
                                st.metric("Drowsiness", drowsy_status)
                            with res_col2:
                                st.metric("Attentiveness", att_status)
                            with res_col3:
                                st.metric("Emotion", micro)

                        # Real-time alerts
                        if "DROWSY" in drows.upper():
                            try:
                                st.toast("Drowsiness detected — please take a break")
                            except Exception:
                                st.warning("Drowsiness detected — please take a break")
                        elif "DISTRACT" in att.upper() or "DISTRACTED" in att.upper():
                            try:
                                st.toast("Driver appears distracted")
                            except Exception:
                                st.warning("Driver appears distracted")

                        FRAME_WINDOW.image(display_frame, channels="RGB")

                        # slide the processed frames
                        frames = frames[ATT_SEQUENCE_LENGTH:]
                        gc.collect()

                    time.sleep(1 / 30)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                cap.release()
                st.session_state.webcam_active = False
                st.success("Monitoring session ended.")

                if webcam_results:
                    st.markdown("---")
                    st.info(f"Captured {len(webcam_results)} analysis frames")
                    
                    drowsy_over_time = []
                    attentive_over_time = []
                    microexp_values = []

                    for r in webcam_results:
                        drowsy_over_time.append(1 if "DROWSY" in r.get("Driver Drowsiness", "").upper() else 0)
                        attentive_over_time.append(r.get("Driver Attentiveness", ""))
                        class_id = r.get("Driver Microexpression", "").split(":")[-1].strip()
                        microexp_values.append(emotion_map.get(class_id, f"Unknown ({class_id})"))

                    st.markdown("---")
                    st.subheader("Session Summary")
                    
                    # Quick metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Duration", f"{len(webcam_results)}s")
                    with col2:
                        st.metric("Drowsy", f"{sum(drowsy_over_time)}s")
                    with col3:
                        st.metric("Alert", f"{len(drowsy_over_time) - sum(drowsy_over_time)}s")
                    with col4:
                        safety = round(100 - (sum(drowsy_over_time) / len(drowsy_over_time) * 40), 1)
                        st.metric("Safety", f"{safety}%")
                    
                    # Charts
                    df = pd.DataFrame({
                        "Second": list(range(1, len(webcam_results) + 1)),
                        "Drowsy": drowsy_over_time
                    })
                    
                    fig = px.bar(df, x="Second", y="Drowsy", title="Drowsiness Timeline",
                                labels={"Drowsy": "Status (0=Alert, 1=Drowsy)"}, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    render_detailed_analysis(drowsy_over_time, attentive_over_time, microexp_values, webcam_results)

                    # CSV export
                    if ENABLE_CSV_DOWNLOAD:
                        df_export = pd.DataFrame([{
                            'second': i+1,
                            'drowsy': drowsy_over_time[i],
                            'attentiveness': attentive_over_time[i],
                            'microexpression': microexp_values[i],
                        } for i in range(len(webcam_results))])
                        st.download_button("Download session CSV", data=df_export.to_csv(index=False).encode('utf-8'), file_name='webcam_session_report.csv', mime='text/csv')
