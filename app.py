import gc
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Pitching 2D Motion Review",
    page_icon="⚾",
    layout="wide",
)

mp_pose = mp.solutions.pose


# ---------------------------------------------------------
# Video / signal utilities
# ---------------------------------------------------------
def write_temp_video(video_bytes: bytes, suffix: str) -> str:
    suffix = suffix if suffix and suffix.startswith(".") else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def odd_window_from_ms(fps: float, milliseconds: float, minimum: int = 3, maximum: int = 61) -> int:
    fps = max(float(fps), 1.0)
    frames = max(minimum, int(round(fps * milliseconds / 1000.0)))
    frames = min(frames, maximum)
    if frames % 2 == 0:
        frames = frames + 1 if frames < maximum else frames - 1
    return max(minimum, min(maximum, frames))


def moving_average(values, window: int = 5):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return arr.copy()
    out = np.full(arr.shape, np.nan, dtype=float)
    half = window // 2
    for i in range(len(arr)):
        part = arr[max(0, i - half):min(len(arr), i + half + 1)]
        part = part[np.isfinite(part)]
        if part.size:
            out[i] = float(np.mean(part))
    return pd.Series(out).interpolate(limit_direction="both").to_numpy(dtype=float)


def smooth_ms(values, fps: float, ms: float):
    return moving_average(values, odd_window_from_ms(fps, ms))


def smooth_points(points, fps: float, ms: float):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return arr
    return np.column_stack([smooth_ms(arr[:, 0], fps, ms), smooth_ms(arr[:, 1], fps, ms)])


def derivative(values, fps: float, smoothing_ms: float):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return np.zeros_like(arr, dtype=float)
    arr = smooth_ms(arr, fps, smoothing_ms)
    return np.gradient(arr, 1.0 / max(float(fps), 1.0))


def interpolate_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return arr
    out = arr.copy()
    for col in range(2):
        s = pd.Series(out[:, col]).replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            out[:, col] = 0.0
        else:
            out[:, col] = s.interpolate(limit_direction="both").to_numpy(dtype=float)
    return out


def angle_2d(p1, p2):
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def unwrap_degrees(values):
    return np.degrees(np.unwrap(np.radians(np.asarray(values, dtype=float))))


def angle_difference_deg(a, b):
    diff = np.radians(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.degrees(np.arctan2(np.sin(diff), np.cos(diff)))


def distance_2d(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def elbow_angle_2d(shoulder, elbow, wrist):
    v1 = np.asarray(shoulder, dtype=float) - np.asarray(elbow, dtype=float)
    v2 = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return np.nan
    cos_theta = np.clip(float(np.dot(v1, v2) / denom), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def clean_scalar_series(values, fallback=0.0):
    s = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.full(len(s), float(fallback), dtype=float)
    return s.interpolate(limit_direction="both").fillna(float(fallback)).to_numpy(dtype=float)


def auto_foot_plant(lead_ankle, fps, smoothing_ms):
    """Rough FC candidate from projected lead-ankle movement; always verify visually."""
    points = np.asarray(lead_ankle, dtype=float)
    n = len(points)
    if n < 8:
        return max(0, n // 2)
    y = smooth_ms(points[:, 1], fps, smoothing_ms)
    vy = np.gradient(y, 1.0 / max(float(fps), 1.0))
    start = max(1, int(0.10 * n))
    end = min(n - 2, int(0.85 * n))
    if end <= start:
        return int(np.nanargmax(y))
    # Looking for the end of the lead ankle's downward movement.
    max_down = start + int(np.nanargmax(vy[start:end + 1]))
    settle = max(2, int(round(0.04 * fps)))
    search_end = min(n - 1, max_down + max(settle * 5, int(0.25 * fps)))
    vy_abs = np.abs(vy[max_down:search_end + 1])
    if len(vy_abs) == 0:
        return max_down
    threshold = max(float(np.percentile(np.abs(vy), 25)), 8.0)
    for idx in range(max_down, search_end + 1):
        section = np.abs(vy[idx:min(n, idx + settle)])
        if len(section) >= max(2, settle // 2) and np.mean(section < threshold) >= 0.65:
            return int(idx)
    return int(max_down + np.nanargmax(y[max_down:search_end + 1]))


def auto_release(wrist_speed, fc_idx, fps):
    """Rough wrist-speed peak candidate 40-300 ms after FC; verify visually."""
    speed = np.asarray(wrist_speed, dtype=float)
    n = len(speed)
    if n < 3:
        return max(0, n - 1)
    start = min(n - 1, fc_idx + max(1, int(round(0.04 * fps))))
    end = min(n - 1, fc_idx + max(3, int(round(0.30 * fps))))
    if end <= start:
        return min(fc_idx + 1, n - 1)
    section = speed[start:end + 1]
    if not np.isfinite(section).any():
        return start
    return int(start + np.nanargmax(section))


def frame_image(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            return None
        max_width = 900
        h, w = frame.shape[:2]
        if w > max_width:
            factor = max_width / w
            frame = cv2.resize(frame, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


@st.cache_data(show_spinner=False, max_entries=1)
def extract_pose_arrays(video_bytes: bytes, suffix: str):
    """Run MediaPipe once and cache only compact landmark arrays, never the video frames."""
    path = write_temp_video(video_bytes, suffix)
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("動画を開けませんでした。別のMP4/MOV形式を試してください。")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        keys = [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "throwing_shoulder", "throwing_elbow", "throwing_wrist",
            "pivot_ankle", "lead_ankle",
        ]
        collected = {key: [] for key in keys}
        detected = 0
        frame_count = 0
        # Downscale pose input to reduce compute and memory, coordinates remain normalized.
        process_width = 960

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h0, w0 = frame.shape[:2]
                if w0 > process_width:
                    factor = process_width / w0
                    process_frame = cv2.resize(frame, (int(w0 * factor), int(h0 * factor)), interpolation=cv2.INTER_AREA)
                else:
                    process_frame = frame
                results = pose.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))

                row = {key: (np.nan, np.nan) for key in keys}
                if results.pose_landmarks:
                    detected += 1
                    lm = results.pose_landmarks.landmark
                    def xy(landmark):
                        # Normalized landmark coordinates scaled to original frame dimensions.
                        return (float(landmark.x * width), float(landmark.y * height))
                    row["left_hip"] = xy(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    row["right_hip"] = xy(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    row["left_shoulder"] = xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
                    row["right_shoulder"] = xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    # Default throwing side is supplied in the UI after extraction; both arms are cached below.
                    row["throwing_shoulder"] = xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                    row["throwing_elbow"] = xy(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
                    row["throwing_wrist"] = xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
                    row["pivot_ankle"] = xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    row["lead_ankle"] = xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    # Cache left arm/legs too, enabling left-handed selection without rerunning MediaPipe.
                    row["left_shoulder_throw"] = xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
                    row["left_elbow_throw"] = xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
                    row["left_wrist_throw"] = xy(lm[mp_pose.PoseLandmark.LEFT_WRIST])
                    row["left_ankle_pivot"] = xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    row["right_ankle_lead"] = xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                for key in keys:
                    collected[key].append(row[key])
                # Extra coordinates for left-handed selection.
                if "left_shoulder_throw" not in collected:
                    collected["left_shoulder_throw"] = []
                    collected["left_elbow_throw"] = []
                    collected["left_wrist_throw"] = []
                    collected["left_ankle_pivot"] = []
                    collected["right_ankle_lead"] = []
                for key in ["left_shoulder_throw", "left_elbow_throw", "left_wrist_throw", "left_ankle_pivot", "right_ankle_lead"]:
                    collected[key].append(row.get(key, (np.nan, np.nan)))
                frame_count += 1

        cap.release()
        if frame_count == 0:
            raise RuntimeError("動画からフレームを読み込めませんでした。")
        arrays = {key: interpolate_points(np.asarray(val, dtype=float)) for key, val in collected.items()}
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "declared_frames": declared_frames,
            "frame_count": frame_count,
            "detected_ratio": detected / max(frame_count, 1),
            "arrays": arrays,
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("⚾ ピッチング動作・2D映像解析")
st.caption("映像から観察できる投影上の動きに限定した解析です。3Dバイオメカニクス測定の代替ではありません。")
st.sidebar.header("⚙️ 解析設定")

dominant_hand = st.sidebar.radio("投手タイプ", ["右投げ", "左投げ"])
video_fps_mode = st.sidebar.selectbox(
    "撮影時FPS（時間軸）",
    [
        "動画ファイルのFPSを使用",
        "通常撮影（30 fps）",
        "スロー撮影（60 fps → 30fps再生）",
        "ハイスピード（120 fps → 30fps再生）",
        "超スロー（240 fps → 30fps再生）",
    ],
    help="スローモーション動画は、ファイルが30fpsでも撮影時FPSが120/240の場合があります。撮影時FPSが分かる場合はそちらを選択してください。",
)
fps_map = {
    "通常撮影（30 fps）": 30.0,
    "スロー撮影（60 fps → 30fps再生）": 60.0,
    "ハイスピード（120 fps → 30fps再生）": 120.0,
    "超スロー（240 fps → 30fps再生）": 240.0,
}
smoothing_ms = st.sidebar.slider("平滑化時間 (ms)", 10, 80, 30, step=5, help="固定フレーム数ではなく、時間幅で平滑化します。")
show_wrist_trail = st.sidebar.checkbox("投球腕の手首軌道を表示", value=True)

uploaded = st.file_uploader("投球動画をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded is None:
    st.info("元の投球映像をアップロードしてください。3D骨格座標がなくても、2D投影上の動きを解析できます。")
    st.markdown(
        "**動画だけで扱う指標**：FC/Releaseの時刻、骨盤・胸郭ラインの2D角度と角速度、投影上の並進速度、手首軌道、2D肘角度。\n\n"
        "**動画だけでは正確に算出できない指標**：床反力、関節トルク、セグメントの運動エネルギー、エネルギー伝達効率、真の3D肩関節外旋角（MER）。"
    )
    st.stop()

video_bytes = uploaded.getvalue()
suffix = Path(uploaded.name).suffix or ".mp4"
video_hash = hashlib.sha1(video_bytes).hexdigest()[:12]

try:
    with st.spinner("MediaPipeで関節位置を抽出しています（同じ動画の再調整時はキャッシュを使います）..."):
        data = extract_pose_arrays(video_bytes, suffix)
except Exception as exc:
    st.error(f"動画解析を開始できませんでした: {exc}")
    st.stop()

orig_fps = float(data["fps"])
width, height = int(data["width"]), int(data["height"])
num_frames = int(data["frame_count"])
analysis_fps = orig_fps if video_fps_mode == "動画ファイルのFPSを使用" else float(fps_map[video_fps_mode])
playback_duration = num_frames / max(orig_fps, 1e-9)
analysis_duration = num_frames / max(analysis_fps, 1e-9)

a = data["arrays"]
left_hips = a["left_hip"]
right_hips = a["right_hip"]
left_shoulders = a["left_shoulder"]
right_shoulders = a["right_shoulder"]
pelvis_centers = (left_hips + right_hips) / 2.0
thorax_centers = (left_shoulders + right_shoulders) / 2.0

if dominant_hand == "右投げ":
    throwing_shoulder = a["throwing_shoulder"]
    throwing_elbow = a["throwing_elbow"]
    throwing_wrist = a["throwing_wrist"]
    pivot_ankle = a["pivot_ankle"]
    lead_ankle = a["lead_ankle"]
else:
    throwing_shoulder = a["left_shoulder_throw"]
    throwing_elbow = a["left_elbow_throw"]
    throwing_wrist = a["left_wrist_throw"]
    pivot_ankle = a["left_ankle_pivot"]
    lead_ankle = a["right_ankle_lead"]

pose_ratio = float(data["detected_ratio"])
st.info(
    f"動画: {width}×{height}px / ファイルFPS: {orig_fps:.2f} / 撮影時FPSとして計算: {analysis_fps:.2f} / "
    f"再生時間: {playback_duration:.2f}s / 動作時間の換算値: {analysis_duration:.2f}s"
)
if abs(analysis_fps - orig_fps) > 0.5:
    st.warning(
        f"ファイルFPSと撮影時FPSが異なる設定です。時間倍率は {analysis_fps / orig_fps:.2f}×。 "
        "これは撮影時FPSが正しく分かっている場合に限り、実動作時間の推定に使えます。"
    )
if pose_ratio < 0.70:
    st.warning(f"Pose検出率は {pose_ratio:.0%} でした。関節が隠れている・暗い・骨格アニメーションの場合、数値の信頼性は下がります。")
if pose_ratio < 0.30:
    st.error("姿勢検出率が低すぎるため、計算結果を表示しません。明るく全身が映る元の投球映像を使ってください。")
    st.stop()

# Time-based point smoothing
left_hips = smooth_points(left_hips, analysis_fps, smoothing_ms)
right_hips = smooth_points(right_hips, analysis_fps, smoothing_ms)
left_shoulders = smooth_points(left_shoulders, analysis_fps, smoothing_ms)
right_shoulders = smooth_points(right_shoulders, analysis_fps, smoothing_ms)
pelvis_centers = smooth_points(pelvis_centers, analysis_fps, smoothing_ms)
thorax_centers = smooth_points(thorax_centers, analysis_fps, smoothing_ms)
throwing_shoulder = smooth_points(throwing_shoulder, analysis_fps, smoothing_ms)
throwing_elbow = smooth_points(throwing_elbow, analysis_fps, smoothing_ms)
throwing_wrist = smooth_points(throwing_wrist, analysis_fps, smoothing_ms)
pivot_ankle = smooth_points(pivot_ankle, analysis_fps, smoothing_ms)
lead_ankle = smooth_points(lead_ankle, analysis_fps, smoothing_ms)

hip_widths = np.array([distance_2d(lh, rh) for lh, rh in zip(left_hips, right_hips)], dtype=float)
valid_hip_widths = hip_widths[np.isfinite(hip_widths) & (hip_widths > 5)]
if len(valid_hip_widths) < max(5, int(0.1 * num_frames)):
    st.error("股関節位置が十分に検出できませんでした。全身が映る、遮蔽物の少ない動画で試してください。")
    st.stop()
median_hip_width_px = float(np.median(valid_hip_widths))

# Use projected hip width as a normalizer, not an assumed real-world measurement.
# This avoids presenting uncalibrated m/s as real physical velocities.
pelvis_x_hw = (pelvis_centers[:, 0] - pelvis_centers[0, 0]) / median_hip_width_px
thorax_x_hw = (thorax_centers[:, 0] - thorax_centers[0, 0]) / median_hip_width_px
stance_vector = np.nanmedian(lead_ankle[:, 0] - pivot_ankle[:, 0])
direction_sign = 1.0 if stance_vector >= 0 else -1.0
pelvis_x_hw *= direction_sign
thorax_x_hw *= direction_sign
pelvis_velocity_hw_s = derivative(pelvis_x_hw, analysis_fps, smoothing_ms)
thorax_velocity_hw_s = derivative(thorax_x_hw, analysis_fps, smoothing_ms)

hip_line_angles = unwrap_degrees(np.array([angle_2d(lh, rh) for lh, rh in zip(left_hips, right_hips)]))
shoulder_line_angles = unwrap_degrees(np.array([angle_2d(ls, rs) for ls, rs in zip(left_shoulders, right_shoulders)]))
hip_line_angles = smooth_ms(hip_line_angles, analysis_fps, smoothing_ms)
shoulder_line_angles = smooth_ms(shoulder_line_angles, analysis_fps, smoothing_ms)
hip_line_angular_velocity = derivative(hip_line_angles, analysis_fps, smoothing_ms)
shoulder_line_angular_velocity = derivative(shoulder_line_angles, analysis_fps, smoothing_ms)
line_angle_difference = angle_difference_deg(shoulder_line_angles, hip_line_angles)

wrist_xy_hw = throwing_wrist / median_hip_width_px
wrist_vx = derivative(wrist_xy_hw[:, 0], analysis_fps, smoothing_ms)
wrist_vy = derivative(wrist_xy_hw[:, 1], analysis_fps, smoothing_ms)
wrist_speed_hw_s = smooth_ms(np.sqrt(wrist_vx**2 + wrist_vy**2), analysis_fps, smoothing_ms)
elbow_angle = smooth_ms(
    np.array([elbow_angle_2d(s, e, w) for s, e, w in zip(throwing_shoulder, throwing_elbow, throwing_wrist)]),
    analysis_fps,
    smoothing_ms,
)

# Automatic candidates are only suggestions; event positions can be manually corrected below.
auto_fc_idx = auto_foot_plant(lead_ankle, analysis_fps, smoothing_ms)
auto_release_idx = auto_release(wrist_speed_hw_s, auto_fc_idx, analysis_fps)
start_mer = min(auto_fc_idx, num_frames - 1)
end_mer = min(max(start_mer + 1, auto_release_idx), num_frames - 1)
if end_mer > start_mer and np.isfinite(elbow_angle[start_mer:end_mer + 1]).any():
    auto_mer_idx = start_mer + int(np.nanargmax(elbow_angle[start_mer:end_mer + 1]))
else:
    auto_mer_idx = start_mer

st.subheader("① イベント時刻の確認")
st.caption("FC/Releaseは自動推定の誤りが起きやすいため、下のフレームを動画で確認して調整してください。MERは2D動画から自動計測できません。初期値は肘角度ピーク付近の参考点です。")
with st.form(f"event_form_{video_hash}"):
    fc_idx = st.slider("FC（前足接地）フレーム", 0, num_frames - 1, int(auto_fc_idx), key=f"fc_{video_hash}")
    mer_idx = st.slider("MERの目視参考フレーム（真の肩外旋角ではありません）", 0, num_frames - 1, int(auto_mer_idx), key=f"mer_{video_hash}")
    release_idx = st.slider("Release（ボール離脱）フレーム", 0, num_frames - 1, int(auto_release_idx), key=f"rel_{video_hash}")
    st.form_submit_button("イベント位置を適用")

if release_idx <= fc_idx:
    st.error("ReleaseはFCより後のフレームを選んでください。下のフレーム指定を修正してください。")
    st.stop()
if not (fc_idx <= mer_idx <= release_idx):
    st.warning("MER目視参考フレームは、通常FC〜Releaseの範囲内に置くのが適切です。必要に応じて修正してください。")

# Frame previews: playback time vs estimated action time.
preview_path = write_temp_video(video_bytes, suffix)
try:
    preview_cols = st.columns(3)
    for col, label, idx in zip(preview_cols, ["FC", "MER目視参考", "Release"], [fc_idx, mer_idx, release_idx]):
        preview = frame_image(preview_path, idx)
        if preview is not None:
            col.image(preview, caption=f"{label}: frame {idx} / playback {idx / max(orig_fps, 1e-9):.3f}s / estimated action {idx / max(analysis_fps, 1e-9):.3f}s", use_container_width=True)
finally:
    try:
        os.unlink(preview_path)
    except OSError:
        pass

# Reported indicators are deliberately limited to image-plane / normalized values.
times_action = np.arange(num_frames, dtype=float) / max(analysis_fps, 1.0)
times_playback = np.arange(num_frames, dtype=float) / max(orig_fps, 1.0)
step_width_hw = distance_2d(lead_ankle[fc_idx], pivot_ankle[fc_idx]) / median_hip_width_px
fc_to_release_ms = (release_idx - fc_idx) / max(analysis_fps, 1.0) * 1000.0
max_pelvis_trans = float(np.nanmax(np.abs(pelvis_velocity_hw_s)))
max_thorax_trans = float(np.nanmax(np.abs(thorax_velocity_hw_s)))
max_hip_angular = float(np.nanmax(np.abs(hip_line_angular_velocity)))
max_shoulder_angular = float(np.nanmax(np.abs(shoulder_line_angular_velocity)))
max_wrist_speed = float(np.nanmax(wrist_speed_hw_s))
mer_reference_elbow_angle = float(elbow_angle[mer_idx]) if np.isfinite(elbow_angle[mer_idx]) else np.nan

st.subheader("📊 2D映像から得られる指標")
c1, c2, c3, c4 = st.columns(4)
c1.metric("骨盤中心の最大投影速度", f"{max_pelvis_trans:.2f} 股関節幅/s")
c2.metric("胸郭中心の最大投影速度", f"{max_thorax_trans:.2f} 股関節幅/s")
c3.metric("骨盤ライン最大角速度", f"{max_hip_angular:.0f} °/s")
c4.metric("肩ライン最大角速度", f"{max_shoulder_angular:.0f} °/s")
c5, c6, c7, c8 = st.columns(4)
c5.metric("FC時の投影ステップ幅", f"{step_width_hw:.2f} 股関節幅")
c6.metric("手首最大投影速度", f"{max_wrist_speed:.2f} 股関節幅/s")
c7.metric("指定MER参考フレームの肘角度", f"{mer_reference_elbow_angle:.1f}°" if np.isfinite(mer_reference_elbow_angle) else "NA")
c8.metric("FC → Release", f"{fc_to_release_ms:.0f} ms")

st.subheader("📍 投球イベント")
e1, e2, e3 = st.columns(3)
e1.metric("FC", f"{times_action[fc_idx]:.3f} s")
e2.metric("MER目視参考", f"{times_action[mer_idx]:.3f} s")
e3.metric("Release", f"{times_action[release_idx]:.3f} s")
events_df = pd.DataFrame([
    {"イベント": label, "フレーム": int(idx), "動画上の時刻(s)": float(times_playback[idx]), "推定実動作時刻(s)": float(times_action[idx])}
    for label, idx in [("FC", fc_idx), ("MER目視参考", mer_idx), ("Release", release_idx)]
])
st.dataframe(events_df, use_container_width=True, hide_index=True)

st.warning(
    "このアプリは単眼2D映像の投影値です。股関節幅で正規化した速度・ステップ幅は、m/s・mの実測値ではありません。\n\n"
    "Next Baseレポートにある重心速度（m/s）、真の肩関節MER、床反力、セグメントエネルギー、関節トルク、エネルギー伝達効率は、3D座標・身体パラメータ・力学計測などがないため、この動画から正確には再現できません。GRFやトルクを推定値として表示することもしません。"
)

# ---------------------------------------------------------
# Annotated H.264 video, streamed to ffmpeg without storing frames in RAM.
# ---------------------------------------------------------
st.subheader("② 解析動画")
with st.spinner("注釈付き動画を生成しています..."):
    input_path = write_temp_video(video_bytes, suffix)
    render_cap = cv2.VideoCapture(input_path)
    render_max_width = 960
    render_scale = min(1.0, render_max_width / max(width, 1))
    render_width = max(2, int(round(width * render_scale)))
    render_height = max(2, int(round(height * render_scale)))
    sx = render_width / max(width, 1)
    sy = render_height / max(height, 1)
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_analysis.mp4").name
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{render_width}x{render_height}", "-r", f"{orig_fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ]
    proc = None
    wrist_trail = []
    max_trail = 45
    render_progress = st.progress(0)
    processed = 0
    try:
        if not render_cap.isOpened():
            raise RuntimeError("動画を描画用に再度開けませんでした。")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for i in range(num_frames):
            ok, frame = render_cap.read()
            if not ok:
                break
            if render_scale < 0.999999:
                draw = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            else:
                draw = frame

            def pt(point):
                return (int(round(point[0] * sx)), int(round(point[1] * sy)))

            pelvis_pt = pt(pelvis_centers[i])
            thorax_pt = pt(thorax_centers[i])
            sh_pt = pt(throwing_shoulder[i])
            el_pt = pt(throwing_elbow[i])
            wr_pt = pt(throwing_wrist[i])
            pivot_pt = pt(pivot_ankle[i])
            lead_pt = pt(lead_ankle[i])
            r = max(4, int(round(7 * render_scale)))
            lw = max(2, int(round(3 * render_scale)))
            cv2.circle(draw, pelvis_pt, r, (255, 80, 0), -1)
            cv2.circle(draw, thorax_pt, r, (0, 220, 70), -1)
            cv2.line(draw, sh_pt, el_pt, (0, 255, 255), lw)
            cv2.line(draw, el_pt, wr_pt, (0, 255, 255), lw)
            cv2.circle(draw, pivot_pt, r, (255, 255, 0), -1)
            cv2.circle(draw, lead_pt, r, (255, 255, 0), -1)
            if show_wrist_trail:
                wrist_trail.append(wr_pt)
                if len(wrist_trail) > max_trail:
                    wrist_trail.pop(0)
                for j in range(1, len(wrist_trail)):
                    cv2.line(draw, wrist_trail[j - 1], wrist_trail[j], (0, 0, 255), lw)
            if i == fc_idx:
                cv2.line(draw, pivot_pt, lead_pt, (255, 0, 255), lw + 1)
                cv2.putText(draw, "FC (user checked)", (max(8, lead_pt[0] + 5), max(20, lead_pt[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            if i == mer_idx:
                cv2.putText(draw, "MER reference (2D only)", (max(8, wr_pt[0] + 5), max(20, wr_pt[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            if i == release_idx:
                cv2.putText(draw, "Release (user checked)", (max(8, wr_pt[0] + 5), min(render_height - 10, wr_pt[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            info = [
                f"Playback time: {times_playback[i]:.3f}s",
                f"Estimated action time: {times_action[i]:.3f}s",
                f"Hip-line angle speed: {hip_line_angular_velocity[i]:.0f} deg/s",
                f"Shoulder-line angle speed: {shoulder_line_angular_velocity[i]:.0f} deg/s",
                f"Wrist speed: {wrist_speed_hw_s[i]:.2f} hip-width/s",
            ]
            for row_idx, text in enumerate(info):
                cv2.putText(draw, text, (10, 22 + row_idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if proc.stdin is None:
                raise BrokenPipeError("ffmpeg stdin is unavailable")
            proc.stdin.write(memoryview(draw).cast("B"))
            processed += 1
            if processed % 10 == 0:
                render_progress.progress(min(processed / max(num_frames, 1), 1.0))

        # Do NOT close stdin manually before communicate(); communicate manages EOF/wait/stderr.
        _, stderr_bytes = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            err = stderr_bytes.decode("utf-8", errors="replace") if isinstance(stderr_bytes, bytes) else str(stderr_bytes)
            raise RuntimeError("ffmpegエンコードに失敗しました: " + (err[-2000:] or "原因不明"))
    except Exception as exc:
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=5)
            except Exception:
                pass
        st.error(f"解析動画を生成できませんでした: {exc}")
        st.stop()
    finally:
        render_cap.release()
        try:
            os.unlink(input_path)
        except OSError:
            pass
    render_progress.progress(1.0)

st.video(out_path)
st.caption("注釈動画はH.264・最大960px幅で出力します。スロー再生上の時刻と推定実動作時刻を分けて表示しています。")

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
def event_lines(fig):
    for idx, label, dash in [(fc_idx, "FC", "dash"), (mer_idx, "MER reference", "dashdot"), (release_idx, "Release", "dot")]:
        fig.add_vline(x=times_action[idx], line_dash=dash, annotation_text=label)

st.subheader("📈 骨盤・胸郭中心の投影速度")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=times_action, y=pelvis_velocity_hw_s, mode="lines", name="Pelvis center"))
fig1.add_trace(go.Scatter(x=times_action, y=thorax_velocity_hw_s, mode="lines", name="Thorax center"))
event_lines(fig1)
fig1.update_layout(xaxis_title="Estimated action time (s)", yaxis_title="Projected velocity (hip-width/s)", height=380, template="plotly_dark")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔄 骨盤・肩ラインの画像内角速度")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=times_action, y=hip_line_angular_velocity, mode="lines", name="Hip line (2D)"))
fig2.add_trace(go.Scatter(x=times_action, y=shoulder_line_angular_velocity, mode="lines", name="Shoulder line (2D)"))
event_lines(fig2)
fig2.update_layout(xaxis_title="Estimated action time (s)", yaxis_title="Image-plane angular velocity (deg/s)", height=380, template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("↔️ 骨盤ライン−肩ラインの2D角度差")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=times_action, y=line_angle_difference, mode="lines", name="Shoulder line − hip line"))
event_lines(fig3)
fig3.update_layout(xaxis_title="Estimated action time (s)", yaxis_title="Projected angle difference (deg)", height=330, template="plotly_dark")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🦾 投球腕の2D肘角度")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=times_action, y=elbow_angle, mode="lines", name="2D elbow angle"))
fig4.add_trace(go.Scatter(x=[times_action[mer_idx]], y=[elbow_angle[mer_idx]], mode="markers+text", text=["Reference frame"], textposition="top center", name="Selected frame"))
event_lines(fig4)
fig4.update_layout(xaxis_title="Estimated action time (s)", yaxis_title="Projected elbow angle (deg)", height=330, template="plotly_dark")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("🖐️ 手首の投影速度")
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=times_action, y=wrist_speed_hw_s, mode="lines", name="Wrist (2D)"))
event_lines(fig5)
fig5.update_layout(xaxis_title="Estimated action time (s)", yaxis_title="Projected wrist speed (hip-width/s)", height=330, template="plotly_dark")
st.plotly_chart(fig5, use_container_width=True)

# CSV values deliberately avoid unsupported physical-unit claims.
df = pd.DataFrame({
    "Frame": np.arange(num_frames),
    "Playback_Time_s": times_playback,
    "Estimated_Action_Time_s": times_action,
    "Pelvis_Center_X_HipWidths": pelvis_x_hw,
    "Thorax_Center_X_HipWidths": thorax_x_hw,
    "Pelvis_Projected_Velocity_HipWidths_s": pelvis_velocity_hw_s,
    "Thorax_Projected_Velocity_HipWidths_s": thorax_velocity_hw_s,
    "Hip_Line_Angle_2D_deg": hip_line_angles,
    "Shoulder_Line_Angle_2D_deg": shoulder_line_angles,
    "Hip_Line_Angular_Velocity_2D_deg_s": hip_line_angular_velocity,
    "Shoulder_Line_Angular_Velocity_2D_deg_s": shoulder_line_angular_velocity,
    "Line_Angle_Difference_2D_deg": line_angle_difference,
    "Wrist_Speed_HipWidths_s": wrist_speed_hw_s,
    "Elbow_Angle_2D_deg": elbow_angle,
    "FC_Selected": np.arange(num_frames) == fc_idx,
    "MER_Reference_Selected": np.arange(num_frames) == mer_idx,
    "Release_Selected": np.arange(num_frames) == release_idx,
})
st.subheader("📥 解析データ")
st.download_button(
    "CSVをダウンロード",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="pitching_2d_analysis.csv",
    mime="text/csv",
)
st.caption("本アプリの値は同一動画・同一撮影条件の動作比較向けです。Next Baseの3D計測レポートの値と数値を直接比較しないでください。")

