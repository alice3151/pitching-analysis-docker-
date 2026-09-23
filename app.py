import gc
import os
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Streamlit
# =========================================================

st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide",
)


mp_pose = mp.solutions.pose


# =========================================================
# Signal processing utilities
# =========================================================

def odd_window_from_ms(
    fps: float,
    milliseconds: float,
    minimum: int = 3,
    maximum: int = 61,
) -> int:
    """時間幅(ms)を奇数フレームの窓幅へ変換する。"""
    fps = max(float(fps), 1.0)
    frames = int(round(fps * milliseconds / 1000.0))
    frames = max(int(minimum), frames)
    frames = min(int(maximum), frames)
    if frames % 2 == 0:
        frames += 1
    return max(int(minimum), min(int(maximum), frames))


def moving_average(data, window=5):
    """NaNに配慮した中心移動平均。"""
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        return arr

    window = max(1, int(window))
    if window == 1:
        return arr.copy()

    if window % 2 == 0:
        window += 1

    out = np.full_like(arr, np.nan, dtype=float)
    half = window // 2

    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        values = arr[start:end]
        valid = values[np.isfinite(values)]
        if len(valid):
            out[i] = np.mean(valid)

    return (
        pd.Series(out)
        .interpolate(limit_direction="both")
        .to_numpy()
    )


def smooth_signal_time(data, fps, smoothing_ms):
    window = odd_window_from_ms(fps, smoothing_ms)
    return moving_average(data, window)


def smooth_points(points, fps, smoothing_ms):
    arr = np.asarray(points, dtype=float)
    if len(arr) == 0:
        return arr
    return np.column_stack(
        [
            smooth_signal_time(arr[:, 0], fps, smoothing_ms),
            smooth_signal_time(arr[:, 1], fps, smoothing_ms),
        ]
    )


def derivative_per_second(data, fps, smoothing_ms):
    """
    まず時間ベースで平滑化し、その後に秒あたりの微分を計算する。
    240fpsを選んだ時に「5フレーム固定」の平滑化にならないことが重要。
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) < 2:
        return np.zeros_like(arr, dtype=float)

    smoothed = smooth_signal_time(arr, fps, smoothing_ms)
    dt = 1.0 / max(float(fps), 1.0)
    return np.gradient(smoothed, dt)


def angle_2d(p1, p2):
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def unwrap_angle_deg(angles):
    arr = np.asarray(angles, dtype=float)
    return np.degrees(np.unwrap(np.radians(arr)))


def distance_2d(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def calculate_elbow_angle(shoulder, elbow, wrist):
    v1 = np.asarray(shoulder, dtype=float) - np.asarray(elbow, dtype=float)
    v2 = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan

    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def robust_scale(values, fallback=0.001):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float(fallback)
    return float(np.median(arr))


def clean_series(values, fallback):
    arr = np.asarray(values, dtype=float)
    arr = (
        pd.Series(arr)
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(float(fallback))
        .to_numpy()
    )
    return arr


def cumulative_scaled_position(points, scale_per_frame, axis):
    """フレーム間変位を、その時点のスケールでm換算して累積する。"""
    arr = np.asarray(points, dtype=float)
    scales = np.asarray(scale_per_frame, dtype=float)

    n = len(arr)
    if n == 0:
        return np.asarray([], dtype=float)
    if len(scales) != n:
        scales = np.full(n, robust_scale(scales), dtype=float)

    fallback = robust_scale(scales)
    scales = clean_series(scales, fallback)

    position = np.zeros(n, dtype=float)
    if n == 1:
        return position

    delta_px = np.diff(arr[:, axis])
    pair_scale = 0.5 * (scales[:-1] + scales[1:])
    delta_m = delta_px * pair_scale
    delta_m[~np.isfinite(delta_m)] = 0.0
    position[1:] = np.cumsum(delta_m)
    return position


def find_foot_plant(lead_ankles, fps, smoothing_ms):
    """
    簡易Foot Plant推定。
    前足の下向き移動が最大になった後、Y速度が落ち着く候補を探す。
    動画だけから真のFCを保証するものではない。
    """
    points = np.asarray(lead_ankles, dtype=float)
    n = len(points)
    if n < 10:
        return max(0, n // 2)

    y = smooth_signal_time(points[:, 1], fps, smoothing_ms)
    vy = np.gradient(y, 1.0 / max(float(fps), 1.0))

    start = max(2, int(n * 0.08))
    search_end = min(n - 5, int(n * 0.80))
    if search_end <= start:
        return int(np.nanargmax(y))

    segment_v = vy[start:search_end]
    finite_v = segment_v[np.isfinite(segment_v)]
    if len(finite_v) == 0:
        return int(np.nanargmax(y))

    # 画像Yは下向きが正。最も強く下へ動いた場所。
    fastest_down = start + int(np.nanargmax(segment_v))

    settle_frames = odd_window_from_ms(
        fps,
        milliseconds=50.0,
        minimum=3,
        maximum=31,
    )

    # 速度閾値は、個別動画の速度スケールに合わせる。
    abs_v = np.abs(segment_v[np.isfinite(segment_v)])
    stable_threshold = max(
        12.0,
        float(np.percentile(abs_v, 25)) * 0.75,
    )

    # fastest_downから一定範囲で最初の「安定候補」を探す。
    candidate_end = min(
        n - settle_frames - 1,
        fastest_down + max(
            int(round(0.40 * fps)),
            settle_frames * 4,
        ),
    )

    for idx in range(fastest_down, candidate_end + 1):
        end = min(n, idx + settle_frames)
        local = np.abs(vy[idx:end])
        local = local[np.isfinite(local)]
        if len(local) < max(2, settle_frames // 2):
            continue

        stable_ratio = float(np.mean(local <= stable_threshold))
        if stable_ratio >= 0.70:
            return int(idx)

    # フォールバック：下方向移動後のY最大付近。
    return int(
        fastest_down
        + np.nanargmax(y[fastest_down:candidate_end + 1])
    )


def find_release(wrist_speed, foot_plant_idx, fps, smoothing_ms):
    """
    2D動画からの簡易Release推定。
    FC後40〜300msを探索し、速度ピークを候補とする。
    「ボール離脱」を直接観測するものではない。
    """
    speed = np.asarray(wrist_speed, dtype=float)
    n = len(speed)
    if n < 3:
        return max(0, n - 1)

    search_speed = smooth_signal_time(speed, fps, max(15.0, smoothing_ms))

    min_offset = max(1, int(round(0.04 * fps)))
    max_offset = max(min_offset + 2, int(round(0.30 * fps)))

    start = min(foot_plant_idx + min_offset, n - 2)
    end = min(foot_plant_idx + max_offset, n - 1)
    if end <= start:
        return min(foot_plant_idx + 1, n - 1)

    segment = search_speed[start:end + 1]
    finite = segment[np.isfinite(segment)]
    if len(finite) == 0:
        return start

    peak = float(np.max(finite))
    threshold = peak * 0.92

    # 最大値付近の最初の局所ピークを優先。
    for i in range(start + 1, end):
        cur = search_speed[i]
        prev_v = search_speed[i - 1]
        next_v = search_speed[i + 1]
        if not (np.isfinite(cur) and np.isfinite(prev_v) and np.isfinite(next_v)):
            continue
        if cur >= threshold and cur >= prev_v and cur >= next_v:
            return int(i)

    return int(start + np.nanargmax(segment))


def normalize_angle_difference_deg(a, b):
    diff = np.radians(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.degrees(np.arctan2(np.sin(diff), np.cos(diff)))


# =========================================================
# UI settings
# =========================================================

st.title("⚾ ピッチング動作・運動力学解析")
st.sidebar.header("⚙️ 解析設定")

dominant_hand = st.sidebar.radio(
    "投手タイプ",
    ["右投げ", "左投げ"],
)

video_fps_mode = st.sidebar.selectbox(
    "撮影時FPS（時間軸）",
    [
        "動画のFPSを使用",
        "通常撮影（30 fps）",
        "スロー撮影（60 fps → 30fps再生）",
        "ハイスピード（120 fps → 30fps再生）",
        "超スロー（240 fps → 30fps再生）",
    ],
)

fps_map = {
    "通常撮影（30 fps）": 30.0,
    "スロー撮影（60 fps → 30fps再生）": 60.0,
    "ハイスピード（120 fps → 30fps再生）": 120.0,
    "超スロー（240 fps → 30fps再生）": 240.0,
}

user_weight = st.sidebar.number_input(
    "体重 (kg)",
    min_value=30.0,
    max_value=120.0,
    value=65.0,
    step=1.0,
)

reference_width_m = st.sidebar.number_input(
    "基準股関節幅 (m)",
    min_value=0.12,
    max_value=0.30,
    value=0.18,
    step=0.01,
)

smoothing_ms = st.sidebar.slider(
    "平滑化時間 (ms)",
    min_value=10,
    max_value=60,
    value=30,
    step=5,
    help="FPSに関係なく同じ時間幅で平滑化します。240fpsでは30ms≒7フレームです。",
)

show_wrist_trail = st.sidebar.checkbox(
    "投球腕の手首軌道を表示",
    value=True,
)

uploaded_file = st.file_uploader(
    "動画ファイルをアップロードしてください",
    type=["mp4", "mov", "avi"],
)


# =========================================================
# Main analysis
# =========================================================

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    input_file.write(uploaded_file.getvalue())
    input_file.close()

    cap = cv2.VideoCapture(input_file.name)
    if not cap.isOpened():
        st.error("動画を開けませんでした。")
        st.stop()

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(orig_fps) or orig_fps <= 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps_mode == "動画のFPSを使用":
        analysis_fps = float(orig_fps)
    else:
        analysis_fps = float(fps_map[video_fps_mode])

    time_scale = analysis_fps / max(orig_fps, 1e-9)
    expected_playback_duration = total_frames / max(orig_fps, 1e-9)
    expected_analysis_duration = total_frames / max(analysis_fps, 1e-9)

    st.info(
        f"動画: {width} × {height}px / "
        f"ファイルFPS: {orig_fps:.2f} / "
        f"解析FPS: {analysis_fps:.2f} / "
        f"時間倍率: {time_scale:.2f}× / "
        f"再生時間: {expected_playback_duration:.2f}s / "
        f"解析時間: {expected_analysis_duration:.2f}s"
    )

    if abs(analysis_fps - orig_fps) > 0.5:
        st.warning(
            "ファイル上のFPSと解析FPSが異なります。"
            "スロー動画を元の撮影FPSに戻して解析する場合に使ってください。"
            "今回のように「240fps撮影→30fps再生」の動画なら、解析FPS=240で正しい時間軸になります。"
        )

    # Landmark selection
    is_right = dominant_hand == "右投げ"
    if is_right:
        throwing_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER
        throwing_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW
        throwing_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST
        pivot_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        lead_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
    else:
        throwing_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER
        throwing_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW
        throwing_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST
        pivot_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        lead_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE

    # Buffers
    left_hips = []
    right_hips = []
    left_shoulders = []
    right_shoulders = []
    pelvis_centers = []
    thorax_centers = []
    throwing_shoulders = []
    throwing_elbows = []
    throwing_wrists = []
    pivot_ankles = []
    lead_ankles = []
    scale_per_frame = []

    st.subheader("① 骨格解析")
    progress = st.progress(0)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        last_values = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                lh = (
                    lm[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                    lm[mp_pose.PoseLandmark.LEFT_HIP].y * height,
                )
                rh = (
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].x * width,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y * height,
                )
                pelvis = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

                ls = (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height,
                )
                rs = (
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height,
                )
                thorax = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)

                ts = (
                    lm[throwing_shoulder_idx].x * width,
                    lm[throwing_shoulder_idx].y * height,
                )
                te = (
                    lm[throwing_elbow_idx].x * width,
                    lm[throwing_elbow_idx].y * height,
                )
                tw = (
                    lm[throwing_wrist_idx].x * width,
                    lm[throwing_wrist_idx].y * height,
                )

                pa = (
                    lm[pivot_ankle_idx].x * width,
                    lm[pivot_ankle_idx].y * height,
                )
                la = (
                    lm[lead_ankle_idx].x * width,
                    lm[lead_ankle_idx].y * height,
                )

                current_scale = distance_2d(lh, rh)
                current_scale = (
                    reference_width_m / current_scale
                    if current_scale > 5
                    else np.nan
                )

                last_values = (lh, rh, pelvis, ls, rs, thorax, ts, te, tw, pa, la)
            else:
                if last_values is None:
                    center = (width / 2, height / 2)
                    lh = rh = pelvis = ls = rs = thorax = ts = te = tw = pa = la = center
                    current_scale = np.nan
                else:
                    lh, rh, pelvis, ls, rs, thorax, ts, te, tw, pa, la = last_values
                    current_scale = np.nan

            left_hips.append(lh)
            right_hips.append(rh)
            left_shoulders.append(ls)
            right_shoulders.append(rs)
            pelvis_centers.append(pelvis)
            thorax_centers.append(thorax)
            throwing_shoulders.append(ts)
            throwing_elbows.append(te)
            throwing_wrists.append(tw)
            pivot_ankles.append(pa)
            lead_ankles.append(la)
            scale_per_frame.append(current_scale)

            frame_idx += 1
            if total_frames > 0 and frame_idx % 5 == 0:
                progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress.progress(1.0)

    num_frames = frame_idx
    if num_frames < 10:
        st.error("動画から十分なフレームを取得できませんでした。")
        st.stop()

    # -----------------------------------------------------
    # Time-based smoothing
    # -----------------------------------------------------
    left_hips = smooth_points(left_hips, analysis_fps, smoothing_ms)
    right_hips = smooth_points(right_hips, analysis_fps, smoothing_ms)
    left_shoulders = smooth_points(left_shoulders, analysis_fps, smoothing_ms)
    right_shoulders = smooth_points(right_shoulders, analysis_fps, smoothing_ms)
    pelvis_centers = smooth_points(pelvis_centers, analysis_fps, smoothing_ms)
    thorax_centers = smooth_points(thorax_centers, analysis_fps, smoothing_ms)
    throwing_shoulders = smooth_points(throwing_shoulders, analysis_fps, smoothing_ms)
    throwing_elbows = smooth_points(throwing_elbows, analysis_fps, smoothing_ms)
    throwing_wrists = smooth_points(throwing_wrists, analysis_fps, smoothing_ms)
    pivot_ankles = smooth_points(pivot_ankles, analysis_fps, smoothing_ms)
    lead_ankles = smooth_points(lead_ankles, analysis_fps, smoothing_ms)

    scale_fallback = reference_width_m / max(1.0, np.nanmedian(
        [distance_2d(lh, rh) for lh, rh in zip(left_hips, right_hips)]
    ))
    scale_per_frame = clean_series(scale_per_frame, scale_fallback)
    scale_per_frame = smooth_signal_time(scale_per_frame, analysis_fps, smoothing_ms)
    scale = robust_scale(scale_per_frame, scale_fallback)

    # -----------------------------------------------------
    # Direction
    # -----------------------------------------------------
    stance_vector = np.nanmedian(lead_ankles[:, 0] - pivot_ankles[:, 0])
    direction_sign = 1.0 if stance_vector >= 0 else -1.0

    # -----------------------------------------------------
    # Translation
    # -----------------------------------------------------
    pelvis_x_m = cumulative_scaled_position(pelvis_centers, scale_per_frame, axis=0)
    thorax_x_m = cumulative_scaled_position(thorax_centers, scale_per_frame, axis=0)
    pelvis_translation = pelvis_x_m * direction_sign
    thorax_translation = thorax_x_m * direction_sign

    pelvis_velocity = derivative_per_second(pelvis_translation, analysis_fps, smoothing_ms)
    thorax_velocity = derivative_per_second(thorax_translation, analysis_fps, smoothing_ms)

    # -----------------------------------------------------
    # Rotation: angle -> time smoothing -> derivative
    # -----------------------------------------------------
    pelvis_angles_raw = np.array([
        angle_2d(lh, rh)
        for lh, rh in zip(left_hips, right_hips)
    ])
    thorax_angles_raw = np.array([
        angle_2d(ls, rs)
        for ls, rs in zip(left_shoulders, right_shoulders)
    ])

    pelvis_angles = unwrap_angle_deg(pelvis_angles_raw)
    thorax_angles = unwrap_angle_deg(thorax_angles_raw)

    pelvis_angles = smooth_signal_time(pelvis_angles, analysis_fps, smoothing_ms)
    thorax_angles = smooth_signal_time(thorax_angles, analysis_fps, smoothing_ms)

    pelvis_rotation_velocity = derivative_per_second(pelvis_angles, analysis_fps, smoothing_ms)
    thorax_rotation_velocity = derivative_per_second(thorax_angles, analysis_fps, smoothing_ms)

    trunk_separation = normalize_angle_difference_deg(thorax_angles, pelvis_angles)

    # -----------------------------------------------------
    # Foot Plant
    # -----------------------------------------------------
    foot_plant_idx = find_foot_plant(
        lead_ankles,
        fps=analysis_fps,
        smoothing_ms=smoothing_ms,
    )

    # -----------------------------------------------------
    # Wrist velocity
    # -----------------------------------------------------
    wrist_x_m = cumulative_scaled_position(throwing_wrists, scale_per_frame, axis=0)
    wrist_y_m = cumulative_scaled_position(throwing_wrists, scale_per_frame, axis=1)

    wrist_vx = derivative_per_second(wrist_x_m, analysis_fps, smoothing_ms)
    wrist_vy = derivative_per_second(wrist_y_m, analysis_fps, smoothing_ms)
    wrist_speed = np.sqrt(wrist_vx ** 2 + wrist_vy ** 2)
    wrist_speed = smooth_signal_time(wrist_speed, analysis_fps, smoothing_ms)

    # -----------------------------------------------------
    # Release
    # -----------------------------------------------------
    release_idx = find_release(
        wrist_speed,
        foot_plant_idx,
        fps=analysis_fps,
        smoothing_ms=smoothing_ms,
    )

    # -----------------------------------------------------
    # MER Proxy = 2D elbow angle, not true shoulder ER
    # -----------------------------------------------------
    elbow_angles_raw = np.array([
        calculate_elbow_angle(s, e, w)
        for s, e, w in zip(throwing_shoulders, throwing_elbows, throwing_wrists)
    ])
    elbow_angles = smooth_signal_time(elbow_angles_raw, analysis_fps, smoothing_ms)

    mer_start = min(foot_plant_idx, num_frames - 1)
    mer_end = max(mer_start + 1, release_idx - 1)
    mer_end = min(mer_end, num_frames - 1)

    mer_values = elbow_angles[mer_start:mer_end + 1]
    if np.any(np.isfinite(mer_values)):
        mer_idx = mer_start + int(np.nanargmax(mer_values))
    else:
        mer_idx = mer_start
    mer_angle = float(elbow_angles[mer_idx])

    # -----------------------------------------------------
    # Step width
    # -----------------------------------------------------
    step_width_px = distance_2d(
        lead_ankles[foot_plant_idx],
        pivot_ankles[foot_plant_idx],
    )
    step_scale = scale_per_frame[foot_plant_idx] if np.isfinite(scale_per_frame[foot_plant_idx]) else scale
    step_width_m = float(step_width_px * step_scale)

    # -----------------------------------------------------
    # Pseudo GRF
    # Acceleration is intentionally smoothed with a slightly wider time window
    # because second derivatives amplify MediaPipe jitter.
    # -----------------------------------------------------
    pelvis_y_m = cumulative_scaled_position(pelvis_centers, scale_per_frame, axis=1)
    accel_smoothing_ms = max(40.0, float(smoothing_ms))
    pelvis_vy = derivative_per_second(pelvis_y_m, analysis_fps, accel_smoothing_ms)
    pelvis_ay = derivative_per_second(pelvis_vy, analysis_fps, accel_smoothing_ms)

    vertical_acc_up = -pelvis_ay
    pseudo_grf = user_weight * (vertical_acc_up + 9.81)
    pseudo_grf = np.maximum(pseudo_grf, 0.0)
    pseudo_grf = smooth_signal_time(pseudo_grf, analysis_fps, accel_smoothing_ms)
    pseudo_grf_bw = pseudo_grf / max(user_weight * 9.81, 1e-9)

    # -----------------------------------------------------
    # Time axis
    # -----------------------------------------------------
    times = np.arange(num_frames, dtype=float) / max(analysis_fps, 1.0)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    max_pelvis_velocity = float(np.nanmax(np.abs(pelvis_velocity)))
    max_thorax_velocity = float(np.nanmax(np.abs(thorax_velocity)))
    max_pelvis_rotation = float(np.nanmax(np.abs(pelvis_rotation_velocity)))
    max_thorax_rotation = float(np.nanmax(np.abs(thorax_rotation_velocity)))
    max_wrist_speed = float(np.nanmax(wrist_speed))
    max_pseudo_grf = float(np.nanmax(pseudo_grf))

    df = pd.DataFrame({
        "Time_s": times,
        "Pelvis_Translation_m": pelvis_translation,
        "Thorax_Translation_m": thorax_translation,
        "Pelvis_Translation_Velocity_m_s": pelvis_velocity,
        "Thorax_Translation_Velocity_m_s": thorax_velocity,
        "Pelvis_Rotation_deg": pelvis_angles,
        "Thorax_Rotation_deg": thorax_angles,
        "Pelvis_Rotation_Velocity_deg_s": pelvis_rotation_velocity,
        "Thorax_Rotation_Velocity_deg_s": thorax_rotation_velocity,
        "Trunk_Separation_deg": trunk_separation,
        "Wrist_Speed_m_s": wrist_speed,
        "Elbow_Angle_2D_deg": elbow_angles,
        "Pseudo_GRF_N": pseudo_grf,
        "Pseudo_GRF_BW": pseudo_grf_bw,
    })

    # =====================================================
    # Render H.264 video
    # =====================================================
    st.subheader("② 解析動画生成")
    video_progress = st.progress(0)

    gc.collect()

    max_render_width = 1280
    render_scale = min(1.0, max_render_width / max(width, 1))
    render_width = max(2, int(round(width * render_scale)))
    render_height = max(2, int(round(height * render_scale)))
    sx = render_width / max(width, 1)
    sy = render_height / max(height, 1)

    overlay_h264_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_analysis_h264.mp4",
    ).name

    render_cap = cv2.VideoCapture(input_file.name)
    if not render_cap.isOpened():
        st.error("解析動画の入力を再度開けませんでした。")
        st.stop()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{render_width}x{render_height}",
        "-r", f"{orig_fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "25",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        overlay_h264_path,
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    MAX_TRAIL = 60
    wrist_trail = []
    i = 0

    def render_point(point):
        return (
            int(round(point[0] * sx)),
            int(round(point[1] * sy)),
        )

    try:
        while render_cap.isOpened():
            ret, frame = render_cap.read()
            if not ret or i >= num_frames:
                break

            if render_scale < 0.999999:
                draw_frame = cv2.resize(
                    frame,
                    (render_width, render_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                draw_frame = frame

            pelvis_pt = render_point(pelvis_centers[i])
            thorax_pt = render_point(thorax_centers[i])
            shoulder_pt = render_point(throwing_shoulders[i])
            elbow_pt = render_point(throwing_elbows[i])
            wrist_pt = render_point(throwing_wrists[i])
            pivot_pt = render_point(pivot_ankles[i])
            lead_pt = render_point(lead_ankles[i])

            circle_r = max(5, int(round(10 * render_scale)))
            small_r = max(4, int(round(7 * render_scale)))
            line_w = max(2, int(round(4 * render_scale)))
            trail_w = max(2, int(round(3 * render_scale)))

            cv2.circle(draw_frame, pelvis_pt, circle_r, (255, 0, 0), -1)
            cv2.circle(draw_frame, thorax_pt, circle_r, (0, 255, 0), -1)
            cv2.line(draw_frame, shoulder_pt, elbow_pt, (0, 255, 255), line_w)
            cv2.line(draw_frame, elbow_pt, wrist_pt, (0, 255, 255), line_w)
            cv2.circle(draw_frame, pivot_pt, small_r, (255, 255, 0), -1)
            cv2.circle(draw_frame, lead_pt, small_r, (255, 255, 0), -1)

            if show_wrist_trail:
                wrist_trail.append(wrist_pt)
                if len(wrist_trail) > MAX_TRAIL:
                    wrist_trail.pop(0)
                for k in range(1, len(wrist_trail)):
                    cv2.line(
                        draw_frame,
                        wrist_trail[k - 1],
                        wrist_trail[k],
                        (0, 0, 255),
                        trail_w,
                    )

            if i > 0:
                cv2.line(
                    draw_frame,
                    render_point(pelvis_centers[i - 1]),
                    pelvis_pt,
                    (255, 0, 0),
                    trail_w,
                )

            label_scale = max(0.38, 0.62 * render_scale)
            label_w = max(1, int(round(2 * render_scale)))

            if i == foot_plant_idx:
                cv2.line(draw_frame, pivot_pt, lead_pt, (255, 0, 255), max(3, int(round(5 * render_scale))))
                cv2.putText(
                    draw_frame,
                    "FOOT PLANT",
                    (lead_pt[0] + max(5, int(round(10 * render_scale))), max(20, lead_pt[1] - max(10, int(round(20 * render_scale))))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, 0.7 * render_scale),
                    (255, 0, 255),
                    label_w,
                )

            if i == mer_idx:
                cv2.putText(
                    draw_frame,
                    "MER PROXY",
                    (wrist_pt[0] + max(5, int(round(10 * render_scale))), max(20, wrist_pt[1] - max(5, int(round(10 * render_scale))))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, 0.8 * render_scale),
                    (0, 255, 255),
                    label_w,
                )

            if i == release_idx:
                cv2.putText(
                    draw_frame,
                    "RELEASE",
                    (wrist_pt[0] + max(5, int(round(10 * render_scale))), wrist_pt[1] + max(10, int(round(25 * render_scale)))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, 0.8 * render_scale),
                    (0, 255, 0),
                    label_w,
                )

            info = [
                f"Analysis Time: {times[i]:.3f}s",
                f"Pelvis Vel: {pelvis_velocity[i]:.2f} m/s",
                f"Thorax Vel: {thorax_velocity[i]:.2f} m/s",
                f"Pelvis Rot: {pelvis_rotation_velocity[i]:.0f} deg/s",
                f"Thorax Rot: {thorax_rotation_velocity[i]:.0f} deg/s",
                f"Wrist Speed: {wrist_speed[i]:.2f} m/s",
                f"Pseudo GRF: {pseudo_grf[i]:.0f} N",
            ]

            for j, text in enumerate(info):
                cv2.putText(
                    draw_frame,
                    text,
                    (max(10, int(round(30 * render_scale))), max(20, int(round((35 + j * 28) * render_scale)))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale,
                    (255, 255, 255),
                    label_w,
                )

            # IMPORTANT:
            # stdinはここでcloseしない。ループ後にcommunicate()がEOF処理まで担当する。
            if ffmpeg_process.stdin is None:
                raise BrokenPipeError

            ffmpeg_process.stdin.write(memoryview(draw_frame).cast("B"))

            i += 1
            if i % 10 == 0:
                video_progress.progress(min(i / max(num_frames, 1), 1.0))

    except BrokenPipeError:
        try:
            if ffmpeg_process.poll() is None:
                ffmpeg_process.kill()
        finally:
            render_cap.release()
            try:
                ffmpeg_process.communicate(timeout=5)
            except Exception:
                pass
        st.error("解析動画のエンコード中にffmpegが終了しました。")
        st.stop()
    finally:
        render_cap.release()

    # stdinは明示的にcloseせず、communicate()にclose + wait + stderr回収を任せる。
    _, ffmpeg_stderr = ffmpeg_process.communicate()
    return_code = ffmpeg_process.returncode

    if return_code != 0:
        stderr_text = (
            ffmpeg_stderr.decode("utf-8", errors="replace")
            if isinstance(ffmpeg_stderr, bytes)
            else str(ffmpeg_stderr)
        )
        st.error(
            "解析動画のH.264エンコードに失敗しました。\n\n"
            + (stderr_text or "原因不明")
        )
        st.stop()

    video_progress.progress(1.0)

    # =====================================================
    # Results
    # =====================================================
    st.success("解析が完了しました！")

    st.subheader("📊 ピッチング指標")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("骨盤最大並進速度", f"{max_pelvis_velocity:.2f} m/s")
    c2.metric("胸郭最大並進速度", f"{max_thorax_velocity:.2f} m/s")
    c3.metric("骨盤最大回旋速度", f"{max_pelvis_rotation:.0f} °/s")
    c4.metric("胸郭最大回旋速度", f"{max_thorax_rotation:.0f} °/s")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("MER Proxy（肘角度）", f"{mer_angle:.1f}°")
    c6.metric("ステップ幅", f"{step_width_m:.2f} m")
    c7.metric("最大手速度", f"{max_wrist_speed:.2f} m/s")
    c8.metric("最大擬似GRF", f"{max_pseudo_grf:.0f} N")

    st.subheader("📍 投球イベント")
    e1, e2, e3 = st.columns(3)
    e1.metric("Foot Plant", f"{times[foot_plant_idx]:.3f} s")
    e2.metric("MER Proxy", f"{times[mer_idx]:.3f} s")
    e3.metric("Release", f"{times[release_idx]:.3f} s")

    fc_release_ms = (times[release_idx] - times[foot_plant_idx]) * 1000.0
    st.caption(
        f"Foot Plant → Release: {fc_release_ms:.0f} ms / "
        f"ファイルFPS {orig_fps:.0f} → 解析FPS {analysis_fps:.0f}（{time_scale:.2f}×）"
    )

    st.warning(
        "このアプリは2D動画からの推定値です。"
        "骨盤・胸郭回旋速度は画像面内の角速度、"
        "MER Proxyは肘角度を用いた2D Proxy、"
        "GRFは骨盤鉛直加速度から算出したPseudo GRFです。"
        "単一カメラの遠近・奥行き誤差も残ります。"
    )

    # =====================================================
    # Videos
    # =====================================================
    st.subheader("📹 実動画 + 解析")
    st.video(overlay_h264_path)
    st.caption(
        "解析動画は元動画の再生FPSで出力。画面上のAnalysis Timeは選択した撮影FPSを基準にした実時間です。"
    )

    # =====================================================
    # Plot helper
    # =====================================================
    def add_event_lines(fig):
        fig.add_vline(
            x=times[foot_plant_idx],
            line_dash="dash",
            annotation_text="Foot Plant",
        )
        fig.add_vline(
            x=times[mer_idx],
            line_dash="dash",
            annotation_text="MER Proxy",
        )
        fig.add_vline(
            x=times[release_idx],
            line_dash="dot",
            annotation_text="Release",
        )

    st.subheader("📈 骨盤・胸郭 並進速度")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=pelvis_velocity, mode="lines", name="Pelvis"))
    fig1.add_trace(go.Scatter(x=times, y=thorax_velocity, mode="lines", name="Thorax"))
    add_event_lines(fig1)
    fig1.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Translation Velocity (m/s)", height=400, template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🔄 骨盤・胸郭 回旋速度")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=pelvis_rotation_velocity, mode="lines", name="Pelvis Rotation"))
    fig2.add_trace(go.Scatter(x=times, y=thorax_rotation_velocity, mode="lines", name="Thorax Rotation"))
    add_event_lines(fig2)
    fig2.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Angular Velocity (deg/s)", height=400, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("↔️ 骨盤−胸郭 Separation")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=trunk_separation, mode="lines", name="Pelvis-Thorax"))
    add_event_lines(fig3)
    fig3.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Separation Angle (deg)", height=350, template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🦾 投球腕 2D角度")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=times, y=elbow_angles, mode="lines", name="Elbow Angle"))
    fig4.add_trace(go.Scatter(x=[times[mer_idx]], y=[elbow_angles[mer_idx]], mode="markers+text", text=["MER Proxy"], textposition="top center", name="MER Proxy"))
    fig4.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Elbow Angle (deg)", height=350, template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🖐️ 投球手首速度")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=times, y=wrist_speed, mode="lines", name="Wrist Speed"))
    fig5.add_vline(x=times[foot_plant_idx], line_dash="dash", annotation_text="Foot Plant")
    fig5.add_vline(x=times[release_idx], line_dash="dot", annotation_text="Release")
    fig5.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Wrist Speed (m/s)", height=350, template="plotly_dark")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("🦶 擬似地面反力")
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=times, y=pseudo_grf, mode="lines", name="Pseudo GRF"))
    fig6.update_layout(xaxis_title="Analysis Time (s)", yaxis_title="Force (N)", height=350, template="plotly_dark")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("📥 解析データ")
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSVをダウンロード",
        data=csv_data,
        file_name="pitching_analysis.csv",
        mime="text/csv",
    )

    st.caption(
        "※ 実際の投球フォームの研究・選手評価に使う場合は、イベント位置とスケールを実動画や既知の計測系で検証してください。"
    )
