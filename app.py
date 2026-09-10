import os
import tempfile
import subprocess

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mediapipe as mp


# =========================================================
# Streamlit
# =========================================================

st.set_page_config(
    page_title="PITCHING KINETIC & ROTATIONAL ANALYSIS",
    page_icon="⚾",
    layout="wide"
)


# =========================================================
# MediaPipe
# =========================================================

mp_pose = mp.solutions.pose


# =========================================================
# Utilities
# =========================================================

def moving_average(
    data,
    window=5
):

    data = np.asarray(
        data,
        dtype=float
    )

    if len(data) == 0:
        return data

    window = max(
        1,
        int(window)
    )

    if window == 1:
        return data.copy()

    result = np.empty_like(
        data,
        dtype=float
    )

    half = window // 2

    for i in range(
        len(data)
    ):

        start = max(
            0,
            i - half
        )

        end = min(
            len(data),
            i + half + 1
        )

        values = data[
            start:end
        ]

        valid = values[
            np.isfinite(values)
        ]

        if len(valid) > 0:

            result[i] = np.mean(
                valid
            )

        else:

            result[i] = np.nan

    result = (
        pd.Series(result)
        .interpolate(
            limit_direction="both"
        )
        .to_numpy()
    )

    return result


def smooth_points(
    points,
    window=5
):

    arr = np.asarray(
        points,
        dtype=float
    )

    if len(arr) == 0:
        return arr

    x = moving_average(
        arr[:, 0],
        window
    )

    y = moving_average(
        arr[:, 1],
        window
    )

    return np.column_stack(
        [x, y]
    )


def distance_2d(
    p1,
    p2
):

    return float(
        np.linalg.norm(
            np.asarray(p1) -
            np.asarray(p2)
        )
    )


def angle_2d(
    p1,
    p2
):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return float(
        np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )
    )


def unwrap_angle_deg(
    angles
):

    return np.degrees(
        np.unwrap(
            np.radians(
                angles
            )
        )
    )


def velocity_1d(
    position,
    dt
):

    return np.gradient(
        position,
        dt
    )


def calculate_scale(
    scales
):

    arr = np.asarray(
        scales,
        dtype=float
    )

    arr = arr[
        np.isfinite(arr)
    ]

    arr = arr[
        arr > 0
    ]

    if len(arr) == 0:

        return 0.001

    return float(
        np.median(arr)
    )


def safe_point(
    p
):

    return (
        int(round(p[0])),
        int(round(p[1]))
    )


def calculate_elbow_angle(
    shoulder,
    elbow,
    wrist
):

    v1 = (
        np.asarray(shoulder) -
        np.asarray(elbow)
    )

    v2 = (
        np.asarray(wrist) -
        np.asarray(elbow)
    )

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:

        return np.nan

    cos_theta = (
        np.dot(v1, v2)
        /
        (n1 * n2)
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(
                cos_theta
            )
        )
    )


def find_foot_plant(
    lead_ankles
):

    """
    簡易Foot Plant推定。

    前足が下方向へ移動した後、
    Y座標が最大に近づく位置を候補とする。

    ※動画だけから完全なFCを保証するものではない。
    """

    y = lead_ankles[:, 1]

    n = len(y)

    if n < 10:

        return n // 2

    start = int(
        n * 0.15
    )

    end = int(
        n * 0.85
    )

    candidate = y[
        start:end
    ]

    return (
        start +
        int(
            np.nanargmax(
                candidate
            )
        )
    )


# =========================================================
# Sidebar
# =========================================================

st.title(
    "⚾ ピッチング動作・運動力学解析"
)

st.sidebar.header(
    "⚙️ 解析設定"
)


dominant_hand = st.sidebar.radio(
    "投手タイプ",
    [
        "右投げ",
        "左投げ"
    ]
)


video_fps_mode = st.sidebar.selectbox(
    "撮影スピード設定",
    [
        "動画のFPSを使用",
        "通常撮影 (30 fps)",
        "スロー撮影 (60 fps)",
        "ハイスピード (120 fps)",
        "超スロー (240 fps)"
    ]
)


fps_map = {

    "通常撮影 (30 fps)":
        30.0,

    "スロー撮影 (60 fps)":
        60.0,

    "ハイスピード (120 fps)":
        120.0,

    "超スロー (240 fps)":
        240.0

}


user_weight = st.sidebar.number_input(
    "体重 (kg)",
    min_value=30.0,
    max_value=120.0,
    value=65.0,
    step=1.0
)


reference_width_m = st.sidebar.number_input(
    "基準股関節幅 (m)",
    min_value=0.12,
    max_value=0.30,
    value=0.18,
    step=0.01
)


smooth_window = st.sidebar.slider(
    "平滑化フレーム数",
    3,
    11,
    5,
    step=2
)


show_wrist_trail = st.sidebar.checkbox(
    "投球腕の手首軌道を表示",
    value=True
)


uploaded_file = st.file_uploader(
    "動画ファイルをアップロードしてください",
    type=[
        "mp4",
        "mov",
        "avi"
    ]
)


# =========================================================
# Main
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # 一時動画
    # =====================================================

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    input_file.write(
        uploaded_file.read()
    )

    input_file.close()


    # =====================================================
    # Video Open
    # =====================================================

    cap = cv2.VideoCapture(
        input_file.name
    )

    if not cap.isOpened():

        st.error(
            "動画を開けませんでした。"
        )

        st.stop()


    orig_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        orig_fps is None
        or not np.isfinite(orig_fps)
        or orig_fps <= 0
    ):

        orig_fps = 30.0


    if (
        video_fps_mode ==
        "動画のFPSを使用"
    ):

        fps = orig_fps

    else:

        fps = fps_map[
            video_fps_mode
        ]


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    st.info(
        f"動画: {width} × {height}px / "
        f"元FPS: {orig_fps:.2f} / "
        f"解析FPS: {fps:.2f}"
    )


    # =====================================================
    # Landmark Index
    # =====================================================

    is_right = (
        dominant_hand ==
        "右投げ"
    )


    if is_right:

        throwing_shoulder_idx = (
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        )

        throwing_elbow_idx = (
            mp_pose.PoseLandmark.RIGHT_ELBOW
        )

        throwing_wrist_idx = (
            mp_pose.PoseLandmark.RIGHT_WRIST
        )

        pivot_ankle_idx = (
            mp_pose.PoseLandmark.RIGHT_ANKLE
        )

        lead_ankle_idx = (
            mp_pose.PoseLandmark.LEFT_ANKLE
        )

    else:

        throwing_shoulder_idx = (
            mp_pose.PoseLandmark.LEFT_SHOULDER
        )

        throwing_elbow_idx = (
            mp_pose.PoseLandmark.LEFT_ELBOW
        )

        throwing_wrist_idx = (
            mp_pose.PoseLandmark.LEFT_WRIST
        )

        pivot_ankle_idx = (
            mp_pose.PoseLandmark.LEFT_ANKLE
        )

        lead_ankle_idx = (
            mp_pose.PoseLandmark.RIGHT_ANKLE
        )


    # =====================================================
    # Buffers
    # =====================================================

    # 動画フレーム自体はRAMに保存しない

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

    scales = []


    # =====================================================
    # Pose
    # =====================================================

    st.subheader(
        "① 骨格解析"
    )

    progress = st.progress(
        0
    )


    # model_pathは/tmpに配置済み。
    # MediaPipe Legacy Pose内部のモデルDL処理は
    # model_complexity=1ならlite/heavyをDLしない。
    #
    # そこで1を使用し、
    # model自体の不足によるPermissionErrorを回避する。

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_idx = 0

        last_values = None

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            results = pose.process(
                rgb
            )


            if results.pose_landmarks:

                lm = (
                    results
                    .pose_landmarks
                    .landmark
                )


                # -----------------------------------------
                # Hip
                # -----------------------------------------

                lh = (
                    lm[
                        mp_pose.PoseLandmark.LEFT_HIP
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.LEFT_HIP
                    ].y * height
                )


                rh = (
                    lm[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].y * height
                )


                pelvis = (
                    (lh[0] + rh[0]) / 2,
                    (lh[1] + rh[1]) / 2
                )


                # -----------------------------------------
                # Shoulder
                # -----------------------------------------

                ls = (
                    lm[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y * height
                )


                rs = (
                    lm[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].x * width,

                    lm[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER
                    ].y * height
                )


                thorax = (
                    (ls[0] + rs[0]) / 2,
                    (ls[1] + rs[1]) / 2
                )


                # -----------------------------------------
                # Throwing Arm
                # -----------------------------------------

                ts = (
                    lm[
                        throwing_shoulder_idx
                    ].x * width,

                    lm[
                        throwing_shoulder_idx
                    ].y * height
                )


                te = (
                    lm[
                        throwing_elbow_idx
                    ].x * width,

                    lm[
                        throwing_elbow_idx
                    ].y * height
                )


                tw = (
                    lm[
                        throwing_wrist_idx
                    ].x * width,

                    lm[
                        throwing_wrist_idx
                    ].y * height
                )


                # -----------------------------------------
                # Feet
                # -----------------------------------------

                pa = (
                    lm[
                        pivot_ankle_idx
                    ].x * width,

                    lm[
                        pivot_ankle_idx
                    ].y * height
                )


                la = (
                    lm[
                        lead_ankle_idx
                    ].x * width,

                    lm[
                        lead_ankle_idx
                    ].y * height
                )


                # -----------------------------------------
                # Save
                # -----------------------------------------

                left_hips.append(
                    lh
                )

                right_hips.append(
                    rh
                )

                left_shoulders.append(
                    ls
                )

                right_shoulders.append(
                    rs
                )

                pelvis_centers.append(
                    pelvis
                )

                thorax_centers.append(
                    thorax
                )

                throwing_shoulders.append(
                    ts
                )

                throwing_elbows.append(
                    te
                )

                throwing_wrists.append(
                    tw
                )

                pivot_ankles.append(
                    pa
                )

                lead_ankles.append(
                    la
                )


                # -----------------------------------------
                # Scale
                # -----------------------------------------

                hip_width_px = distance_2d(
                    lh,
                    rh
                )

                if hip_width_px > 5:

                    scales.append(
                        reference_width_m /
                        hip_width_px
                    )


                last_values = (
                    lh,
                    rh,
                    pelvis,
                    ls,
                    rs,
                    thorax,
                    ts,
                    te,
                    tw,
                    pa,
                    la
                )


            else:

                if last_values is not None:

                    (
                        lh,
                        rh,
                        pelvis,
                        ls,
                        rs,
                        thorax,
                        ts,
                        te,
                        tw,
                        pa,
                        la
                    ) = last_values

                else:

                    center = (
                        width / 2,
                        height / 2
                    )

                    lh = center
                    rh = center
                    pelvis = center

                    ls = center
                    rs = center
                    thorax = center

                    ts = center
                    te = center
                    tw = center

                    pa = center
                    la = center


                left_hips.append(lh)
                right_hips.append(rh)

                left_shoulders.append(ls)
                right_shoulders.append(rs)

                pelvis_centers.append(
                    pelvis
                )

                thorax_centers.append(
                    thorax
                )

                throwing_shoulders.append(
                    ts
                )

                throwing_elbows.append(
                    te
                )

                throwing_wrists.append(
                    tw
                )

                pivot_ankles.append(
                    pa
                )

                lead_ankles.append(
                    la
                )


            frame_idx += 1


            if total_frames > 0:

                progress.progress(
                    min(
                        frame_idx /
                        total_frames,
                        1.0
                    )
                )


    cap.release()


    # 1パス目に実際に解析したフレーム数
    num_frames = frame_idx


    if num_frames < 10:

        st.error(
            "動画から十分なフレームを取得できませんでした。"
        )

        st.stop()


    # =====================================================
    # Smooth
    # =====================================================

    pelvis_centers = smooth_points(
        pelvis_centers,
        smooth_window
    )

    thorax_centers = smooth_points(
        thorax_centers,
        smooth_window
    )

    left_hips = smooth_points(
        left_hips,
        smooth_window
    )

    right_hips = smooth_points(
        right_hips,
        smooth_window
    )

    left_shoulders = smooth_points(
        left_shoulders,
        smooth_window
    )

    right_shoulders = smooth_points(
        right_shoulders,
        smooth_window
    )

    throwing_shoulders = smooth_points(
        throwing_shoulders,
        smooth_window
    )

    throwing_elbows = smooth_points(
        throwing_elbows,
        smooth_window
    )

    throwing_wrists = smooth_points(
        throwing_wrists,
        smooth_window
    )

    pivot_ankles = smooth_points(
        pivot_ankles,
        smooth_window
    )

    lead_ankles = smooth_points(
        lead_ankles,
        smooth_window
    )


    # =====================================================
    # Scale
    # =====================================================

    scale = calculate_scale(
        scales
    )

    dt = 1.0 / fps


    # =====================================================
    # 投球方向
    # =====================================================

    stance_vector = np.nanmedian(
        lead_ankles[:, 0] -
        pivot_ankles[:, 0]
    )

    direction_sign = (
        1.0
        if stance_vector >= 0
        else -1.0
    )


    # =====================================================
    # Pelvis / Thorax Translation
    # =====================================================

    pelvis_x_m = (
        pelvis_centers[:, 0] *
        scale
    )

    thorax_x_m = (
        thorax_centers[:, 0] *
        scale
    )


    pelvis_translation = (
        pelvis_x_m *
        direction_sign
    )

    thorax_translation = (
        thorax_x_m *
        direction_sign
    )


    pelvis_velocity = moving_average(
        velocity_1d(
            pelvis_translation,
            dt
        ),
        smooth_window
    )


    thorax_velocity = moving_average(
        velocity_1d(
            thorax_translation,
            dt
        ),
        smooth_window
    )


    # =====================================================
    # Pelvis Rotation
    # =====================================================

    pelvis_angles = np.array(
        [
            angle_2d(
                lh,
                rh
            )
            for lh, rh in zip(
                left_hips,
                right_hips
            )
        ]
    )


    pelvis_angles = unwrap_angle_deg(
        pelvis_angles
    )


    pelvis_rotation_velocity = moving_average(
        np.gradient(
            pelvis_angles,
            dt
        ),
        smooth_window
    )


    # =====================================================
    # Thorax Rotation
    # =====================================================

    thorax_angles = np.array(
        [
            angle_2d(
                ls,
                rs
            )
            for ls, rs in zip(
                left_shoulders,
                right_shoulders
            )
        ]
    )


    thorax_angles = unwrap_angle_deg(
        thorax_angles
    )


    thorax_rotation_velocity = moving_average(
        np.gradient(
            thorax_angles,
            dt
        ),
        smooth_window
    )


    # =====================================================
    # Separation
    # =====================================================

    trunk_separation = (
        thorax_angles -
        pelvis_angles
    )

    trunk_separation = unwrap_angle_deg(
        trunk_separation
    )


    # =====================================================
    # Foot Plant
    # =====================================================

    foot_plant_idx = find_foot_plant(
        lead_ankles
    )


    # =====================================================
    # Wrist Velocity
    # =====================================================

    wrist_x_m = (
        throwing_wrists[:, 0] *
        scale
    )

    wrist_y_m = (
        throwing_wrists[:, 1] *
        scale
    )


    wrist_vx = velocity_1d(
        wrist_x_m,
        dt
    )

    wrist_vy = velocity_1d(
        wrist_y_m,
        dt
    )


    wrist_speed = moving_average(
        np.sqrt(
            wrist_vx ** 2 +
            wrist_vy ** 2
        ),
        smooth_window
    )


    # =====================================================
    # Release
    # =====================================================

    release_start = min(
        foot_plant_idx + 1,
        num_frames - 1
    )


    release_values = wrist_speed[
        release_start:
    ]


    if len(release_values) > 0:

        release_idx = (
            release_start +
            int(
                np.nanargmax(
                    release_values
                )
            )
        )

    else:

        release_idx = (
            num_frames - 1
        )


    # =====================================================
    # MER
    # =====================================================

    elbow_angles = np.array(
        [
            calculate_elbow_angle(
                s,
                e,
                w
            )
            for s, e, w in zip(
                throwing_shoulders,
                throwing_elbows,
                throwing_wrists
            )
        ]
    )


    elbow_angles = moving_average(
        elbow_angles,
        smooth_window
    )


    mer_start = min(
        foot_plant_idx,
        num_frames - 1
    )

    mer_end = min(
        max(
            release_idx,
            mer_start + 1
        ),
        num_frames - 1
    )


    mer_values = elbow_angles[
        mer_start:
        mer_end + 1
    ]


    if np.any(
        np.isfinite(
            mer_values
        )
    ):

        mer_idx = (
            mer_start +
            int(
                np.nanargmax(
                    mer_values
                )
            )
        )

    else:

        mer_idx = mer_start


    mer_angle = float(
        elbow_angles[
            mer_idx
        ]
    )


    # =====================================================
    # Step Width
    # =====================================================

    step_width_px = distance_2d(
        lead_ankles[
            foot_plant_idx
        ],
        pivot_ankles[
            foot_plant_idx
        ]
    )


    step_width_m = (
        step_width_px *
        scale
    )


    # =====================================================
    # Pseudo GRF
    # =====================================================

    pelvis_y_m = (
        pelvis_centers[:, 1] *
        scale
    )


    pelvis_vy = velocity_1d(
        pelvis_y_m,
        dt
    )


    pelvis_ay = moving_average(
        velocity_1d(
            pelvis_vy,
            dt
        ),
        smooth_window
    )


    # image y is downward
    vertical_acc_up = (
        -pelvis_ay
    )


    pseudo_grf = (
        user_weight *
        (
            vertical_acc_up +
            9.81
        )
    )


    pseudo_grf = np.maximum(
        pseudo_grf,
        0
    )


    pseudo_grf = moving_average(
        pseudo_grf,
        smooth_window
    )


    pseudo_grf_bw = (
        pseudo_grf /
        (
            user_weight *
            9.81
        )
    )


    # =====================================================
    # Time
    # =====================================================

    times = (
        np.arange(
            num_frames
        ) * dt
    )


    # =====================================================
    # Metrics
    # =====================================================

    max_pelvis_velocity = float(
        np.nanmax(
            np.abs(
                pelvis_velocity
            )
        )
    )

    max_thorax_velocity = float(
        np.nanmax(
            np.abs(
                thorax_velocity
            )
        )
    )

    max_pelvis_rotation = float(
        np.nanmax(
            np.abs(
                pelvis_rotation_velocity
            )
        )
    )

    max_thorax_rotation = float(
        np.nanmax(
            np.abs(
                thorax_rotation_velocity
            )
        )
    )

    max_wrist_speed = float(
        np.nanmax(
            wrist_speed
        )
    )

    max_pseudo_grf = float(
        np.nanmax(
            pseudo_grf
        )
    )


    # =====================================================
    # DataFrame
    # =====================================================

    df = pd.DataFrame({

        "Time_s":
        times,

        "Pelvis_Translation_m":
        pelvis_translation,

        "Thorax_Translation_m":
        thorax_translation,

        "Pelvis_Translation_Velocity_m_s":
        pelvis_velocity,

        "Thorax_Translation_Velocity_m_s":
        thorax_velocity,

        "Pelvis_Rotation_deg":
        pelvis_angles,

        "Thorax_Rotation_deg":
        thorax_angles,

        "Pelvis_Rotation_Velocity_deg_s":
        pelvis_rotation_velocity,

        "Thorax_Rotation_Velocity_deg_s":
        thorax_rotation_velocity,

        "Trunk_Separation_deg":
        trunk_separation,

        "Wrist_Speed_m_s":
        wrist_speed,

        "Elbow_Angle_2D_deg":
        elbow_angles,

        "Pseudo_GRF_N":
        pseudo_grf,

        "Pseudo_GRF_BW":
        pseudo_grf_bw

    })


    # =====================================================
    # Video Output
    # =====================================================

    # Renderはメモリ節約のため1本だけ生成する。
    # 1920x1080の元動画は最大1280px幅へ縮小して描画し、
    # OpenCVの中間MP4を作らず、ffmpegへ直接H.264として流す。
    max_render_width = 1280

    render_scale = min(
        1.0,
        max_render_width / max(width, 1)
    )

    render_width = max(
        2,
        int(round(width * render_scale))
    )

    render_height = max(
        2,
        int(round(height * render_scale))
    )

    sx = render_width / max(width, 1)
    sy = render_height / max(height, 1)

    overlay_h264_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_analysis_h264.mp4"
    ).name


    st.subheader(
        "② 解析動画生成"
    )


    video_progress = st.progress(
        0
    )


    # メモリ回収。1パス目のMediaPipe一時領域を
    # 2パス目に持ち越さないようにする。
    import gc
    gc.collect()


    # 2パス目: 元動画を再読込して描画
    render_cap = cv2.VideoCapture(
        input_file.name
    )

    if not render_cap.isOpened():

        st.error(
            "解析動画の入力を再度開けませんでした。"
        )
        st.stop()


    # ffmpegへraw BGRフレームを直接渡す。
    # 中間mp4と2本目の動画を作らないことでメモリピークを抑える。
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


    wrist_trail = []

    # 軌道描画を直近60フレームに制限。
    MAX_TRAIL = 60

    i = 0

    try:

        while render_cap.isOpened():

            ret, frame = render_cap.read()

            if not ret:
                break

            if i >= num_frames:
                break

            # 1280px幅を上限に縮小して描画。
            if render_scale < 0.999999:

                draw_frame = cv2.resize(
                    frame,
                    (render_width, render_height),
                    interpolation=cv2.INTER_AREA
                )

            else:

                draw_frame = frame


            def render_point(p):

                return (
                    int(round(p[0] * sx)),
                    int(round(p[1] * sy))
                )


            # -------------------------------------------------
            # Points
            # -------------------------------------------------

            pelvis_pt = render_point(
                pelvis_centers[i]
            )

            thorax_pt = render_point(
                thorax_centers[i]
            )

            shoulder_pt = render_point(
                throwing_shoulders[i]
            )

            elbow_pt = render_point(
                throwing_elbows[i]
            )

            wrist_pt = render_point(
                throwing_wrists[i]
            )

            pivot_pt = render_point(
                pivot_ankles[i]
            )

            lead_pt = render_point(
                lead_ankles[i]
            )


            # -------------------------------------------------
            # Pelvis
            # -------------------------------------------------

            cv2.circle(
                draw_frame,
                pelvis_pt,
                max(5, int(round(10 * render_scale))),
                (255, 0, 0),
                -1
            )


            # -------------------------------------------------
            # Thorax
            # -------------------------------------------------

            cv2.circle(
                draw_frame,
                thorax_pt,
                max(5, int(round(10 * render_scale))),
                (0, 255, 0),
                -1
            )


            # -------------------------------------------------
            # Throwing Arm
            # -------------------------------------------------

            arm_thickness = max(
                2,
                int(round(4 * render_scale))
            )

            cv2.line(
                draw_frame,
                shoulder_pt,
                elbow_pt,
                (0, 255, 255),
                arm_thickness
            )

            cv2.line(
                draw_frame,
                elbow_pt,
                wrist_pt,
                (0, 255, 255),
                arm_thickness
            )


            # -------------------------------------------------
            # Feet
            # -------------------------------------------------

            cv2.circle(
                draw_frame,
                pivot_pt,
                max(4, int(round(7 * render_scale))),
                (255, 255, 0),
                -1
            )

            cv2.circle(
                draw_frame,
                lead_pt,
                max(4, int(round(7 * render_scale))),
                (255, 255, 0),
                -1
            )


            # -------------------------------------------------
            # Wrist Trail
            # -------------------------------------------------

            if show_wrist_trail:

                wrist_trail.append(
                    wrist_pt
                )

                if len(wrist_trail) > MAX_TRAIL:
                    wrist_trail.pop(0)

                trail_thickness = max(
                    2,
                    int(round(3 * render_scale))
                )

                for k in range(
                    1,
                    len(wrist_trail)
                ):

                    cv2.line(
                        draw_frame,
                        wrist_trail[k - 1],
                        wrist_trail[k],
                        (0, 0, 255),
                        trail_thickness
                    )


            # -------------------------------------------------
            # Pelvis Trail
            # -------------------------------------------------

            if i > 0:

                prev_pelvis = render_point(
                    pelvis_centers[i - 1]
                )

                cv2.line(
                    draw_frame,
                    prev_pelvis,
                    pelvis_pt,
                    (255, 0, 0),
                    max(2, int(round(3 * render_scale)))
                )


            # -------------------------------------------------
            # Foot Plant
            # -------------------------------------------------

            if i == foot_plant_idx:

                cv2.line(
                    draw_frame,
                    pivot_pt,
                    lead_pt,
                    (255, 0, 255),
                    max(3, int(round(5 * render_scale)))
                )

                cv2.putText(
                    draw_frame,
                    "FOOT PLANT",
                    (
                        lead_pt[0] + max(5, int(round(10 * render_scale))),
                        max(20, lead_pt[1] - max(10, int(round(20 * render_scale))))
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, 0.7 * render_scale),
                    (255, 0, 255),
                    max(1, int(round(2 * render_scale)))
                )


            # -------------------------------------------------
            # MER
            # -------------------------------------------------

            if i == mer_idx:

                cv2.putText(
                    draw_frame,
                    "MER",
                    (
                        wrist_pt[0] + max(5, int(round(10 * render_scale))),
                        max(20, wrist_pt[1] - max(5, int(round(10 * render_scale))))
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, 0.8 * render_scale),
                    (0, 255, 255),
                    max(1, int(round(2 * render_scale)))
                )


            # -------------------------------------------------
            # Release
            # -------------------------------------------------

            if i == release_idx:

                cv2.putText(
                    draw_frame,
                    "RELEASE",
                    (
                        wrist_pt[0] + max(5, int(round(10 * render_scale))),
                        wrist_pt[1] + max(10, int(round(25 * render_scale)))
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, 0.8 * render_scale),
                    (0, 255, 0),
                    max(1, int(round(2 * render_scale)))
                )


            # -------------------------------------------------
            # Info
            # -------------------------------------------------

            info = [
                f"Time: {times[i]:.3f}s",
                f"Pelvis Vel: {pelvis_velocity[i]:.2f} m/s",
                f"Thorax Vel: {thorax_velocity[i]:.2f} m/s",
                f"Pelvis Rot: {pelvis_rotation_velocity[i]:.0f} deg/s",
                f"Thorax Rot: {thorax_rotation_velocity[i]:.0f} deg/s",
                f"Wrist Speed: {wrist_speed[i]:.2f} m/s",
                f"Pseudo GRF: {pseudo_grf[i]:.0f} N"
            ]


            font_scale = max(
                0.38,
                0.62 * render_scale
            )

            text_thickness = max(
                1,
                int(round(2 * render_scale))
            )

            for j, text in enumerate(info):

                cv2.putText(
                    draw_frame,
                    text,
                    (
                        max(10, int(round(30 * render_scale))),
                        max(20, int(round((35 + j * 28) * render_scale)))
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    text_thickness
                )


            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.write(
                    memoryview(draw_frame).cast("B")
                )


            if i % 10 == 0:

                video_progress.progress(
                    min(
                        (i + 1) /
                        max(num_frames, 1),
                        1.0
                    )
                )

            i += 1


    except BrokenPipeError:

        st.error(
            "解析動画のエンコード中にffmpegが終了しました。"
        )
        st.stop()

    finally:

        render_cap.release()

        if ffmpeg_process.stdin is not None:
            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass


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


    video_progress.progress(
        1.0
    )


    # =====================================================
    # Metrics UI
    # =====================================================

    st.success(
        "解析が完了しました！"
    )


    st.subheader(
        "📊 ピッチング指標"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "骨盤最大並進速度",
        f"{max_pelvis_velocity:.2f} m/s"
    )


    c2.metric(
        "胸郭最大並進速度",
        f"{max_thorax_velocity:.2f} m/s"
    )


    c3.metric(
        "骨盤最大回旋速度",
        f"{max_pelvis_rotation:.0f} °/s"
    )


    c4.metric(
        "胸郭最大回旋速度",
        f"{max_thorax_rotation:.0f} °/s"
    )


    c5, c6, c7, c8 = st.columns(4)


    c5.metric(
        "MER 2D Proxy",
        f"{mer_angle:.1f}°"
    )


    c6.metric(
        "ステップ幅",
        f"{step_width_m:.2f} m"
    )


    c7.metric(
        "最大手速度",
        f"{max_wrist_speed:.2f} m/s"
    )


    c8.metric(
        "最大擬似GRF",
        f"{max_pseudo_grf:.0f} N"
    )


    # =====================================================
    # Events
    # =====================================================

    st.subheader(
        "📍 投球イベント"
    )


    e1, e2, e3 = st.columns(3)


    e1.metric(
        "Foot Plant",
        f"{times[foot_plant_idx]:.3f} s"
    )


    e2.metric(
        "MER",
        f"{times[mer_idx]:.3f} s"
    )


    e3.metric(
        "Release",
        f"{times[release_idx]:.3f} s"
    )


    # =====================================================
    # 注意
    # =====================================================

    st.warning(
        "2D動画解析による推定値です。"
        "骨盤・胸郭回旋速度は画像面内の角速度、"
        "MERは肘角度を用いた2D Proxy、"
        "GRFは骨盤鉛直加速度から算出したPseudo GRFです。"
    )


    # =====================================================
    # Videos
    # =====================================================

    st.subheader(
        "📹 実動画 + 解析"
    )

    st.video(
        overlay_h264_path
    )

    st.caption(
        "Renderのメモリ節約のため、解析動画は最大1280px幅・H.264で1本のみ生成しています。"
    )


    # =====================================================
    # Translation Graph
    # =====================================================

    st.subheader(
        "📈 骨盤・胸郭 並進速度"
    )


    fig1 = go.Figure()


    fig1.add_trace(
        go.Scatter(
            x=times,
            y=pelvis_velocity,
            mode="lines",
            name="Pelvis"
        )
    )


    fig1.add_trace(
        go.Scatter(
            x=times,
            y=thorax_velocity,
            mode="lines",
            name="Thorax"
        )
    )


    fig1.add_vline(
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )


    fig1.add_vline(
        x=times[
            mer_idx
        ],
        line_dash="dash",
        annotation_text="MER"
    )


    fig1.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
    )


    fig1.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Translation Velocity (m/s)",
        height=400,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # =====================================================
    # Rotation Graph
    # =====================================================

    st.subheader(
        "🔄 骨盤・胸郭 回旋速度"
    )


    fig2 = go.Figure()


    fig2.add_trace(
        go.Scatter(
            x=times,
            y=pelvis_rotation_velocity,
            mode="lines",
            name="Pelvis Rotation"
        )
    )


    fig2.add_trace(
        go.Scatter(
            x=times,
            y=thorax_rotation_velocity,
            mode="lines",
            name="Thorax Rotation"
        )
    )


    fig2.add_vline(
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )


    fig2.add_vline(
        x=times[
            mer_idx
        ],
        line_dash="dash",
        annotation_text="MER"
    )


    fig2.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
    )


    fig2.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Angular Velocity (deg/s)",
        height=400,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # =====================================================
    # Separation
    # =====================================================

    st.subheader(
        "↔️ 骨盤−胸郭 Separation"
    )


    fig3 = go.Figure()


    fig3.add_trace(
        go.Scatter(
            x=times,
            y=trunk_separation,
            mode="lines",
            name="Pelvis-Thorax"
        )
    )


    fig3.add_vline(
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )


    fig3.add_vline(
        x=times[
            mer_idx
        ],
        line_dash="dash",
        annotation_text="MER"
    )


    fig3.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Separation Angle (deg)",
        height=350,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # =====================================================
    # MER Graph
    # =====================================================

    st.subheader(
        "🦾 投球腕 2D角度"
    )


    fig4 = go.Figure()


    fig4.add_trace(
        go.Scatter(
            x=times,
            y=elbow_angles,
            mode="lines",
            name="Elbow Angle"
        )
    )


    fig4.add_trace(
        go.Scatter(
            x=[
                times[
                    mer_idx
                ]
            ],
            y=[
                elbow_angles[
                    mer_idx
                ]
            ],
            mode="markers+text",
            text=["MER"],
            textposition="top center",
            name="MER"
        )
    )


    fig4.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Elbow Angle (deg)",
        height=350,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig4,
        use_container_width=True
    )


    # =====================================================
    # Wrist Graph
    # =====================================================

    st.subheader(
        "🖐️ 投球手首速度"
    )


    fig5 = go.Figure()


    fig5.add_trace(
        go.Scatter(
            x=times,
            y=wrist_speed,
            mode="lines",
            name="Wrist Speed"
        )
    )


    fig5.add_vline(
        x=times[
            foot_plant_idx
        ],
        line_dash="dash",
        annotation_text="Foot Plant"
    )


    fig5.add_vline(
        x=times[
            release_idx
        ],
        line_dash="dot",
        annotation_text="Release"
    )


    fig5.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Wrist Speed (m/s)",
        height=350,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig5,
        use_container_width=True
    )


    # =====================================================
    # Pseudo GRF Graph
    # =====================================================

    st.subheader(
        "🦶 擬似地面反力"
    )


    fig6 = go.Figure()


    fig6.add_trace(
        go.Scatter(
            x=times,
            y=pseudo_grf,
            mode="lines",
            name="Pseudo GRF"
        )
    )


    fig6.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=350,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig6,
        use_container_width=True
    )


    # =====================================================
    # CSV
    # =====================================================

    st.subheader(
        "📥 解析データ"
    )


    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


    st.download_button(
        label="CSVをダウンロード",
        data=csv_data,
        file_name="pitching_analysis.csv",
        mime="text/csv"
    )
