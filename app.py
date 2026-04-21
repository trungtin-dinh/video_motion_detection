import os
import uuid
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np


DEFAULT_VIDEO_URL = "https://videos.pexels.com/video-files/6077402/6077402-uhd_4096_2160_25fps.mp4"


def make_temp_path(suffix: str) -> str:
    return os.path.join("/tmp", f"motion_detection_{uuid.uuid4().hex}{suffix}")


def normalize_odd_kernel(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def is_remote_path(path: Optional[str]) -> bool:
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))


def is_video_source_available(path: Optional[str]) -> bool:
    if not path:
        return False
    if is_remote_path(path):
        return True
    return os.path.exists(path)


def fit_size(width: int, height: int, max_dimension: int) -> Tuple[int, int]:
    width = int(width)
    height = int(height)
    max_dimension = max(64, int(max_dimension))

    if max(width, height) <= max_dimension:
        new_w, new_h = width, height
    elif width >= height:
        scale = max_dimension / float(width)
        new_w = max_dimension
        new_h = int(round(height * scale))
    else:
        scale = max_dimension / float(height)
        new_h = max_dimension
        new_w = int(round(width * scale))

    if new_w % 2 == 1:
        new_w -= 1
    if new_h % 2 == 1:
        new_h -= 1

    new_w = max(64, new_w)
    new_h = max(64, new_h)
    return new_w, new_h


def build_subtractor(
    method: str,
    history: int,
    mog2_var_threshold: float,
    knn_dist2_threshold: float,
    detect_shadows: bool,
):
    if method == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(
            history=int(history),
            varThreshold=float(mog2_var_threshold),
            detectShadows=bool(detect_shadows),
        )
    if method == "KNN":
        return cv2.createBackgroundSubtractorKNN(
            history=int(history),
            dist2Threshold=float(knn_dist2_threshold),
            detectShadows=bool(detect_shadows),
        )
    return None


def clean_mask(mask: np.ndarray, open_kernel: int, close_kernel: int, min_area: int) -> np.ndarray:
    cleaned = mask.copy()

    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if min_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        filtered = np.zeros_like(cleaned)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == label] = 255
        cleaned = filtered

    return cleaned


def process_video(
    video_path: Optional[str],
    method: str,
    history: int,
    mog2_var_threshold: float,
    knn_dist2_threshold: float,
    detect_shadows: bool,
    learning_rate: float,
    diff_threshold: int,
    blur_kernel: int,
    open_kernel: int,
    close_kernel: int,
    min_area: int,
    max_dimension: int,
    contour_thickness: int,
    max_frames: int,
    progress=gr.Progress(),
):
    if not video_path:
        video_path = DEFAULT_VIDEO_URL

    if not is_video_source_available(video_path):
        raise gr.Error("No video available.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the video.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(source_fps) if source_fps and source_fps > 0 else 25.0

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise gr.Error("Could not read the video dimensions.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width, output_height = fit_size(source_width, source_height, max_dimension)

    mask_video_path = make_temp_path(".mp4")
    overlay_video_path = make_temp_path(".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (output_width, output_height), True)
    overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (output_width, output_height), True)

    if not mask_writer.isOpened() or not overlay_writer.isOpened():
        cap.release()
        raise gr.Error("Could not create output videos.")

    blur_kernel = normalize_odd_kernel(blur_kernel)
    open_kernel = normalize_odd_kernel(open_kernel)
    close_kernel = normalize_odd_kernel(close_kernel)
    diff_threshold = max(1, int(diff_threshold))
    min_area = max(1, int(min_area))
    contour_thickness = max(1, int(contour_thickness))
    max_frames = max(1, int(max_frames))
    learning_rate = float(learning_rate)

    subtractor = build_subtractor(
        method=method,
        history=history,
        mog2_var_threshold=mog2_var_threshold,
        knn_dist2_threshold=knn_dist2_threshold,
        detect_shadows=detect_shadows,
    )

    previous_gray = None
    running_background = None
    processed_frames = 0

    while True:
        if processed_frames >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        if (frame.shape[1], frame.shape[0]) != (output_width, output_height):
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

        working = frame.copy()
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

        if blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if method in {"MOG2", "KNN"}:
            raw_mask = subtractor.apply(working, learningRate=learning_rate)
            threshold_value = 200 if detect_shadows else 1
            _, binary_mask = cv2.threshold(raw_mask, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "Frame Difference":
            if previous_gray is None:
                previous_gray = gray
                binary_mask = np.zeros_like(gray)
            else:
                delta = cv2.absdiff(gray, previous_gray)
                _, binary_mask = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)
                previous_gray = gray
        else:
            if running_background is None:
                running_background = gray.astype(np.float32)
                binary_mask = np.zeros_like(gray)
            else:
                cv2.accumulateWeighted(gray, running_background, learning_rate)
                background_u8 = cv2.convertScaleAbs(running_background)
                delta = cv2.absdiff(gray, background_u8)
                _, binary_mask = cv2.threshold(delta, diff_threshold, 255, cv2.THRESH_BINARY)

        binary_mask = clean_mask(
            mask=binary_mask,
            open_kernel=open_kernel,
            close_kernel=close_kernel,
            min_area=min_area,
        )

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = frame.copy()
        color_layer = np.zeros_like(overlay)
        color_layer[:, :, 1] = binary_mask
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.45, 0.0)

        object_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            object_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), contour_thickness)

        cv2.putText(
            overlay,
            f"Objects: {object_count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        mask_writer.write(mask_bgr)
        overlay_writer.write(overlay)

        processed_frames += 1
        progress_denominator = min(total_frames, max_frames) if total_frames > 0 else max_frames
        progress((processed_frames, progress_denominator), desc="Processing video")

    cap.release()
    mask_writer.release()
    overlay_writer.release()

    if processed_frames == 0:
        raise gr.Error("The video has no readable frames.")

    return mask_video_path, overlay_video_path


def update_method_visibility(method: str):
    return (
        gr.update(visible=(method == "MOG2")),
        gr.update(visible=(method == "KNN")),
        gr.update(visible=(method == "Frame Difference")),
        gr.update(visible=(method == "Running Average")),
    )


def run_wrapper(
    video_path,
    method,
    history_mog2,
    mog2_var_threshold,
    mog2_detect_shadows,
    mog2_learning_rate,
    history_knn,
    knn_dist2_threshold,
    knn_detect_shadows,
    knn_learning_rate,
    frame_diff_threshold,
    running_avg_learning_rate,
    running_avg_threshold,
    blur_kernel,
    open_kernel,
    close_kernel,
    min_area,
    max_dimension,
    contour_thickness,
    max_frames,
):
    if method == "MOG2":
        history = history_mog2
        current_mog2_var_threshold = mog2_var_threshold
        current_knn_dist2_threshold = 1
        detect_shadows = mog2_detect_shadows
        learning_rate = mog2_learning_rate
        diff_threshold = 1
    elif method == "KNN":
        history = history_knn
        current_mog2_var_threshold = 1
        current_knn_dist2_threshold = knn_dist2_threshold
        detect_shadows = knn_detect_shadows
        learning_rate = knn_learning_rate
        diff_threshold = 1
    elif method == "Frame Difference":
        history = 1
        current_mog2_var_threshold = 1
        current_knn_dist2_threshold = 1
        detect_shadows = False
        learning_rate = 0.01
        diff_threshold = frame_diff_threshold
    else:
        history = 1
        current_mog2_var_threshold = 1
        current_knn_dist2_threshold = 1
        detect_shadows = False
        learning_rate = running_avg_learning_rate
        diff_threshold = running_avg_threshold

    return process_video(
        video_path=video_path,
        method=method,
        history=history,
        mog2_var_threshold=current_mog2_var_threshold,
        knn_dist2_threshold=current_knn_dist2_threshold,
        detect_shadows=detect_shadows,
        learning_rate=learning_rate,
        diff_threshold=diff_threshold,
        blur_kernel=blur_kernel,
        open_kernel=open_kernel,
        close_kernel=close_kernel,
        min_area=min_area,
        max_dimension=max_dimension,
        contour_thickness=contour_thickness,
        max_frames=max_frames,
    )


with gr.Blocks(title="Motion Detection from Video") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(
                label="Input video",
                value=DEFAULT_VIDEO_URL,
                sources=["upload"],
                height=360,
            )

            with gr.Accordion("Settings", open=True):
                method = gr.Dropdown(
                    choices=["MOG2", "KNN", "Frame Difference", "Running Average"],
                    value="MOG2",
                    label="Method",
                )

                with gr.Group(visible=True) as mog2_group:
                    with gr.Row():
                        with gr.Column():
                            history_mog2 = gr.Slider(1, 10000, value=150, step=1, label="History")
                            mog2_var_threshold = gr.Slider(1, 10000, value=16, step=1, label="MOG2 variance threshold")
                        with gr.Column():
                            mog2_detect_shadows = gr.Checkbox(value=False, label="Detect shadows")
                            mog2_learning_rate = gr.Slider(0.0001, 1.0, value=0.01, step=0.0001, label="Learning rate")

                with gr.Group(visible=False) as knn_group:
                    with gr.Row():
                        with gr.Column():
                            history_knn = gr.Slider(1, 10000, value=150, step=1, label="History")
                            knn_dist2_threshold = gr.Slider(1, 10000, value=400, step=1, label="KNN distance threshold")
                        with gr.Column():
                            knn_detect_shadows = gr.Checkbox(value=False, label="Detect shadows")
                            knn_learning_rate = gr.Slider(0.0001, 1.0, value=0.01, step=0.0001, label="Learning rate")

                with gr.Group(visible=False) as frame_diff_group:
                    with gr.Row():
                        with gr.Column():
                            frame_diff_threshold = gr.Slider(1, 255, value=30, step=1, label="Difference threshold")
                        with gr.Column():
                            gr.Markdown("")

                with gr.Group(visible=False) as running_avg_group:
                    with gr.Row():
                        with gr.Column():
                            running_avg_learning_rate = gr.Slider(0.0001, 1.0, value=0.02, step=0.0001, label="Learning rate")
                            running_avg_threshold = gr.Slider(1, 255, value=30, step=1, label="Difference threshold")
                        with gr.Column():
                            gr.Markdown("")

                with gr.Row():
                    with gr.Column():
                        blur_kernel = gr.Slider(1, 101, value=5, step=1, label="Blur kernel size")
                        open_kernel = gr.Slider(1, 101, value=3, step=1, label="Opening kernel size")
                        close_kernel = gr.Slider(1, 101, value=7, step=1, label="Closing kernel size")
                    with gr.Column():
                        min_area = gr.Slider(1, 10000, value=400, step=1, label="Minimum object area")
                        max_dimension = gr.Slider(64, 10000, value=720, step=1, label="Maximum output dimension")
                        contour_thickness = gr.Slider(1, 100, value=2, step=1, label="Box thickness")
                        max_frames = gr.Slider(1, 10000, value=10000, step=1, label="Max frames to process")

            run_button = gr.Button("Process", variant="primary")

        with gr.Column(scale=1):
            mask_video = gr.Video(label="Foreground mask video", height=360)
            overlay_video = gr.Video(label="Detected motion video", height=360)

    method.change(
        fn=update_method_visibility,
        inputs=method,
        outputs=[mog2_group, knn_group, frame_diff_group, running_avg_group],
    )

    run_button.click(
        fn=run_wrapper,
        inputs=[
            input_video,
            method,
            history_mog2,
            mog2_var_threshold,
            mog2_detect_shadows,
            mog2_learning_rate,
            history_knn,
            knn_dist2_threshold,
            knn_detect_shadows,
            knn_learning_rate,
            frame_diff_threshold,
            running_avg_learning_rate,
            running_avg_threshold,
            blur_kernel,
            open_kernel,
            close_kernel,
            min_area,
            max_dimension,
            contour_thickness,
            max_frames,
        ],
        outputs=[mask_video, overlay_video],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()