import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False


DEFAULT_VIDEO_URL = "https://videos.pexels.com/video-files/6077402/6077402-uhd_4096_2160_25fps.mp4"


# -----------------------------------------------------------------------------
# Documentation utilities
# -----------------------------------------------------------------------------


@st.cache_data
def read_text_file(filename: str, fallback: str = "") -> str:
    app_dir = Path(__file__).resolve().parent
    file_path = app_dir / filename
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fallback


@st.cache_data
def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        title = lines[0].strip()

        if title.lower() in {"table des matières", "table of contents"}:
            continue

        sections[title] = "## " + part

    if not sections and markdown_text.strip():
        sections["Documentation"] = markdown_text.strip()

    return sections


DOCUMENTATION_fr = read_text_file(
    "documentation_fr.md",
    "## Documentation FR\n\nThe file `documentation_fr.md` was not found.",
)
DOCUMENTATION_en = read_text_file(
    "documentation_en.md",
    "## Documentation EN\n\nThe file `documentation_en.md` was not found.",
)

DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)


# -----------------------------------------------------------------------------
# Video processing utilities
# -----------------------------------------------------------------------------


def make_temp_path(suffix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"motion_detection_{uuid.uuid4().hex}{suffix}")


def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = make_temp_path(suffix)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


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


class BrowserVideoWriter:
    """Write browser-playable MP4 files.

    The preferred backend is imageio/ffmpeg with software H.264 encoding.
    This avoids OpenCV trying to use unavailable hardware encoders such as
    h264_v4l2m2m. If imageio is not installed, an OpenCV mp4v fallback is used.
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        self.path = path
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.size = (int(size[0]), int(size[1]))
        self.backend = None
        self.writer = None

        if IMAGEIO_AVAILABLE:
            try:
                self.writer = imageio.get_writer(
                    path,
                    format="FFMPEG",
                    mode="I",
                    fps=self.fps,
                    codec="libx264",
                    macro_block_size=2,
                    output_params=[
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                    ],
                )
                self.backend = "imageio"
                return
            except Exception:
                self.writer = None
                self.backend = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.size, True)
        if self.writer.isOpened():
            self.backend = "opencv"
        else:
            self.writer.release()
            self.writer = None

    def is_opened(self) -> bool:
        return self.writer is not None

    def write_bgr(self, frame_bgr: np.ndarray) -> None:
        if self.backend == "imageio":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.writer.append_data(frame_rgb)
        elif self.backend == "opencv":
            self.writer.write(frame_bgr)
        else:
            raise RuntimeError("Video writer is not opened.")

    def release(self) -> None:
        if self.writer is None:
            return
        if self.backend == "imageio":
            self.writer.close()
        else:
            self.writer.release()
        self.writer = None


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
    progress_bar=None,
    progress_text=None,
) -> tuple[str, str, str]:
    if not video_path:
        video_path = DEFAULT_VIDEO_URL

    if not is_video_source_available(video_path):
        raise ValueError("No video available.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(source_fps) if source_fps and source_fps > 0 else 25.0

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise ValueError("Could not read the video dimensions.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width, output_height = fit_size(source_width, source_height, max_dimension)

    mask_video_path = make_temp_path(".mp4")
    overlay_video_path = make_temp_path(".mp4")

    mask_writer = BrowserVideoWriter(mask_video_path, fps, (output_width, output_height))
    overlay_writer = BrowserVideoWriter(overlay_video_path, fps, (output_width, output_height))

    if not mask_writer.is_opened() or not overlay_writer.is_opened():
        cap.release()
        mask_writer.release()
        overlay_writer.release()
        raise ValueError(
            "Could not create output videos. Install `imageio` and `imageio-ffmpeg`, "
            "or verify that OpenCV can write MP4 files on this system."
        )

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
    progress_denominator = min(total_frames, max_frames) if total_frames > 0 else max_frames

    try:
        while True:
            if processed_frames >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if (frame.shape[1], frame.shape[0]) != (output_width, output_height):
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

            working = frame.copy()
            if blur_kernel > 1:
                working = cv2.GaussianBlur(working, (blur_kernel, blur_kernel), 0)

            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

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
            mask_writer.write_bgr(mask_bgr)
            overlay_writer.write_bgr(overlay)

            processed_frames += 1
            if progress_bar is not None:
                progress_bar.progress(
                    min(1.0, processed_frames / float(progress_denominator)),
                    text=f"Processing video: {processed_frames}/{progress_denominator} frames",
                )
            if progress_text is not None:
                progress_text.caption(f"Processed frames: {processed_frames}")
    finally:
        cap.release()
        mask_writer.release()
        overlay_writer.release()

    if processed_frames == 0:
        raise ValueError("The video has no readable frames.")

    backend = mask_writer.backend or "unknown"
    message = f"Processed {processed_frames} frame(s). Output backend: {backend}."
    return mask_video_path, overlay_video_path, message


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
    progress_bar=None,
    progress_text=None,
):
    if method == "MOG2":
        history = history_mog2
        current_mog2_var_threshold = mog2_var_threshold
        current_knn_dist2_threshold = 400
        detect_shadows = mog2_detect_shadows
        learning_rate = mog2_learning_rate
        diff_threshold = 30
    elif method == "KNN":
        history = history_knn
        current_mog2_var_threshold = 16
        current_knn_dist2_threshold = knn_dist2_threshold
        detect_shadows = knn_detect_shadows
        learning_rate = knn_learning_rate
        diff_threshold = 30
    elif method == "Frame Difference":
        history = 1
        current_mog2_var_threshold = 16
        current_knn_dist2_threshold = 400
        detect_shadows = False
        learning_rate = 0.01
        diff_threshold = frame_diff_threshold
    else:
        history = 1
        current_mog2_var_threshold = 16
        current_knn_dist2_threshold = 400
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
        progress_bar=progress_bar,
        progress_text=progress_text,
    )


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def initialise_session_state() -> None:
    defaults = {
        "mask_video_path": None,
        "overlay_video_path": None,
        "status_message": "Ready",
        "status_kind": "info",
        "selected_doc_fr": None,
        "selected_doc_en": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_status(message: str, kind: str = "info") -> None:
    st.session_state.status_message = message
    st.session_state.status_kind = kind


def show_status() -> None:
    kind = st.session_state.status_kind
    message = st.session_state.status_message
    if kind == "error":
        st.error(message)
    elif kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.info(message)


def render_video_player(path: str, label: str) -> None:
    st.markdown(f"#### {label}")
    with open(path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes, format="video/mp4")
    st.download_button(
        label=f"Download {label}",
        data=video_bytes,
        file_name=f"{label.lower().replace(' ', '_')}.mp4",
        mime="video/mp4",
        width="stretch",
    )


def render_app_tab() -> None:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_video = st.file_uploader(
            "Input video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
        )

        if uploaded_video is None:
            st.caption("No uploaded file. The default remote video will be used.")
            st.video(DEFAULT_VIDEO_URL)
            input_video_path = DEFAULT_VIDEO_URL
        else:
            st.video(uploaded_video)
            input_video_path = save_uploaded_video(uploaded_video)

        with st.expander("Settings", expanded=True):
            method = st.selectbox(
                "Method",
                ["MOG2", "KNN", "Frame Difference", "Running Average"],
                index=0,
            )

            history_mog2 = 150
            mog2_var_threshold = 16
            mog2_detect_shadows = False
            mog2_learning_rate = 0.01
            history_knn = 150
            knn_dist2_threshold = 400
            knn_detect_shadows = False
            knn_learning_rate = 0.01
            frame_diff_threshold = 30
            running_avg_learning_rate = 0.02
            running_avg_threshold = 30

            if method == "MOG2":
                st.markdown("##### MOG2 parameters")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    history_mog2 = st.slider("History", 1, 500, value=150, step=1)
                    mog2_var_threshold = st.slider("MOG2 variance threshold", 4, 250, value=16, step=1)
                with m_col2:
                    mog2_detect_shadows = st.checkbox("Detect shadows", value=False)
                    mog2_learning_rate = st.slider("Learning rate", 0.0001, 1.0, value=0.01, step=0.0001, format="%.4f")

            elif method == "KNN":
                st.markdown("##### KNN parameters")
                k_col1, k_col2 = st.columns(2)
                with k_col1:
                    history_knn = st.slider("History", 1, 500, value=150, step=1)
                    knn_dist2_threshold = st.slider("KNN distance threshold", 10, 2000, value=400, step=10)
                with k_col2:
                    knn_detect_shadows = st.checkbox("Detect shadows", value=False)
                    knn_learning_rate = st.slider("Learning rate", 0.0001, 1.0, value=0.01, step=0.0001, format="%.4f")

            elif method == "Frame Difference":
                st.markdown("##### Frame Difference parameters")
                frame_diff_threshold = st.slider("Difference threshold", 1, 255, value=30, step=1)

            else:
                st.markdown("##### Running Average parameters")
                running_avg_learning_rate = st.slider("Learning rate", 0.0001, 1.0, value=0.02, step=0.0001, format="%.4f")
                running_avg_threshold = st.slider("Difference threshold", 1, 255, value=30, step=1)

            st.markdown("##### Common post-processing parameters")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                blur_kernel = st.slider("Blur kernel size", 1, 51, value=5, step=2)
                open_kernel = st.slider("Opening kernel size", 1, 51, value=3, step=2)
                close_kernel = st.slider("Closing kernel size", 1, 51, value=7, step=2)
            with c_col2:
                min_area = st.slider("Minimum object area", 1, 10000, value=400, step=1)
                max_dimension = st.slider("Maximum output dimension", 64, 1920, value=720, step=1)
                contour_thickness = st.slider("Bounding box thickness", 1, 10, value=2, step=1)
                max_frames = st.slider("Max frames to process", 1, 10000, value=10000, step=1)

        process_clicked = st.button("Process", type="primary", width="stretch")

        if process_clicked:
            progress_bar = st.progress(0.0, text="Processing video")
            progress_text = st.empty()
            try:
                mask_path, overlay_path, message = run_wrapper(
                    video_path=input_video_path,
                    method=method,
                    history_mog2=history_mog2,
                    mog2_var_threshold=mog2_var_threshold,
                    mog2_detect_shadows=mog2_detect_shadows,
                    mog2_learning_rate=mog2_learning_rate,
                    history_knn=history_knn,
                    knn_dist2_threshold=knn_dist2_threshold,
                    knn_detect_shadows=knn_detect_shadows,
                    knn_learning_rate=knn_learning_rate,
                    frame_diff_threshold=frame_diff_threshold,
                    running_avg_learning_rate=running_avg_learning_rate,
                    running_avg_threshold=running_avg_threshold,
                    blur_kernel=blur_kernel,
                    open_kernel=open_kernel,
                    close_kernel=close_kernel,
                    min_area=min_area,
                    max_dimension=max_dimension,
                    contour_thickness=contour_thickness,
                    max_frames=max_frames,
                    progress_bar=progress_bar,
                    progress_text=progress_text,
                )
                st.session_state.mask_video_path = mask_path
                st.session_state.overlay_video_path = overlay_path
                set_status(message, "success")
            except Exception as error:
                st.session_state.mask_video_path = None
                st.session_state.overlay_video_path = None
                set_status(str(error), "error")

    with right_col:
        if st.session_state.mask_video_path and st.session_state.overlay_video_path:
            render_video_player(st.session_state.mask_video_path, "Foreground mask")
            render_video_player(st.session_state.overlay_video_path, "Motion overlay")
        else:
            st.markdown("#### Foreground mask")
            st.info("Process a video to display the foreground mask.")
            st.markdown("#### Motion overlay")
            st.info("Process a video to display the motion overlay.")

        show_status()


def render_documentation_tab(sections: dict[str, str], state_key: str) -> None:
    titles = list(sections.keys())
    if not titles:
        st.warning("No documentation section found.")
        return

    if st.session_state[state_key] not in titles:
        st.session_state[state_key] = titles[0]

    button_col, markdown_col = st.columns([1, 2])

    with button_col:
        for title in titles:
            button_type = "primary" if st.session_state[state_key] == title else "secondary"
            if st.button(title, key=f"{state_key}_{title}", type=button_type, width="stretch"):
                st.session_state[state_key] = title

    with markdown_col:
        st.markdown(sections[st.session_state[state_key]], unsafe_allow_html=False)


def main() -> None:
    st.set_page_config(page_title="Video Motion Detection", layout="wide")
    initialise_session_state()

    app_tab, doc_fr_tab, doc_en_tab = st.tabs(["App", "Documentation FR", "Documentation EN"])

    with app_tab:
        render_app_tab()

    with doc_fr_tab:
        render_documentation_tab(DOC_FR_SECTIONS, "selected_doc_fr")

    with doc_en_tab:
        render_documentation_tab(DOC_EN_SECTIONS, "selected_doc_en")


if __name__ == "__main__":
    main()
