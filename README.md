---
title: Video Motion Detection
emoji: 🏃
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Video motion detection with background subtraction and objec
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Video Motion Detection

This repository contains an interactive video motion detection mini app based on classical background subtraction and frame differencing methods.

The app is designed as an educational and portfolio demo for video processing and computer vision. It detects moving regions in a video, generates a binary foreground mask, and overlays the detected moving objects on the original frames.

A Streamlit deployment is available here:

https://video-motion-detection.streamlit.app/

## Main features

- Upload a video or use the default remote video example.
- Detect moving objects with classical computer vision methods.
- Generate a foreground mask video.
- Generate a motion overlay video with highlighted moving regions.
- Draw bounding boxes around detected connected components.
- Tune method-specific parameters directly from the interface.
- Apply common preprocessing and post-processing operations.
- Download the generated mask and overlay videos.
- Read the English and French documentation tabs.

## Implemented methods

The app includes four motion detection methods.

### MOG2

MOG2 is a Mixture of Gaussians background subtraction method.

Each pixel is modelled using a mixture of Gaussian distributions, allowing the background model to represent multi-modal variations such as flickering leaves, water motion, or repetitive background changes.

The app exposes parameters such as history length, variance threshold, shadow detection, and learning rate.

### KNN

KNN background subtraction compares recent pixel samples and decides whether a new pixel value belongs to the background distribution.

It is another adaptive background modelling method available in OpenCV and can be useful in scenes where the background changes gradually.

The app exposes parameters such as history length, distance threshold, shadow detection, and learning rate.

### Frame Difference

Frame Difference is the simplest motion detection baseline.

It computes the absolute difference between the current grayscale frame and the previous grayscale frame. Pixels with a difference above a threshold are detected as moving.

This method is fast and intuitive, but it mainly detects changes between consecutive frames and may produce hollow masks for uniform moving objects.

### Running Average

Running Average maintains an adaptive background image using an exponential moving average.

The current frame is compared with this estimated background. The learning rate controls how quickly the background adapts to changes.

This method is more stable than simple frame differencing but can absorb slowly moving or stationary objects into the background over time.

## Processing pipeline

The app follows a classical video processing pipeline:

1. Load the input video.
2. Resize frames to a chosen maximum output dimension.
3. Apply Gaussian blur to reduce noise and compression artefacts.
4. Convert frames to grayscale when required by the method.
5. Estimate a foreground mask using the selected motion detection method.
6. Clean the mask with morphological opening and closing.
7. Remove small connected components using an area threshold.
8. Extract contours and draw bounding boxes.
9. Save two output videos:
   - a binary foreground mask,
   - a motion overlay with highlighted moving regions.

## Parameters

### Method-specific parameters

Depending on the selected method, the interface exposes:

- history length,
- variance threshold for MOG2,
- distance threshold for KNN,
- learning rate,
- difference threshold,
- shadow detection.

### Common post-processing parameters

The following parameters are shared across methods:

- blur kernel size,
- opening kernel size,
- closing kernel size,
- minimum object area,
- maximum output dimension,
- bounding box thickness,
- maximum number of frames to process.

These parameters make it possible to balance sensitivity, robustness, computational cost, and visual clarity.

## Repository structure

```text
.
├── app.py                 # Gradio / Hugging Face Space entry point
├── app_sl.py              # Streamlit version of the app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # License file
└── README.md              # Repository and Hugging Face Space description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/video_motion_detection.git
cd video_motion_detection
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If needed, install the main dependencies manually:

```bash
pip install gradio streamlit numpy opencv-python-headless imageio imageio-ffmpeg
```

## Run the Gradio app

```bash
python app.py
```

The local interface will usually be available at:

```text
http://127.0.0.1:7860
```

## Run the Streamlit app

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Hugging Face Space notes

The YAML block at the top of this README is used by Hugging Face Spaces.

The current metadata launches the Gradio version:

```yaml
sdk: gradio
app_file: app.py
```

If you want Hugging Face to launch the Streamlit version instead, update the metadata to:

```yaml
sdk: streamlit
app_file: app_sl.py
```

In that case, make sure `streamlit` is included in `requirements.txt`.

## Notes on video output

The Streamlit version writes browser-playable MP4 videos using an `imageio` and FFmpeg backend when available. This avoids common OpenCV issues with unavailable hardware H.264 encoders on online platforms.

For online deployment, short videos or a limited number of processed frames are recommended to keep execution time reasonable.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation.
- `documentation_fr.md` for the French documentation.

These files explain the motion detection problem, grayscale conversion, Gaussian blur, background subtraction, frame difference, running average, MOG2, KNN, shadow detection, mathematical morphology, connected components, contour detection, bounding boxes, and parameter interpretation.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image, and computer vision mini apps.
