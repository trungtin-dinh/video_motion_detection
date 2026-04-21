---

## Table of Contents

1. [Problem Statement and Context](#1-problem-statement-and-context)
2. [Preprocessing: Grayscale Conversion and Gaussian Blur](#2-preprocessing-grayscale-conversion-and-gaussian-blur)
3. [Background Subtraction: General Formulation](#3-background-subtraction-general-formulation)
4. [Method 1 — Frame Difference](#4-method-1--frame-difference)
5. [Method 2 — Running Average](#5-method-2--running-average)
6. [Method 3 — MOG2: Mixture of Gaussians](#6-method-3--mog2-mixture-of-gaussians)
7. [Method 4 — KNN: K-Nearest Neighbours Background Subtractor](#7-method-4--knn-k-nearest-neighbours-background-subtractor)
8. [Shadow Detection](#8-shadow-detection)
9. [Post-processing: Mathematical Morphology](#9-post-processing-mathematical-morphology)
10. [Connected Components and Area Filtering](#10-connected-components-and-area-filtering)
11. [Contour Detection and Bounding Boxes](#11-contour-detection-and-bounding-boxes)
12. [Parameter Guide](#12-parameter-guide)

---

## 1. Problem Statement and Context

Motion detection is the task of automatically identifying **which pixels in a video frame belong to a moving object**, as opposed to a static background. It is a foundational problem in computer vision with applications in surveillance, traffic monitoring, human-computer interaction, sports analytics, and robotics.

Formally, consider a video as a discrete sequence of frames $I_t : \Omega \to \mathbb{R}^C$, where $\Omega \subset \mathbb{Z}^2$ is the spatial pixel grid, $t \in \mathbb{N}$ is the frame index, and $C = 3$ for a colour image (BGR channels). The goal is to compute, for each frame $I_t$, a **binary foreground mask**:

$$M_t(x, y) = \begin{cases} 255 & \text{if pixel } (x,y) \text{ belongs to a moving object} \\ 0 & \text{otherwise} \end{cases}$$

This is harder than it looks. The background is rarely perfectly static: lighting changes, swaying trees, camera vibration, and gradual illumination drift all modify background pixels over time. A robust detector must distinguish these *nuisance* changes from genuine object motion.

The four methods implemented in this app span a progression from the simplest possible approach (frame difference) to learned statistical models (MOG2, KNN), representing the main families of classical background subtraction algorithms.

---

## 2. Preprocessing: Grayscale Conversion and Gaussian Blur

### 2.1 Grayscale Conversion

All four methods operate on **single-channel intensity images**. Each colour frame $I_t$ is converted to a grayscale image $G_t : \Omega \to [0, 255]$ using the ITU-R BT.601 luminance formula:

$$G_t(x,y) = 0.299 \cdot R(x,y) + 0.587 \cdot V(x,y) + 0.114 \cdot B(x,y)$$

where $R$, $V$, $B$ are the red, green, and blue channel values respectively. The coefficients reflect the non-uniform sensitivity of human vision to each primary colour, green being the dominant perceptual contributor.

Working in grayscale reduces computational cost by a factor of 3 and is sufficient because motion manifests primarily as a change in *intensity*, not hue.

### 2.2 Gaussian Blur

Before any differencing operation is applied, each grayscale frame is convolved with a **Gaussian kernel** to suppress high-frequency noise:

$$\tilde{G}_t = G_t * h_\sigma$$

where the 2D Gaussian kernel is:

$$h_\sigma(u, v) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{u^2 + v^2}{2\sigma^2}\right)$$

In practice, a discrete kernel of size $(k \times k)$ is used, with $k$ odd and $\sigma$ implicitly set by OpenCV as $\sigma = 0.3 \cdot \frac{k-1}{2} + 0.8$. The blur radius $k$ is exposed as the **Blur kernel size** parameter.

**Why blur?** Sensor noise and compression artefacts introduce isolated pixels that flicker between frames. Without blurring, these would produce thousands of false positive foreground detections. Blurring trades a small amount of spatial resolution for dramatically improved signal-to-noise ratio on the difference image.

> **Important**: the Gaussian blur is applied **only in the Frame Difference and Running Average methods**, where pixel-level differences are computed manually. The MOG2 and KNN subtractors perform their own internal statistical noise handling, so blur is applied to the frame before passing it to these methods as well, providing consistent preprocessing across all modes.

---

## 3. Background Subtraction: General Formulation

All four methods instantiate the same high-level paradigm: maintain (implicitly or explicitly) an estimate of the **background model** $B_t(x,y)$, and declare a pixel as foreground if it deviates sufficiently from this estimate:

$$M_t(x,y) = \mathbf{1}\!\left[\,d\!\left(G_t(x,y),\; B_t(x,y)\right) > \tau\,\right]$$

where $d(\cdot, \cdot)$ is a distance measure, $\tau$ is a threshold, and $\mathbf{1}[\cdot]$ is the indicator function. The four methods differ in (i) how $B_t$ is represented, (ii) how $d$ is defined, and (iii) how the model is updated over time.

---

## 4. Method 1 — Frame Difference

### 4.1 Principle

Frame difference is the most direct approach: the background model at time $t$ is simply **the previous frame** $G_{t-1}$. The difference image is:

$$D_t(x,y) = \left| G_t(x,y) - G_{t-1}(x,y) \right|$$

A pixel is declared foreground if this absolute difference exceeds a threshold $\tau$:

$$M_t(x,y) = \mathbf{1}\!\left[D_t(x,y) > \tau\right]$$

### 4.2 Properties and Limitations

Frame difference is extremely fast and has zero memory overhead (only one previous frame is stored). However, it suffers from a fundamental structural problem known as the **double-edge effect** or **aperture problem of difference images**. Consider a solid-coloured moving rectangle: only its leading and trailing edges change between frames, while its interior (if the object is uniform) produces a zero difference. The resulting mask is a hollow shell of the object's contour, not a filled silhouette.

Mathematically, the mask $M_t$ approximates the **temporal gradient** of the video:

$$M_t \approx \mathbf{1}\!\left[\left|\frac{\partial I}{\partial t}(x,y,t)\right| > \tau\right]$$

This is large only where intensity is *changing*, not necessarily where an object *is*. For objects with textured interiors, this produces better fills. For smooth or uniform objects, the interior goes undetected.

A second limitation is sensitivity to rapid camera shake or global illumination changes: any global shift in brightness affects every pixel simultaneously, producing a near-total false-positive mask. This method is appropriate only for stationary cameras and scenes with stable lighting.

The **Difference threshold** $\tau$ (slider range 1–255) controls the sensitivity. A low value detects subtle motion but increases false positives from noise; a high value requires strong contrast at moving edges.

---

## 5. Method 2 — Running Average

### 5.1 Exponential Moving Average Background Model

Rather than using only the immediately preceding frame, the Running Average method maintains a **continuous, adaptive estimate** of the background by computing an exponentially weighted average of all past frames:

$$B_t(x,y) = (1 - \alpha)\, B_{t-1}(x,y) + \alpha\, G_t(x,y)$$

where $\alpha \in (0, 1)$ is the **learning rate**, a critical hyperparameter. Solving this recurrence relation gives the closed-form expression:

$$B_t(x,y) = \alpha \sum_{k=0}^{t} (1-\alpha)^{t-k}\, G_k(x,y)$$

This is an **Infinite Impulse Response (IIR) low-pass filter** applied along the temporal axis. Each past frame contributes with exponentially decaying weight, so recent frames have more influence on the background estimate than older ones. The effective temporal memory of this filter — the number of frames that contribute substantially — scales as $1/\alpha$.

### 5.2 Foreground Detection

Once $B_t$ is computed, the detection step mirrors frame difference:

$$D_t(x,y) = \left| G_t(x,y) - B_t(x,y) \right|, \qquad M_t(x,y) = \mathbf{1}\!\left[D_t(x,y) > \tau\right]$$

### 5.3 Effect of the Learning Rate

The learning rate $\alpha$ governs the trade-off between **adaptability** and **stability**:

- **High $\alpha$** (e.g., 0.5): the background model adapts quickly to changes, which helps in scenes with progressive illumination changes. However, a foreground object that stays still long enough will be absorbed into the background model, causing it to disappear from the mask.
- **Low $\alpha$** (e.g., 0.001): the background is very stable and a stopped object remains detected as foreground for a long time. The downside is slow adaptation to genuine background changes (e.g., a light being switched on).

This fundamental tension between **persistence** and **plasticity** is a universal challenge in adaptive background modelling. In the frequency domain, the running average is a first-order low-pass filter with a 3 dB cut-off at $f_c \approx \alpha / (2\pi)$ frames$^{-1}$.

### 5.4 Comparison with Frame Difference

Unlike frame difference, the running average background model integrates information over many frames. Its $D_t$ image corresponds more closely to the **displacement** of the object from its long-term average position, rather than the instantaneous inter-frame change. This gives better filled masks for slowly moving or briefly stopped objects, at the cost of a warm-up period: the first frames of the video produce an unreliable background estimate since $B_t$ has not yet converged.

---

## 6. Method 3 — MOG2: Mixture of Gaussians

### 6.1 Motivation: Per-Pixel Distribution Modelling

Both of the preceding methods represent the background at each pixel as a **single value**. This is fragile in the presence of *multi-modal* background behaviour: consider a pixel that alternates between a bright sky and the leaves of a swaying tree. No single value adequately represents this pixel's background state.

The Mixture of Gaussians (MoG) family of methods, and MOG2 in particular (Zivkovic, 2004; Zivkovic & van der Heijden, 2006), models each pixel's intensity as a **mixture of $K$ Gaussian distributions**:

$$p(x_{t}) = \sum_{k=1}^{K} w_{k,t} \cdot \mathcal{N}(x_t;\, \mu_{k,t},\, \sigma_{k,t}^2)$$

where $w_{k,t}$ is the weight (mixing coefficient) of the $k$-th Gaussian at time $t$, $\mu_{k,t}$ is its mean, and $\sigma_{k,t}^2$ is its variance. Note that in OpenCV's implementation, a scalar variance (isotropic) is used per component for efficiency, rather than a full covariance matrix.

### 6.2 Background and Foreground Classification

Not all $K$ components represent the background. At each time step, the components are sorted by the ratio $w_{k}/\sigma_{k}$, which heuristically ranks components by how "stable and frequent" they are — a narrow, high-weight Gaussian corresponds to a stable, repeating background state. The first $B$ components whose cumulative weight exceeds a threshold represent the background:

$$B = \arg\min_b \left\{ \sum_{k=1}^{b} w_k > T_{bg} \right\}$$

A new observation $x_t$ at pixel $(x,y)$ is classified as **background** if it falls within $2.5\sigma$ of at least one of the background components (Mahalanobis distance test):

$$\text{foreground}(x_t) = \mathbf{1}\!\left[\, \nexists\, k \leq B : \frac{(x_t - \mu_k)^2}{\sigma_k^2} < \lambda \,\right]$$

where $\lambda$ is the **variance threshold** (the MOG2 variance threshold parameter). Increasing $\lambda$ makes the classifier more tolerant of deviation from the background model, reducing false positives at the cost of sensitivity.

### 6.3 Online Model Update

The MOG2 model is updated online with every new frame. For the matched component $k^*$, the update equations are:

$$w_{k^*,t} \leftarrow (1 - \alpha)\, w_{k^*,t} + \alpha$$

$$\mu_{k^*,t} \leftarrow (1 - \rho)\, \mu_{k^*,t} + \rho\, x_t$$

$$\sigma_{k^*,t}^2 \leftarrow (1 - \rho)\, \sigma_{k^*,t}^2 + \rho\, (x_t - \mu_{k^*,t})^2$$

where $\rho = \alpha \cdot \mathcal{N}(x_t; \mu_{k^*}, \sigma_{k^*}^2)$ is a per-component learning rate scaled by how well the observation fits the component. All other weights are decreased: $w_{k \neq k^*} \leftarrow (1-\alpha)\, w_{k}$. If no component matches, the least probable component is replaced with a new one initialised at $x_t$.

The **learning rate** $\alpha$ is equivalent to $1/H$ where $H$ is the **history** parameter: it controls how many past frames effectively contribute to the model. A history of 150 means the model has a temporal memory of roughly 150 frames.

### 6.4 Advantages of MOG2

MOG2 handles complex backgrounds robustly due to the multi-modal representation: each pixel can maintain several distinct background states simultaneously. It naturally handles gradual illumination changes (slow shift in $\mu_k$) and periodic background motion such as foliage (a dedicated component for each state). Its main cost is computational and memory overhead: $K$ Gaussians must be maintained for every pixel, with $K$ typically between 3 and 5.

---

## 7. Method 4 — KNN: K-Nearest Neighbours Background Subtractor

### 7.1 Non-Parametric Background Modelling

KNN background subtraction (Zivkovic & van der Heijden, 2006) takes a fundamentally different, **non-parametric** approach. Instead of fitting Gaussian distributions, it maintains for each pixel a **sample set** $\mathcal{S}_{t} = \{s_1, s_2, \ldots, s_N\}$ of the $N$ most recent intensity values observed at that pixel under background conditions.

The background model is not a closed-form probability density but an empirical distribution represented directly by samples. Classification of a new observation $x_t$ is based on its **nearest neighbours** in $\mathcal{S}_t$: if at least $k$ of the $N$ stored samples are closer to $x_t$ than a distance threshold $d^2$, the pixel is classified as background:

$$\text{background}(x_t) = \mathbf{1}\!\left[\left|\left\{s_i \in \mathcal{S}_t : (x_t - s_i)^2 \leq d^2\right\}\right| \geq k\right]$$

The **KNN distance threshold** $d^2$ is the squared Euclidean distance in intensity space. A pixel is foreground if fewer than $k$ background samples are within distance $d$ of the current observation.

### 7.2 Comparison with MOG2

| Property | MOG2 | KNN |
|---|---|---|
| Background representation | Parametric (mixture of Gaussians) | Non-parametric (sample set) |
| Multi-modal support | Yes ($K$ components) | Yes (implicitly, via diverse samples) |
| Complexity per pixel | $O(K)$ | $O(N)$ |
| Adapts to new distributions | Gradually via weight update | Immediately when sample is replaced |
| Main parameter | Variance threshold | Distance threshold $d^2$ |

KNN tends to outperform MOG2 in scenes with very complex, non-Gaussian background dynamics (e.g., water surfaces, flames, screens) because it makes no distributional assumptions. MOG2 is generally faster for small $K$ and can be more numerically stable.

### 7.3 Sample Update

The sample set is updated by replacing old samples with new background observations at a rate governed by the learning rate. Because only confirmed background pixels contribute new samples, the model is self-reinforcing: once the initial background is learned, spurious foreground events do not corrupt the background model.

---

## 8. Shadow Detection

Both MOG2 and KNN include an optional **shadow detection** step, activated by the *Detect shadows* checkbox. This addresses a common failure mode: the shadow cast by a moving object is darker than the background but is not part of the object itself. Without shadow detection, shadows are included in the foreground mask, making bounding boxes much larger than the actual object.

### 8.1 Shadow Model

Shadow detection uses a simple but effective colour model (Prati et al., 2003). A pixel $(x,y)$ is classified as a **shadow** (rather than foreground) if:

$$\alpha_{shadow} \leq \frac{I_t(x,y)}{B_t(x,y)} \leq \beta_{shadow} \qquad \text{and} \qquad \left|\arg(I_t) - \arg(B_t)\right| < \tau_{hue}$$

where $\alpha_{shadow}$ and $\beta_{shadow}$ bound the acceptable ratio of current intensity to background intensity (a shadow darkens but does not dramatically change colour), and the angular difference in the hue channel must be small. In the HSV colour space, a shadow mainly reduces the $V$ (value) channel while leaving $H$ (hue) largely unchanged.

### 8.2 Output Encoding

When shadow detection is active, the raw mask returned by the subtractor uses a **tri-value encoding**:
- $0$: background
- $127$: shadow
- $255$: foreground

After the subtractor, a threshold is applied at value 200 to binarize the mask, thereby discarding shadow pixels (127) and keeping only true foreground (255). When shadow detection is disabled, the threshold is set to 1 (any non-zero value is foreground).

---

## 9. Post-processing: Mathematical Morphology

The raw binary mask $M_t$ produced by any of the four detection methods typically contains **noise**: isolated spurious foreground pixels due to sensor noise or compression artefacts, and small gaps or holes within the detected objects due to low-texture regions. Mathematical morphology is applied to clean the mask before contour detection.

### 9.1 Structuring Element

All morphological operations use an **elliptical structuring element** $\mathcal{B}$ of size $k \times k$:

$$\mathcal{B} = \{(u,v) : (2u/k)^2 + (2v/k)^2 \leq 1\}$$

The ellipse is preferred over a square because it is more isotropic: it avoids introducing artificial square artefacts into the mask geometry, and is less aggressive at the corners of diagonal structures.

### 9.2 Erosion and Dilation

The two elementary morphological operations are:

**Erosion** — a pixel $(x,y)$ in the eroded image is 1 only if *all* pixels in the neighbourhood $\mathcal{B}$ centred at $(x,y)$ are 1 in the original mask:

$$(\text{Erosion: } M \ominus \mathcal{B})(x,y) = \min_{(u,v) \in \mathcal{B}} M(x+u, y+v)$$

**Dilation** — a pixel $(x,y)$ in the dilated image is 1 if *at least one* pixel in the neighbourhood is 1:

$$(\text{Dilation: } M \oplus \mathcal{B})(x,y) = \max_{(u,v) \in \mathcal{B}} M(x+u, y+v)$$

### 9.3 Opening (Noise Removal)

**Morphological opening** is defined as erosion followed by dilation:

$$M_{\text{open}} = (M \ominus \mathcal{B}) \oplus \mathcal{B}$$

Geometrically, opening removes all foreground structures that are **smaller than the structuring element**: isolated pixels, thin filaments, and small blobs are erased. Larger connected regions are preserved but their boundaries are slightly smoothed. This is the primary operation for suppressing false positives caused by sensor noise.

The **Opening kernel size** parameter controls the size of $\mathcal{B}$. A value of 1 disables this operation. Increasing it is aggressive: foreground regions smaller than the kernel radius are completely removed.

### 9.4 Closing (Gap Filling)

**Morphological closing** is dilation followed by erosion:

$$M_{\text{close}} = (M \oplus \mathcal{B}) \ominus \mathcal{B}$$

Closing fills **small holes and gaps** within foreground regions: dark spots inside a moving object (caused by uniform-coloured patches that produce small differences between frames) are filled in. It also merges nearby foreground fragments into a single coherent region.

The **Closing kernel size** parameter controls the extent of gap filling. A large closing kernel will merge nearby but distinct objects into a single detection.

**Processing order**: opening is applied first, then closing. This is the standard sequence: first remove noise (opening), then fill the resulting cleaned mask (closing). Reversing the order would fill holes first and then potentially reopen them.

---

## 10. Connected Components and Area Filtering

After morphological post-processing, the binary mask $M_t$ may still contain residual small connected blobs that passed through the morphological filter. A second stage of filtering is based on **connected component analysis**.

### 10.1 Connected Components

A **connected component** in a binary image is a maximal set of foreground pixels that are all mutually reachable by a path of adjacent foreground pixels. **8-connectivity** is used here, meaning each pixel has up to 8 neighbours (horizontal, vertical, and diagonal).

The algorithm (Suzuki, 1985) assigns a unique integer label $\ell$ to each connected component and returns, for each component, a set of statistics including:
- $A_\ell$: the area in pixels
- $(x_\ell, y_\ell, w_\ell, h_\ell)$: the bounding box

### 10.2 Area Filtering

Each component is retained only if its area exceeds the **minimum object area** threshold $A_{min}$:

$$M_t^{\text{filtered}}(x,y) = \mathbf{1}\!\left[\ell(x,y) \neq 0 \;\wedge\; A_{\ell(x,y)} \geq A_{min}\right]$$

This step eliminates spurious small blobs that survived morphological opening. Setting $A_{min}$ appropriately requires knowing the expected minimum size of objects of interest in pixels, which depends on the resolution and the distance of objects from the camera.

---

## 11. Contour Detection and Bounding Boxes

The final step transforms the binary mask into interpretable output: **visual annotations** on the original frame.

### 11.1 Contour Extraction

**Contours** are the boundaries of foreground regions. OpenCV's `findContours` implements the algorithm of Suzuki & Abe (1985), which traces the outer boundary of each connected component as an ordered sequence of pixel coordinates. The `RETR_EXTERNAL` mode retains only the outermost contour of each region, ignoring interior holes. `CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments, storing only their endpoints.

### 11.2 Bounding Rectangles

For each contour, the **axis-aligned bounding rectangle** is computed:

$$(x_c, y_c, w_c, h_c) = \arg\min_{x,y,w,h} \{wh \;:\; \text{contour} \subseteq [x, x+w] \times [y, y+h]\}$$

This is the smallest upright rectangle enclosing all points of the contour. It is a crude but computationally cheap approximation of the object's extent. More precise shape descriptors (minimum-area rotated rectangle, convex hull, ellipse fit) could be substituted at higher cost.

### 11.3 Overlay Composition

Two output videos are produced:

**Foreground mask video** — the binary mask $M_t$ (grayscale, converted to BGR for the video writer) is saved directly. This gives a clear, unambiguous view of what the detector classifies as foreground at each frame.

**Overlay video** — the original colour frame is blended with a green-channel highlight of the mask using a weighted sum:

$$I_{\text{overlay}} = \alpha_1 \cdot I_{\text{frame}} + \alpha_2 \cdot I_{\text{green}}$$

with $\alpha_1 = 1.0$ and $\alpha_2 = 0.45$, where $I_{\text{green}}$ is a three-channel image that is zero everywhere except in the green channel, where it equals $M_t$. The bounding rectangle of each detected object is then drawn on top in red.

---

## 12. Parameter Guide

The table below summarises all user-facing parameters, their mathematical role, and practical tuning advice.

| Parameter | Method(s) | Mathematical role | Tuning advice |
|---|---|---|---|
| **History** | MOG2, KNN | Temporal memory $H \approx 1/\alpha$; number of frames used to build the model | Increase for slow-changing backgrounds; decrease for fast changes |
| **MOG2 variance threshold** $\lambda$ | MOG2 | Mahalanobis distance threshold for background classification | Increase to reduce false positives; decrease to improve sensitivity |
| **KNN distance threshold** $d^2$ | KNN | Squared intensity distance for NN matching | Larger = less sensitive; smaller = more false positives from noise |
| **Detect shadows** | MOG2, KNN | Enables HSV-based shadow/foreground discrimination | Enable when object shadows cause oversized detections |
| **Learning rate** $\alpha$ | MOG2, KNN, Running Avg | Weight of the current frame in background model update | High: fast adaptation but objects may vanish if still; Low: stable but slow to adapt |
| **Difference threshold** $\tau$ | Frame Diff, Running Avg | Binarization threshold on the absolute difference image | Tune based on expected contrast of moving objects vs background |
| **Blur kernel size** $k$ | All | Size of Gaussian pre-smoothing kernel (must be odd) | Increase for noisy videos; keep small to preserve fine motion detail |
| **Opening kernel size** | All | Structuring element for erosion+dilation (noise removal) | Increase to eliminate small spurious detections |
| **Closing kernel size** | All | Structuring element for dilation+erosion (gap filling) | Increase to obtain more solid, filled object masks |
| **Minimum object area** $A_{min}$ | All | Connected component area filter threshold (pixels) | Set to the minimum expected object size in pixels |
| **Maximum output dimension** | All | Rescaling of the video's largest dimension before processing | Reduce to accelerate processing on high-resolution videos |
| **Box thickness** | All | Visual thickness of bounding rectangles in the overlay | Aesthetic only |
| **Max frames** | All | Limits the number of processed frames | Reduce for quick previews |

---
