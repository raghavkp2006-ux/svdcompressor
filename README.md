# Data Science Image Compressor (SVD, PCA, NMF, DCT)

An interactive, mathematical image compression laboratory that leverages advanced linear algebra and matrix factorization to reduce file sizes while maintaining image quality.

Built with a high-performance Python (Flask + NumPy + Scikit-Learn) backend and a beautiful, responsive frontend featuring a modern glassmorphism UI with real-time analytics.

---

## ✨ Features

- **Four Powerful Algorithms:**
  - `SVD (Randomized)`: Blazing fast Singular Value Decomposition using random projections.
  - `PCA (Principal Components)`: Covariance-based dimensionality reduction.
  - `NMF (Non-negative Matrix Factorization)`: Additive, parts-based image reconstruction.
  - `DCT (Discrete Cosine Transform)`: Frequency domain compression (the math behind JPEG).
- **Dual Compression Modes:**
  - `Basic Mode`: Manually set the number of components ($k$) to see immediate compression vs. quality tradeoffs (available for all algorithms).
  - `Adaptive Mode`: Set a target energy retention % (e.g. 95%) — the system automatically calculates and selects the optimal $k$ **per color channel** using complexity pre-screening (available for **SVD** and **PCA**).
- **Algorithm Speed Benchmark:**
  - Instantly race all 4 algorithms against each other on the same image to visualize execution times in a comparative bar chart.
- **Real-Time Analytics & Metrics:**
  - Computes MSE, PSNR (dB), SSIM, and Compression Ratio instantly after each compression.
- **Interactive Visualizations:**
  - Before/After drag slider for pixel-perfect comparison.
  - Rich Chart.js charts: Singular Value Decay, Cumulative Energy, PSNR vs $k$, SSIM vs $k$, Storage Savings vs Quality, and Algorithm Benchmark.
  - Error map and block complexity heatmap.
- **PDF Report:** Download a formatted PDF of compression results.
- **Premium UI/UX:** Glassmorphism, animated aurora background, persistent dark/light mode toggle.

---

## 🏗️ System Architecture

```mermaid
graph TD
    U[("User / Browser")] -->|HTTP POST /api/compress| F["Flask Backend (app.py)"]
    F -->|returns JSON| U
    U -->|renders| FE["Frontend\n(HTML + CSS + JS + Chart.js)"]
    
    F -->|calls| C["lib/compress.py"]
    F -->|calls| A["lib/algorithms.py"]

    subgraph "Compression Library (lib/)"
        C -->|calls| PS["prescreening.py\n(Complexity Scoring)"]
        C -->|calls| RV["rsvd.py\n(Randomized SVD)"]
        A --> PCA["sklearn.decomposition.PCA"]
        A --> NMF["sklearn.decomposition.NMF"]
        A --> DCT["scipy.fftpack.dct"]
    end
```

---

## 🔄 Adaptive Request Flow

```mermaid
sequenceDiagram
    participant Browser
    participant Flask as app.py
    participant compress as compress.py
    participant prescreening as prescreening.py
    participant algo as algorithms.py

    Browser->>Flask: POST /api/compress (image, mode="adaptive", energy)
    Flask->>Flask: Open image, convert to float64
    Flask->>Flask: Compute singular values & energy curve for charts

    alt algo == "svd" or "pca"
        loop For each channel (R, G, B)
            Flask->>prescreening: complexity_score(channel)
            prescreening-->>Flask: score (float)
            Flask->>prescreening: recommend_rank(score, energy%, max_dim)
            prescreening-->>Flask: optimal k
        end
        
        alt algo == "svd"
            Flask->>compress: compress_image(array, auto_k)
        else algo == "pca"
            Flask->>algo: compress_image_algo(array, auto_k, 'pca')
        end
        
    end

    Flask->>Flask: Compute MSE, PSNR, SSIM, CR
    Flask->>Flask: Generate error map & block heatmap
    Flask-->>Browser: JSON { original, compressed, metrics, charts }
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask 3.x, Flask-CORS |
| **Data Science** | NumPy, Scikit-Learn, SciPy |
| **Image I/O** | Pillow ≥ 10.0 |
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js 4.4.1 |

---

## 🚀 Quickstart

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Server:**
```bash
python app.py
```

3. **Open in Browser:**
Navigate to `http://localhost:5000`

---
*Created as an educational exploration into linear algebra, matrix factorization, and advanced data compression techniques.*
