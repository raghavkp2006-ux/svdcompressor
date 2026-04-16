# SVD Image Compressor

An interactive, mathematical image compression tool that leverages Singular Value Decomposition (SVD) to reduce file sizes while maintaining image quality.

Built with a high-performance Python (Flask + NumPy) backend and a beautiful, responsive frontend featuring a modern glassmorphism UI with real-time analytics.

![SVD Compressor — Dark Mode Hero](docs/screenshot.png)

---

## ✨ Features

- **Dual Compression Modes:**
  - `Basic SVD`: Manually set the number of singular values ($k$) to see immediate compression vs. quality tradeoffs.
  - `Adaptive SVD`: Set a target energy retention % (e.g. 95%) — the algorithm auto-selects the optimal $k$ **per color channel** using complexity pre-screening.
- **Real-Time Analytics & Metrics:**
  - Computes MSE, PSNR (dB), SSIM, and Compression Ratio instantly after each compression.
- **Interactive Visualizations:**
  - Before/After drag slider for pixel-perfect comparison.
  - Rich Chart.js charts: Singular Value Decay, Cumulative Energy, PSNR vs $k$, SSIM vs $k$.
  - Error map and block complexity heatmap.
- **PDF Report:** Download a formatted PDF of compression results via ReportLab.
- **Premium UI/UX:** Glassmorphism, animated aurora background, persistent dark/light mode toggle.
- **Fast In-Memory Processing** — no disk writes, powered entirely by NumPy.

---

## 🏗️ System Architecture

```mermaid
graph TD
    U[("User / Browser")] -->|HTTP POST /api/compress| F["Flask Backend (app.py)"]
    F -->|returns JSON| U
    U -->|renders| FE["Frontend\n(HTML + CSS + JS + Chart.js)"]
    F -->|calls| C["lib/compress.py\n(Top-level API)"]

    subgraph "Compression Library (lib/)"
        C -->|calls| PS["lib/prescreening.py\n(Complexity Scoring)"]
        C -->|calls| RV["lib/rsvd.py\n(Randomized SVD)"]
    end
```

---

## 🔄 Request Flow (Sequence Diagram)

```mermaid
sequenceDiagram
    participant Browser
    participant Flask as Flask (app.py)
    participant compress as compress.py
    participant prescreening as prescreening.py
    participant rsvd as rsvd.py

    Browser->>Flask: POST /api/compress (image, mode, k, energy)
    Flask->>Flask: Open image, resize if > 800px, convert to float64
    Flask->>Flask: Compute singular values & energy curve for charts

    alt mode == "adaptive"
        Flask->>compress: compress_image(array, rank=None, energy%)
        loop For each channel (R, G, B)
            compress->>prescreening: complexity_score(channel)
            prescreening-->>compress: score (float)
            alt score < SKIP_THRESHOLD (80.0)
                compress-->>compress: Skip SVD (channel is flat)
            else
                compress->>prescreening: recommend_rank(score, energy%, max_dim)
                prescreening-->>compress: optimal k
                compress->>rsvd: rsvd(channel, k)
                rsvd-->>compress: U, S, Vt
                compress->>rsvd: reconstruct(U, S, Vt)
                rsvd-->>compress: compressed_channel
            end
        end
        compress-->>Flask: compressed_array, k_values, scores
    else mode == "basic"
        Flask->>compress: compress_image_basic(array, k)
        loop For each channel
            compress->>rsvd: rsvd(channel, k)
            rsvd-->>compress: U, S, Vt → reconstruct
        end
        compress-->>Flask: compressed_array
    end

    Flask->>Flask: Compute MSE, PSNR, SSIM, CR
    Flask->>Flask: Generate error map & block heatmap
    Flask->>Flask: Build k-comparison sweep for charts
    Flask-->>Browser: JSON { original, compressed, error_map, heatmap, metrics, charts }
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask 3.x, Flask-CORS |
| **Numerical** | NumPy ≥ 1.24 |
| **Image I/O** | Pillow ≥ 10.0 |
| **PDF Reports** | ReportLab ≥ 4.0 |
| **Frontend** | HTML5, Vanilla CSS3, Vanilla JS (ES6+) |
| **Charts** | Chart.js 4.x (CDN) |
| **Fonts** | Inter + JetBrains Mono (Google Fonts) |
| **Deployment** | Vercel (serverless, `vercel.json` included) |

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ installed.

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/raghavkp2006-ux/svdcompressor.git
cd svdcompressor

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## 🧠 How It Works

### 1. Decompose
Each color channel (R, G, B) is treated as matrix **A** and decomposed:

$$A = U \Sigma V^T$$

### 2. Truncate
Only the top $k$ singular values are kept — the rest are discarded:

$$A_k = U_k \Sigma_k V_k^T$$

By the **Eckart–Young theorem**, this is the optimal rank-$k$ approximation.

### 3. Reconstruct
The compressed image is rebuilt from far fewer numbers:

$$\text{Storage: } k(m + n + 1) \text{ vs original } m \times n$$

### 4. Evaluate
Quality is measured via PSNR, SSIM, MSE, and Compression Ratio — all displayed in real time.

---

## 📁 Project Structure

```
svdcompressor/
├── app.py                  # Flask backend — routes, metrics, PDF generation
├── requirements.txt
├── vercel.json             # Vercel deployment config
├── docs/
│   └── screenshot.png      # App screenshot
├── lib/
│   ├── rsvd.py             # Randomized SVD core (Halko et al. 2011)
│   ├── prescreening.py     # Block-variance complexity scoring
│   └── compress.py         # Top-level compression API
├── static/
│   ├── style.css           # Design system (1678 lines, glassmorphism)
│   └── script.js           # Frontend logic & Chart.js rendering
└── templates/
    └── index.html          # Single-page application
```

---

## 📊 Expected Results

| Rank (k) | Compression Ratio | PSNR | SSIM | Quality |
|---|---|---|---|---|
| 5 | ~50× | ~20 dB | ~0.60 | Heavy artifacts |
| 20 | ~15× | ~28 dB | ~0.80 | Recognizable |
| 50 | ~6× | ~35 dB | ~0.92 | Good |
| 100 | ~3× | ~40 dB | ~0.97 | Near-lossless |
| 200 | ~1.5× | ~45 dB | ~0.99 | Indistinguishable |

---

## 📄 References

1. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). *Finding Structure with Randomness*. SIAM Review.
2. Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika.
3. Wang, Z. et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP.

---

## 📝 License

Open-sourced under the MIT License.
