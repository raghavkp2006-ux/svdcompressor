# SVD Image Compressor

An interactive, mathematical image compression tool that leverages Singular Value Decomposition (SVD) to reduce file sizes while maintaining image quality. 

Built with a high-performance Python (Flask + NumPy) backend and a beautiful, responsive frontend featuring a modern glassmorphism UI with real-time analytics.

![SVD Compressor UI](https://via.placeholder.com/800x450?text=SVD+Compressor+UI+Preview) <!-- Replace with an actual screenshot of the app -->

## ✨ Features

- **Dual Compression Modes:**
  - `Basic SVD`: Manually constrain the number of singular values ($k$) to visualize immediate compression vs. quality tradeoffs.
  - `Adaptive SVD`: Specify a target energy retention percentage (e.g., 95%) and let the algorithm automatically compute the optimal $k$ value.
- **Real-Time Analytics & Metrics:**
  - Calculates MSE (Mean Squared Error), PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and CR (Compression Ratio) instantly.
- **Interactive Visualizations:**
  - Before/After interactive slider to compare the original and compressed images.
  - Rich Data Charts (via Chart.js) mapping Singular Value Decay, Cumulative Energy, PSNR vs. $k$, and SSIM vs. $k$.
- **Premium UI/UX:**
  - Liquid glass aesthetics powered by Vanilla CSS.
  - Persistent Light & Dark mode toggle.
- **Fast Local Array Processing** powered by `NumPy`.

## 🛠️ Technology Stack

- **Backend:** Python, Flask, NumPy, Pillow
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/svd-compression.git
   cd svd-compression
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   Navigate to `http://127.0.0.1:5000` to interact with the SVD Image Compressor.

## 🧠 How It Works

This project demonstrates the core principle of **Low-Rank Matrix Approximation**.

1. **Decompose:** Each color channel (Red, Green, Blue) of the uploaded image is treated as a 2D matrix $A$ and decomposed into three matrices: $A = U \Sigma V^T$.
2. **Truncate:** Based on your settings, only the top $k$ largest singular values in $\Sigma$ are retained. The remaining values are discarded, drastically reducing the data required to approximate the matrix.
3. **Reconstruct:** The image is mathematically reconstructed using the truncated matrices $A_k = U_k \Sigma_k V_k^T$.
4. **Evaluate:** The server measures and returns exact statistical values showing how much space was saved and how structural integrity was preserved.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
This project is open-sourced under the MIT License.
