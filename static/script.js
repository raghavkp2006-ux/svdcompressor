/* ═══════════════════════════════════════════════════════════════════
   SVD COMPRESSOR — Frontend Logic
   Handles file upload, API calls, comparison slider, Chart.js,
   and dark/light theme toggle (CogniVest-style)
   ═══════════════════════════════════════════════════════════════════ */

// ── State ──
let currentMode = 'basic';
let selectedFile = null;
let lastResponse = null;

// Chart instances
let chartSV = null;
let chartEnergy = null;
let chartPSNR = null;
let chartSSIM = null;
let chartCRPSNR = null;
let chartBenchmark = null;


// ══════════════════════════════════════════════════════
// THEME TOGGLE — CogniVest-style dark/light mode
// Persists to localStorage, respects OS preference
// ══════════════════════════════════════════════════════

function initTheme() {
    const stored = localStorage.getItem('svd-theme');
    if (stored === 'light') {
        document.documentElement.classList.remove('dark');
    } else if (stored === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        // First visit: use OS preference, default to dark
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            document.documentElement.classList.remove('dark');
        } else {
            document.documentElement.classList.add('dark');
        }
    }
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('svd-theme', isDark ? 'dark' : 'light');

    // Re-render charts with new theme colors if data exists
    if (lastResponse) {
        setTimeout(() => {
            showCharts(lastResponse);
        }, 100);
    }
}

// Run theme init immediately (before DOM ready) to prevent flash
initTheme();

// Listen for OS theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('svd-theme')) {
        if (e.matches) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }
});

// ── DOM Elements ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const uploadZone   = $('#upload-zone');
const fileInput    = $('#file-input');
const previewImg   = $('#preview-img');
const uploadPreview = $('#upload-preview');
const btnRemove    = $('#btn-remove');
const btnCompress  = $('#btn-compress');
const btnBasic     = $('#btn-basic');
const btnAdaptive  = $('#btn-adaptive');
const modeIndicator = $('#mode-indicator');
const kControl     = $('#k-control');
const energyControl = $('#energy-control');
const kSlider      = $('#k-slider');
const energySlider = $('#energy-slider');
const kValue       = $('#k-value');
const energyValue  = $('#energy-value');
const resultsSection = $('#results-section');
const chartsSection  = $('#charts-section');
const compRange    = $('#comparison-range');
const compOverlay  = $('#comparison-overlay');
const compLine     = $('#comparison-line');
const btnDownload  = $('#btn-download');
const btnReport    = $('#btn-report');
const btnBenchmark = $('#btn-benchmark');
const themeToggle  = $('#theme-toggle');
const algoSelect   = $('#algo-select');

// ── Theme Toggle Listener ──
themeToggle.addEventListener('click', toggleTheme);


// ── Utility Functions ──
function getPSNRQuality(psnrValue) {
    /**
     * PSNR Quality Rating:
     * > 40 dB: Excellent (imperceptible difference)
     * 30-40 dB: Good (perceptible difference in dark areas)
     * 20-30 dB: Fair (noticeable difference)
     * < 20 dB: Poor (significant degradation)
     */
    if (psnrValue >= 40) return { label: 'Excellent', class: 'excellent' };
    if (psnrValue >= 30) return { label: 'Good', class: 'good' };
    if (psnrValue >= 20) return { label: 'Fair', class: 'fair' };
    return { label: 'Poor', class: 'poor' };
}


// ── Upload Handling ──
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});
btnRemove.addEventListener('click', removeFile);

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadZone.style.display = 'none';
        uploadPreview.style.display = 'block';
        btnCompress.disabled = false;
        
        // Automatically detect and set best algorithm/parameters
        autoDetectAlgorithm();
    };
    reader.readAsDataURL(file);
}

function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadZone.style.display = 'flex';
    uploadPreview.style.display = 'none';
    btnCompress.disabled = true;
}


// ── Mode Toggle ──
btnBasic.addEventListener('click', () => switchMode('basic'));
btnAdaptive.addEventListener('click', () => switchMode('adaptive'));

function switchMode(mode) {
    currentMode = mode;
    btnBasic.classList.toggle('active', mode === 'basic');
    btnAdaptive.classList.toggle('active', mode === 'adaptive');
    modeIndicator.classList.toggle('right', mode === 'adaptive');
    kControl.style.display     = mode === 'basic'    ? 'flex' : 'none';
    energyControl.style.display = mode === 'adaptive' ? 'flex' : 'none';
}


// ── Sliders ──
kSlider.addEventListener('input', () => {
    kValue.textContent = kSlider.value;
});

energySlider.addEventListener('input', () => {
    energyValue.textContent = energySlider.value + '%';
});

// ── Algorithm Toggle ──
algoSelect.addEventListener('change', (e) => {
    if (e.target.value !== 'svd' && e.target.value !== 'pca') {
        $('#mode-control-group').style.display = 'none';
        if (currentMode === 'adaptive') {
            switchMode('basic');
        }
    } else {
        $('#mode-control-group').style.display = 'flex';
    }
});

async function autoDetectAlgorithm() {
    if (!selectedFile) return;
    
    // Show some loading state on the button
    const originalText = btnCompress.innerHTML;
    btnCompress.innerHTML = '🤖 Auto-Detecting Best Settings...';
    btnCompress.disabled = true;
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const res = await fetch('/api/recommend_algorithm', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (!data.success) {
            console.error('Error auto-detecting:', data.error);
            return;
        }
        
        // Select the recommended algorithm
        algoSelect.value = data.recommended_algorithm;
        algoSelect.dispatchEvent(new Event('change'));
        
        // Select the recommended mode
        if (data.recommended_mode && (data.recommended_algorithm === 'svd' || data.recommended_algorithm === 'pca')) {
            switchMode(data.recommended_mode);
        }
        
        // Set the recommended k value
        if (data.recommended_k) {
            kSlider.value = data.recommended_k;
            kValue.textContent = data.recommended_k;
        }
        
    } catch (err) {
        console.error('Failed to auto-detect settings:', err.message);
    } finally {
        btnCompress.innerHTML = originalText;
        btnCompress.disabled = false;
    }
}

// ── Compress ──
btnCompress.addEventListener('click', compress);

async function compress() {
    if (!selectedFile) return;

    // Show loading state
    const btnContent = $('.btn-content');
    const btnLoading = $('.btn-loading');
    btnContent.style.display = 'none';
    btnLoading.style.display = 'flex';
    btnCompress.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('mode', currentMode);
    formData.append('k', kSlider.value);
    formData.append('energy', energySlider.value);
    formData.append('algo', algoSelect.value);

    try {
        const res = await fetch('/api/compress', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (!data.success) {
            alert('Error: ' + (data.error || 'Unknown error'));
            return;
        }

        lastResponse = data;
        showResults(data);
        showCharts(data);

    } catch (err) {
        alert('Failed to connect to server: ' + err.message);
    } finally {
        btnContent.style.display = 'flex';
        btnLoading.style.display = 'none';
        btnCompress.disabled = false;
    }
}


// ── Results Display ──
function showResults(data) {
    const { metrics } = data;

    // Show section
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Animate metrics with count-up
    animateValue('val-psnr', metrics.psnr, 2, ' dB');
    animateValue('val-ssim', metrics.ssim, 4, '');
    animateValue('val-cr', metrics.cr, 2, '×');
    animateValue('val-mse', metrics.mse, 2, '');

    // Add quality label for PSNR
    const qualityLabel = getPSNRQuality(metrics.psnr);
    const qualityEl = $('#quality-label');
    qualityEl.textContent = qualityLabel.label;
    qualityEl.className = `metric-quality-label ${qualityLabel.class}`;

    // Before/After images
    const imgOrig = $('#img-original');
    const imgComp = $('#img-compressed');
    imgOrig.src = data.original;
    imgComp.src = data.compressed;

    // Error map and heatmap
    const imgErrorMap = $('#img-error-map');
    const imgHeatmap = $('#img-heatmap');
    if (data.error_map) imgErrorMap.src = data.error_map;
    if (data.heatmap) imgHeatmap.src = data.heatmap;

    // Comparison overlay size fix
    imgOrig.onload = () => {
        const w = imgOrig.naturalWidth;
        const h = imgOrig.naturalHeight;
        const containerW = imgOrig.clientWidth;
        const scale = containerW / w;
        const displayH = h * scale;
        imgComp.style.width = containerW + 'px';
        imgComp.style.height = displayH + 'px';
    };

    // Labels
    $('#comparison-k').textContent = metrics.k_used;
    $('#info-size').textContent = `${metrics.image_size[0]} × ${metrics.image_size[1]}`;
    $('#info-mode').textContent = metrics.mode === 'adaptive' ? 'Adaptive SVD' : 'Basic SVD';
    $('#info-time').textContent = metrics.compute_time.toFixed(2) + 's';
    $('#info-algo').textContent = metrics.algorithm || 'rSVD';

    // Reset comparison slider
    compRange.value = 50;
    updateComparison(50);

    // Store metrics for report download
    window.lastMetrics = metrics;
    window.lastData = data;

    // Trigger card animations
    $$('.metric-card').forEach(el => {
        el.classList.remove('animate-in');
        void el.offsetWidth; // reflow
        el.classList.add('animate-in');
    });
}

function animateValue(elId, target, decimals, suffix) {
    const el = document.getElementById(elId);
    const duration = 800;
    const start = performance.now();
    const from = 0;

    function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = from + (target - from) * eased;
        el.textContent = current.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}


// ── Comparison Slider ──
compRange.addEventListener('input', () => {
    updateComparison(compRange.value);
});

function updateComparison(percent) {
    compOverlay.style.width = percent + '%';
    compLine.style.left = percent + '%';
}


// ── Download ──
btnDownload.addEventListener('click', () => {
    if (!lastResponse) return;

    // Create download link
    const a = document.createElement('a');
    a.href = lastResponse.compressed;
    a.download = 'svd_compressed.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

btnReport.addEventListener('click', downloadReport);

function downloadReport() {
    if (!window.lastData || !window.lastMetrics) {
        alert('No compression data available. Please compress an image first.');
        return;
    }

    const reportData = {
        metrics: window.lastMetrics,
        original: window.lastData.original,
        compressed: window.lastData.compressed,
        error_map: window.lastData.error_map || '',
        heatmap: window.lastData.heatmap || ''
    };

    // Show loading feedback
    const originalText = btnReport.textContent;
    btnReport.textContent = 'Generating...';
    btnReport.disabled = true;

    fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportData)
    })
        .then(res => {
            if (!res.ok) throw new Error('Report generation failed');
            return res.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'svd_compression_report.pdf';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        })
        .catch(err => {
            alert('Failed to generate report: ' + err.message);
        })
        .finally(() => {
            btnReport.textContent = originalText;
            btnReport.disabled = false;
        });
}


// ── Benchmark ──
btnBenchmark.addEventListener('click', runBenchmark);

async function runBenchmark() {
    if (!selectedFile) return;
    const originalText = btnBenchmark.innerHTML;
    btnBenchmark.innerHTML = '<span class="spinner" style="display:inline-block; vertical-align:middle; width:12px; height:12px; margin-right:4px;"></span> Running...';
    btnBenchmark.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('k', kSlider.value);

    try {
        const res = await fetch('/api/benchmark', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (!data.success) throw new Error(data.error);
        
        $('#card-benchmark').style.display = 'block';
        
        // ensure it animates in
        $('#card-benchmark').classList.remove('animate-in');
        void $('#card-benchmark').offsetWidth;
        $('#card-benchmark').classList.add('animate-in');
        
        renderBenchmarkChart(data.times);
        
        $('#card-benchmark').scrollIntoView({ behavior: 'smooth', block: 'end' });
    } catch(err) {
        alert('Benchmark failed: ' + err.message);
    } finally {
        btnBenchmark.innerHTML = originalText;
        btnBenchmark.disabled = false;
    }
}


// ── Charts (theme-aware) ──
const chartColors = {
    red: 'rgba(251, 113, 133, 0.9)',
    green: 'rgba(52, 211, 153, 0.9)',
    blue: 'rgba(96, 165, 250, 0.9)',
    purple: 'rgba(167, 139, 250, 0.9)',
    gray: 'rgba(161, 161, 170, 0.9)',
    redBg: 'rgba(251, 113, 133, 0.1)',
    greenBg: 'rgba(52, 211, 153, 0.1)',
    blueBg: 'rgba(96, 165, 250, 0.1)',
    purpleBg: 'rgba(167, 139, 250, 0.1)',
    grayBg: 'rgba(161, 161, 170, 0.1)',
};

function isDarkMode() {
    return document.documentElement.classList.contains('dark');
}

function getChartDefaults() {
    const dark = isDarkMode();
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: dark ? 'rgba(161, 161, 170, 0.8)' : 'rgba(82, 82, 91, 0.8)',
                    font: { family: "'Inter', sans-serif", size: 11 },
                    boxWidth: 12,
                    padding: 12
                }
            },
            tooltip: {
                backgroundColor: dark ? 'rgba(15, 15, 18, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                titleColor: dark ? '#fafafa' : '#09090b',
                bodyColor: dark ? '#a1a1aa' : '#52525b',
                borderColor: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 10,
                titleFont: { family: "'Inter', sans-serif", weight: '600' },
                bodyFont: { family: "'JetBrains Mono', monospace", size: 12 }
            }
        },
        scales: {
            x: {
                grid: { color: dark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)', drawBorder: false },
                ticks: { color: dark ? 'rgba(113, 113, 122, 0.8)' : 'rgba(113, 113, 122, 0.7)', font: { size: 10 } }
            },
            y: {
                grid: { color: dark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)', drawBorder: false },
                ticks: { color: dark ? 'rgba(113, 113, 122, 0.8)' : 'rgba(113, 113, 122, 0.7)', font: { size: 10 } }
            }
        }
    };
}

function showCharts(data) {
    chartsSection.classList.remove('hidden');

    // Delay for smooth entrance
    setTimeout(() => {
        if (Object.keys(data.singular_values || {}).length > 0) {
            $('#chart-sv').parentElement.parentElement.style.display = 'block';
            renderSVChart(data.singular_values);
        } else {
            $('#chart-sv').parentElement.parentElement.style.display = 'none';
        }
        
        if (Object.keys(data.energy_curve || {}).length > 0) {
            $('#chart-energy').parentElement.parentElement.style.display = 'block';
            renderEnergyChart(data.energy_curve);
        } else {
            $('#chart-energy').parentElement.parentElement.style.display = 'none';
        }
        
        renderComparisonCharts(data.comparison);
    }, 200);

    // Animate chart cards
    $$('.chart-card').forEach(el => {
        el.classList.remove('animate-in');
        void el.offsetWidth;
        el.classList.add('animate-in');
    });
}

function renderSVChart(svData) {
    if (chartSV) chartSV.destroy();
    const defaults = getChartDefaults();

    const datasets = [];
    const channelConfig = {
        red:   { color: chartColors.red,   bg: chartColors.redBg,   label: 'Red Channel' },
        green: { color: chartColors.green,  bg: chartColors.greenBg,  label: 'Green Channel' },
        blue:  { color: chartColors.blue,   bg: chartColors.blueBg,   label: 'Blue Channel' },
        gray:  { color: chartColors.gray,   bg: chartColors.grayBg,   label: 'Grayscale' }
    };

    for (const [key, values] of Object.entries(svData)) {
        const cfg = channelConfig[key] || channelConfig.gray;
        datasets.push({
            label: cfg.label,
            data: values,
            borderColor: cfg.color,
            backgroundColor: cfg.bg,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true
        });
    }

    const labels = Array.from({ length: datasets[0]?.data.length || 0 }, (_, i) => i + 1);

    chartSV = new Chart($('#chart-sv'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...defaults,
            scales: {
                ...defaults.scales,
                x: { ...defaults.scales.x, title: { display: true, text: 'Index', color: '#71717a', font: { size: 11 } } },
                y: { ...defaults.scales.y, title: { display: true, text: 'Singular Value', color: '#71717a', font: { size: 11 } }, type: 'logarithmic' }
            }
        }
    });
}

function renderEnergyChart(energyData) {
    if (chartEnergy) chartEnergy.destroy();
    const defaults = getChartDefaults();

    const datasets = [];
    const channelConfig = {
        red:   { color: chartColors.red,   bg: chartColors.redBg,   label: 'Red' },
        green: { color: chartColors.green,  bg: chartColors.greenBg,  label: 'Green' },
        blue:  { color: chartColors.blue,   bg: chartColors.blueBg,   label: 'Blue' },
        gray:  { color: chartColors.gray,   bg: chartColors.grayBg,   label: 'Grayscale' }
    };

    for (const [key, values] of Object.entries(energyData)) {
        const cfg = channelConfig[key] || channelConfig.gray;
        datasets.push({
            label: cfg.label,
            data: values,
            borderColor: cfg.color,
            backgroundColor: cfg.bg,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true
        });
    }

    // Add reference lines
    const len = datasets[0]?.data.length || 100;
    datasets.push({
        label: '95% threshold',
        data: Array(len).fill(95),
        borderColor: 'rgba(251, 191, 36, 0.5)',
        borderWidth: 1,
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false
    });

    const labels = Array.from({ length: len }, (_, i) => i + 1);

    chartEnergy = new Chart($('#chart-energy'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...defaults,
            scales: {
                ...defaults.scales,
                x: { ...defaults.scales.x, title: { display: true, text: 'k (Singular Values)', color: '#71717a', font: { size: 11 } } },
                y: { ...defaults.scales.y, title: { display: true, text: 'Energy (%)', color: '#71717a', font: { size: 11 } }, min: 0, max: 100 }
            }
        }
    });
}

function renderComparisonCharts(comparison) {
    if (chartPSNR) chartPSNR.destroy();
    if (chartSSIM) chartSSIM.destroy();
    if (chartCRPSNR) chartCRPSNR.destroy();
    const defaults = getChartDefaults();

    const labels = comparison.map(d => 'k=' + d.k);

    chartPSNR = new Chart($('#chart-psnr'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'PSNR (dB)',
                data: comparison.map(d => d.psnr),
                backgroundColor: comparison.map((_, i) =>
                    `rgba(167, 139, 250, ${0.3 + 0.1 * i})`
                ),
                borderColor: chartColors.purple,
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.6
            }]
        },
        options: {
            ...defaults,
            scales: {
                ...defaults.scales,
                y: { ...defaults.scales.y, title: { display: true, text: 'PSNR (dB)', color: '#71717a', font: { size: 11 } } }
            }
        }
    });

    chartSSIM = new Chart($('#chart-ssim'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'SSIM',
                data: comparison.map(d => d.ssim),
                backgroundColor: comparison.map((_, i) =>
                    `rgba(52, 211, 153, ${0.3 + 0.1 * i})`
                ),
                borderColor: chartColors.green,
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.6
            }]
        },
        options: {
            ...defaults,
            scales: {
                ...defaults.scales,
                y: { ...defaults.scales.y, title: { display: true, text: 'SSIM Score', color: '#71717a', font: { size: 11 } }, min: 0, max: 1 }
            }
        }
    });

    chartCRPSNR = new Chart($('#chart-cr-psnr'), {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Compression Ratio (x)',
                data: comparison.map(d => d.cr),
                borderColor: chartColors.blue,
                backgroundColor: chartColors.blueBg,
                borderWidth: 2,
                pointBackgroundColor: chartColors.blue,
                pointRadius: 4,
                yAxisID: 'y'
            }, {
                label: 'PSNR (dB)',
                data: comparison.map(d => d.psnr),
                borderColor: chartColors.red,
                backgroundColor: chartColors.redBg,
                borderWidth: 2,
                borderDash: [5, 5],
                pointBackgroundColor: chartColors.red,
                pointRadius: 4,
                yAxisID: 'y1'
            }]
        },
        options: {
            ...defaults,
            scales: {
                ...defaults.scales,
                y: { ...defaults.scales.y, type: 'linear', display: true, position: 'left', title: { display: true, text: 'Compression Ratio (x)', color: '#71717a', font: { size: 11 } } },
                y1: { ...defaults.scales.y, type: 'linear', display: true, position: 'right', title: { display: true, text: 'PSNR (dB)', color: '#71717a', font: { size: 11 } }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

function renderBenchmarkChart(times) {
    if (chartBenchmark) chartBenchmark.destroy();
    const defaults = getChartDefaults();
    
    const algos = [
        { id: 'svd', label: 'rSVD', color: chartColors.purple, bg: 'rgba(167, 139, 250, 0.4)' },
        { id: 'dct', label: 'DCT', color: chartColors.blue, bg: 'rgba(96, 165, 250, 0.4)' },
        { id: 'pca', label: 'PCA', color: chartColors.green, bg: 'rgba(52, 211, 153, 0.4)' },
        { id: 'nmf', label: 'NMF', color: chartColors.red, bg: 'rgba(251, 113, 133, 0.4)' }
    ];
    
    const dataVals = algos.map(a => times[a.id] || 0);
    const labels = algos.map(a => a.label);
    const bgColors = algos.map(a => a.bg);
    const borderColors = algos.map(a => a.color);

    chartBenchmark = new Chart($('#chart-benchmark'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Execution Time (s)',
                data: dataVals,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.6
            }]
        },
        options: {
            ...defaults,
            indexAxis: 'y',
            scales: {
                ...defaults.scales,
                x: { ...defaults.scales.x, title: { display: true, text: 'Time (Seconds)', color: '#71717a', font: { size: 11 } } },
                y: { ...defaults.scales.y, grid: { display: false } }
            }
        }
    });
}


// ── Smooth scroll for nav links ──
$$('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});



// ── Intersection Observer for reveal animations ──
const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    },
    { threshold: 0.1 }
);

$$('.glass-card, .explanation-card').forEach(el => observer.observe(el));
