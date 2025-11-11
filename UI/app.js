// API Configuration
const API_URL = 'http://localhost:8000';

// Emotion Icons
const emotionIcons = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprised': '😲',
    'fearful': '😨',
    'disgusted': '🤢',
    'neutral': '😐'
};

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const emotionIcon = document.getElementById('emotionIcon');
const emotionName = document.getElementById('emotionName');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const loadHistoryBtn = document.getElementById('loadHistoryBtn');
const historyList = document.getElementById('historyList');

let selectedFile = null;

// Upload Box Click
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// File Selection
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Handle File Selection
function handleFileSelect(file) {
    if (!file) return;

    // Check if it's an image
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Change Image Button
changeImageBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadBox.style.display = 'block';
    analyzeBtn.disabled = true;
    resultSection.style.display = 'none';
});

// Analyze Button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    hideError();
    analyzeBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send to API
        const response = await fetch(`${API_URL}/api/predict_emotion`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to analyze image');
        }

        // Show result
        displayResult(data);

    } catch (err) {
        showError(err.message);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

// Display Result
function displayResult(data) {
    const emotion = data.emotion.toLowerCase();
    const confidence = Math.round(data.confidence * 100);

    // Set emotion icon
    emotionIcon.textContent = emotionIcons[emotion] || '🤔';

    // Set emotion name
    emotionName.textContent = data.emotion;

    // Set confidence bar
    confidenceFill.style.width = `${confidence}%`;
    confidenceText.textContent = `Confidence: ${confidence}%`;

    // Show result section
    resultSection.style.display = 'block';

    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load History
loadHistoryBtn.addEventListener('click', async () => {
    loadHistoryBtn.disabled = true;
    loadHistoryBtn.textContent = 'Loading...';

    try {
        const response = await fetch(`${API_URL}/api/history`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error('Failed to load history');
        }

        displayHistory(data.predictions);

    } catch (err) {
        showError(err.message);
    } finally {
        loadHistoryBtn.disabled = false;
        loadHistoryBtn.textContent = 'Refresh History';
    }
});

// Display History
function displayHistory(predictions) {
    if (!predictions || predictions.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No predictions yet</div>';
        return;
    }

    historyList.innerHTML = predictions.map(pred => {
        const confidence = Math.round(pred.confidence * 100);
        const date = new Date(pred.created_at).toLocaleString();

        return `
            <div class="history-item">
                <div>
                    <div class="history-item-emotion">
                        ${emotionIcons[pred.emotion.toLowerCase()] || '🤔'} ${pred.emotion}
                    </div>
                    <div class="history-item-date">${date}</div>
                </div>
                <div class="history-item-confidence">${confidence}%</div>
            </div>
        `;
    }).join('');
}

// Show Error
function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'block';
}

// Hide Error
function hideError() {
    error.style.display = 'none';
}

// Check API Connection on Load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            showError('Cannot connect to API. Make sure the API is running at http://localhost:8000');
        }
    } catch (err) {
        showError('Cannot connect to API. Make sure the API is running at http://localhost:8000');
    }
});
