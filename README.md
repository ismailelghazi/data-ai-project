# Emotion Detection App

A simple app that detects emotions from face images using AI.

## What does it do?

This project can look at a photo of someone's face and tell you what emotion they're feeling (happy, sad, angry, etc.).

It has two parts:
- **ML folder**: The AI model that detects emotions
- **API folder**: A web service that lets you upload images and get results

## Emotions it can detect

- Happy 😊
- Sad 😢
- Angry 😠
- Surprised 😲
- Fearful 😨
- Disgusted 🤢
- Neutral 😐

## How to use it

### Step 1: Install Python

Make sure you have Python 3.8+ installed on your computer.

### Step 2: Install dependencies

```bash
# Install ML dependencies
pip install tensorflow opencv-python numpy matplotlib

# Install API dependencies
pip install fastapi uvicorn sqlalchemy python-multipart
```

### Step 3: Download the face detector

```bash
cd ML
python download_haarcascade.py
```

**Important:** Make sure you have the trained model file `emotion_model.h5` in the `ML/models/` folder. The API needs this model to work!

### Step 4: Run the API

```bash
cd API
uvicorn main:app --reload
```

The API will start at: `http://localhost:8000`

### Step 5: Test it out

Open your browser and go to:
- `http://localhost:8000/docs` - Try the API with a simple interface
- `http://localhost:8000` - See the welcome message

## How to use the ML model directly

```bash
cd ML
python detect.py path/to/your/image.jpg
```

## Project Structure

```
emotion-detection/
├── ML/                    # Machine Learning code
│   ├── detect.py          # Script to detect emotions
│   ├── emotion.ipynb      # Notebook to train the model
│   └── models/            # Trained AI models
│
├── API/                   # Web API
│   ├── main.py           # API startup
│   ├── routes/           # API endpoints (uses ML model)
│   ├── core/             # Database setup
│   ├── models/           # Database models
│   └── schemas/          # Data structures
│
└── README.md             # This file
```

## API Endpoints

Once the API is running, you can:

- **POST /api/predict_emotion** - Upload an image to detect emotion
- **GET /api/history** - See all past predictions
- **GET /health** - Check if API is working

The API automatically loads the ML model from `ML/models/emotion_model.h5` when it starts.

## Tech Stack

**Machine Learning:**
- TensorFlow - AI framework
- OpenCV - Image processing
- NumPy - Math operations

**API:**
- FastAPI - Web framework
- SQLAlchemy - Database
- Uvicorn - Server

## Model Info

- **Accuracy**: ~54% on test data
- **Image size**: 48x48 pixels
- **Training images**: 28,709
- **Test images**: 7,178

## How it works

1. You upload an image to the API
2. The API loads the ML model from `ML/models/`
3. The app finds faces in the image using Haar Cascade
4. Each face is checked by the AI model (`emotion_model.h5`)
5. The AI predicts the emotion and confidence
6. The result is saved to the database
7. You get the emotion with a confidence score

## Notes

- Works best with clear front-facing photos
- Needs good lighting
- One face per image works best
- The model was trained on the FER dataset

## Need Help?

Check the `/docs` page when the API is running - it has examples and lets you test everything.

## Database

The API saves predictions in a SQLite database (`emotion_predictions.db`) so you can see your history.

---

Made with ❤️ using Python and AI
