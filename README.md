# Emotion Detection App

A simple app that detects emotions from face images using AI.

## What does it do?

This project can look at a photo of someone's face and tell you what emotion they're feeling (happy, sad, angry, etc.).

It has three parts:
- **ML folder**: The AI model that detects emotions
- **API folder**: A web service that lets you upload images and get results
- **UI folder**: A simple web interface to use the app

## Emotions it can detect

- Happy 😊
- Sad 😢
- Angry 😠
- Surprised 😲
- Fearful 😨
- Disgusted 🤢
- Neutral 😐

## How to use it

### Option 1: Using Docker (Recommended)

The easiest way to run the entire application is with Docker. This runs everything in containers without needing to install dependencies.

#### Prerequisites
- Docker and Docker Compose installed on your computer
- Make sure ports 80, 8000, and 5432 are available

#### Steps

1. **Clone or navigate to the project directory**
```bash
cd emotion-detection
```

2. **Configure environment variables (optional)**
```bash
# Copy the example env file and edit if needed
cp .env.example .env
# Edit .env with your preferred database credentials
```

3. **Build and start all services**
```bash
docker-compose up --build
```

This will start:
- PostgreSQL database on port 5432
- FastAPI backend on port 8000
- Nginx frontend on port 80

4. **Access the application**
- Open your browser and go to: `http://localhost`
- The UI will automatically connect to the API
- API docs available at: `http://localhost/api/docs`

5. **Stop the application**
```bash
# Press Ctrl+C, then run:
docker-compose down

# To remove all data including database:
docker-compose down -v
```

#### Docker Commands Reference
```bash
# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f

# Restart a specific service
docker-compose restart api

# Rebuild after code changes
docker-compose up --build

# Check running containers
docker-compose ps
```

---

### Option 2: Manual Setup (Without Docker)

If you prefer to run the application without Docker:

#### Step 1: Install Python

Make sure you have Python 3.8+ installed on your computer.

#### Step 2: Install dependencies

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

#### Step 3: Setup PostgreSQL Database

Install PostgreSQL and create a database:
```bash
createdb test
```

Update the database credentials in `API/.env` or set environment variables.

#### Step 4: Download the face detector (if needed)

```bash
cd ML
python download_haarcascade.py
```

**Important:** Make sure you have the trained model file `emotion_model.h5` in the `ML/models/` folder. The API needs this model to work!

#### Step 5: Run the API

```bash
cd API
uvicorn main:app --reload
```

The API will start at: `http://localhost:8000`

#### Step 6: Open the Web Interface

Simply open the file `UI/index.html` in your web browser!

**Or use the API directly:**
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
├── UI/                    # Web Interface
│   ├── index.html        # Main page
│   ├── style.css         # Styling
│   └── app.js            # JavaScript for API calls
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
- SQLAlchemy - Database ORM
- PostgreSQL - Database
- Uvicorn - ASGI Server

**Frontend:**
- Vanilla JavaScript
- HTML5/CSS3
- Nginx (when using Docker)

**DevOps:**
- Docker - Containerization
- Docker Compose - Multi-container orchestration

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

## Using the Web Interface

The UI is super simple to use:

1. Make sure the API is running first (see Step 4)
2. Open `UI/index.html` in any web browser
3. Click or drag & drop an image
4. Click "Analyze Emotion"
5. See the result with emoji and confidence score
6. Click "Load History" to see past predictions

## Notes

- Works best with clear front-facing photos
- Needs good lighting
- One face per image works best
- The model was trained on the FER dataset
- The web interface needs the API running to work

## Need Help?

Check the `/docs` page when the API is running - it has examples and lets you test everything.

## Database

The API saves predictions in a PostgreSQL database so you can see your history. When using Docker, the database is automatically set up and configured. For manual setup, you need to install PostgreSQL and configure the connection in your `.env` file.

---

Made with ❤️ using Python and AI
