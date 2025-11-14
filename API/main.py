from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import engine, Base
from routes.prediction import router as prediction_router

# Créer les tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Emotion Detection API",
    description="API de détection d'émotions faciales",
    version="1.0.0"
)

# # Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À ajuster en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
 )

# Inclure les routes
app.include_router(prediction_router, prefix="/api", tags=["Predictions"])

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de Détection d'Émotions!",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}