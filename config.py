"""
Configuración central del proyecto de reconocimiento facial
"""
import os

# Directorios del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
ALIGNED_FACES_DIR = os.path.join(DATA_DIR, "aligned_faces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_IMAGES_DIR, ALIGNED_FACES_DIR, 
                  EMBEDDINGS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Parámetros del modelo
EMBEDDING_SIZE = 128  # FaceNet genera embeddings de 128 dimensiones
IMAGE_SIZE = 160  # Tamaño de entrada para FaceNet
CONFIDENCE_THRESHOLD = 0.9  # Umbral para detección de rostros (MTCNN)

# Parámetros de reconocimiento
DISTANCE_THRESHOLD = 0.6  # Umbral para considerar que dos rostros son la misma persona
# Valores típicos: 0.4 (muy estricto) - 0.6 (recomendado) - 0.8 (permisivo)

# Parámetros de captura
MIN_IMAGES_PER_PERSON = 20  # Mínimo de imágenes recomendado por persona
CAPTURE_INTERVAL = 0.5  # Segundos entre capturas automáticas
