"""
Sistema de reconocimiento facial en tiempo real.
Detecta rostros en la webcam, los compara con la base de datos de embeddings
y muestra el nombre de la persona identificada.

Método de comparación:
- Calcula la distancia euclidiana entre el embedding del rostro detectado
  y todos los embeddings en la base de datos.
- Si la distancia mínima está por debajo del umbral, identifica a la persona.
- También se puede usar distancia coseno (implementada como alternativa).

Uso:
    python 04_recognition_realtime.py
    
Controles:
    - Presiona 'q' para salir
"""

import cv2
import numpy as np
import pickle
import os
from keras_facenet import FaceNet
from mtcnn import MTCNN
from config import (EMBEDDINGS_DIR, IMAGE_SIZE, DISTANCE_THRESHOLD, 
                    CONFIDENCE_THRESHOLD)

class FaceRecognitionSystem:
    """Sistema de reconocimiento facial en tiempo real."""
    
    def __init__(self):
        """Inicializa el sistema de reconocimiento."""
        self.embedder = None
        self.detector = None
        self.database = None
        self.load_models()
        self.load_database()
    
    def load_models(self):
        """Carga los modelos de detección y embedding."""
        print("⏳ Cargando modelos...")
        
        # Cargar FaceNet para embeddings
        print("   - Cargando FaceNet...")
        self.embedder = FaceNet()
        
        # Cargar MTCNN para detección
        print("   - Cargando MTCNN...")
        self.detector = MTCNN()
        
        print("✅ Modelos cargados correctamente")
    
    def load_database(self):
        """Carga la base de datos de embeddings."""
        database_path = os.path.join(EMBEDDINGS_DIR, 'face_embeddings.pkl')
        
        if not os.path.exists(database_path):
            raise FileNotFoundError(
                f"❌ No se encontró la base de datos en: {database_path}\n"
                f"💡 Primero ejecuta: 03_generate_embeddings.py"
            )
        
        print("⏳ Cargando base de datos de embeddings...")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)
        
        print(f"✅ Base de datos cargada: {len(self.database['embeddings'])} embeddings")
        print(f"👥 Personas registradas: {len(np.unique(self.database['labels']))}")
        
        # Mostrar personas
        for person in np.unique(self.database['labels']):
            count = np.sum(self.database['labels'] == person)
            print(f"   - {person}: {count} embeddings")
    
    def get_face_embedding(self, face_image):
        """
        Genera el embedding de un rostro.
        
        Args:
            face_image (np.array): Imagen del rostro (RGB, 160x160)
            
        Returns:
            np.array: Vector de embedding normalizado
        """
        # Asegurar dimensiones correctas
        if face_image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Expandir dimensiones para batch
        face_batch = np.expand_dims(face_image, axis=0)
        
        # Generar embedding
        embedding = self.embedder.embeddings(face_batch)[0]
        
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def euclidean_distance(self, embedding1, embedding2):
        """
        Calcula la distancia euclidiana entre dos embeddings.
        
        Args:
            embedding1 (np.array): Primer embedding
            embedding2 (np.array): Segundo embedding
            
        Returns:
            float: Distancia euclidiana
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def cosine_distance(self, embedding1, embedding2):
        """
        Calcula la distancia coseno entre dos embeddings.
        
        Args:
            embedding1 (np.array): Primer embedding
            embedding2 (np.array): Segundo embedding
            
        Returns:
            float: Distancia coseno (1 - similitud coseno)
        """
        dot_product = np.dot(embedding1, embedding2)
        return 1 - dot_product  # Ya están normalizados
    
    def recognize_face(self, face_embedding, use_cosine=False):
        """
        Reconoce un rostro comparándolo con la base de datos.
        
        Args:
            face_embedding (np.array): Embedding del rostro a reconocer
            use_cosine (bool): Usar distancia coseno en lugar de euclidiana
            
        Returns:
            tuple: (nombre, distancia) o (None, None) si no se reconoce
        """
        # Calcular distancias con todos los embeddings en la base de datos
        distances = []
        
        for db_embedding in self.database['embeddings']:
            if use_cosine:
                dist = self.cosine_distance(face_embedding, db_embedding)
            else:
                dist = self.euclidean_distance(face_embedding, db_embedding)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Encontrar el embedding más cercano
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Verificar si está dentro del umbral
        if min_distance < DISTANCE_THRESHOLD:
            recognized_name = self.database['labels'][min_distance_idx]
            return recognized_name, min_distance
        else:
            return None, min_distance
    
    def draw_face_box(self, frame, detection, name=None, distance=None):
        """
        Dibuja un rectángulo alrededor del rostro y muestra el nombre.
        
        Args:
            frame (np.array): Frame de video
            detection (dict): Información de detección de MTCNN
            name (str): Nombre de la persona reconocida
            distance (float): Distancia del embedding
        """
        x, y, width, height = detection['box']
        x, y = abs(x), abs(y)
        
        # Color según si se reconoce o no
        if name:
            color = (0, 255, 0)  # Verde si se reconoce
            label = f"{name}"
            confidence = f"Dist: {distance:.2f}"
        else:
            color = (0, 0, 255)  # Rojo si no se reconoce
            label = "Desconocido"
            confidence = f"Dist: {distance:.2f}" if distance else ""
        
        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        # Dibujar fondo para el texto
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
        
        # Dibujar nombre
        cv2.putText(frame, label, (x + 5, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dibujar confianza/distancia
        if confidence:
            cv2.putText(frame, confidence, (x, y + height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """Ejecuta el sistema de reconocimiento en tiempo real."""
        print("\n" + "=" * 60)
        print("🎥 INICIANDO RECONOCIMIENTO FACIAL EN TIEMPO REAL")
        print("=" * 60)
        print(f"\n⚙️  Configuración:")
        print(f"   - Umbral de distancia: {DISTANCE_THRESHOLD}")
        print(f"   - Umbral de confianza MTCNN: {CONFIDENCE_THRESHOLD}")
        print(f"   - Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"\n⌨️  Controles:")
        print(f"   - Presiona 'q' para salir")
        print(f"   - Presiona 'c' para cambiar entre distancia euclidiana/coseno")
        print("\n📸 Abriendo cámara...\n")
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        use_cosine = False
        frame_count = 0
        process_every_n_frames = 3  # Procesar cada N frames para mejor rendimiento
        
        print("✅ Sistema listo. Mostrando cámara...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al leer el frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Procesar solo cada N frames
            if frame_count % process_every_n_frames == 0:
                # Convertir a RGB para MTCNN
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detectar rostros
                detections = self.detector.detect_faces(rgb_frame)
                
                # Procesar cada rostro detectado
                for detection in detections:
                    # Filtrar por confianza
                    if detection['confidence'] < CONFIDENCE_THRESHOLD:
                        continue
                    
                    try:
                        # Extraer rostro
                        x, y, width, height = detection['box']
                        x, y = abs(x), abs(y)
                        
                        # Agregar margen
                        margin = int(0.15 * max(width, height))
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(rgb_frame.shape[1], x + width + margin)
                        y2 = min(rgb_frame.shape[0], y + height + margin)
                        
                        face = rgb_frame[y1:y2, x1:x2]
                        
                        if face.size == 0:
                            continue
                        
                        # Generar embedding
                        face_embedding = self.get_face_embedding(face)
                        
                        # Reconocer rostro
                        name, distance = self.recognize_face(face_embedding, use_cosine)
                        
                        # Dibujar resultado
                        self.draw_face_box(display_frame, detection, name, distance)
                        
                    except Exception as e:
                        print(f"⚠️  Error al procesar rostro: {str(e)}")
                        continue
            
            # Mostrar información en pantalla
            method = "Coseno" if use_cosine else "Euclidiana"
            cv2.putText(display_frame, f"Metodo: {method}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Umbral: {DISTANCE_THRESHOLD}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Presiona 'q' para salir | 'c' cambiar metodo", 
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Mostrar frame
            cv2.imshow('Reconocimiento Facial - Face Recognition System', display_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                use_cosine = not use_cosine
                method = "Coseno" if use_cosine else "Euclidiana"
                print(f"🔄 Cambiado a distancia {method}")
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("✅ Sistema de reconocimiento finalizado")
        print("=" * 60)

def main():
    """Función principal."""
    try:
        # Crear sistema de reconocimiento
        system = FaceRecognitionSystem()
        
        # Ejecutar
        system.run()
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
