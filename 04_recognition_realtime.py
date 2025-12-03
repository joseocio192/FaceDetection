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
        self.detector = MTCNN(min_face_size=80)
        
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
        Dibuja un rectángulo y muestra el nombre con porcentaje de fiabilidad.
        """
        x, y, width, height = detection['box']
        x, y = abs(x), abs(y)
        
        # Calcular porcentaje de fiabilidad (Heurística simple)
        # Distancia 0.0 = 100%, Distancia 1.0 = 0%
        # FaceNet suele dar distancias < 0.6 para matches
        reliability = 0.0
        if distance is not None:
            reliability = max(0.0, (0.8 - distance) / 0.8 * 100)
        # Determinar color y texto
        if name:
            # Si hay reconocimiento (Match)
            if reliability > 60:
                color = (0, 255, 0)   # Verde fuerte para alta confianza
            else:
                color = (0, 255, 255) # Amarillo para confianza media/baja
            
            label = name
            conf_text = f"{reliability:.1f}%"
        else:
            # Desconocido
            color = (0, 0, 255)       # Rojo
            label = "Desconocido"
            conf_text = f"{reliability:.1f}%" if distance else ""

        # --- DIBUJADO EN PANTALLA ---
        
        # 1. Rectángulo del rostro
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        # 2. Fondo para el nombre (arriba)
        # Calcular tamaño del texto para el fondo
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 30), (x + text_w + 10, y), color, -1)
        
        # 3. Nombre
        cv2.putText(frame, label, (x + 5, y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. Fondo para el porcentaje (abajo)
        if conf_text:
            conf_label = f"Fiabilidad: {conf_text}"
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Dibujar fondo negro semitransparente abajo para que se lea bien
            sub_y = y + height
            cv2.rectangle(frame, (x, sub_y), (x + conf_w + 10, sub_y + 25), (0, 0, 0), -1)
            
            # Texto de porcentaje
            cv2.putText(frame, conf_label, (x + 5, sub_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """
        Ejecuta el sistema con 'Memoria de Persistencia' para evitar parpadeos.
        """
        print("\n" + "=" * 60)
        print("🎥 INICIANDO RECONOCIMIENTO FACIA")
        print("=" * 60)
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # --- CONFIGURACIÓN DE RENDIMIENTO Y PERSISTENCIA ---
        use_cosine = True
        frame_count = 0
        process_every_n_frames = 15
        
        # PERSISTENCIA: Cuántos frames mantener el cuadro si se pierde el rostro
        # 20 frames es aprox 0.5 - 1 segundo (dependiendo de la velocidad de tu PC)
        PERSISTENCE_LIMIT = 20  
        frames_without_detection = 0
        
        # Memoria para guardar los últimos rostros detectados
        active_faces = [] 
        
        print("✅ Sistema listo. Mostrando cámara...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frames_without_detection += 1  # Asumimos que no hay detección hasta probar lo contrario
            display_frame = frame.copy()
            
            # --- FASE 1: DETECCIÓN (Solo cada N frames) ---
            if frame_count % process_every_n_frames == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detectar rostros
                    detections = self.detector.detect_faces(rgb_frame)
                    
                    # Lista temporal para este frame
                    current_detected_faces = []
                    
                    for detection in detections:
                        if detection['confidence'] < CONFIDENCE_THRESHOLD:
                            continue
                        
                        # Extraer coordenadas
                        x, y, width, height = detection['box']
                        x, y = abs(x), abs(y)
                        
                        # Margen
                        margin = int(0.15 * max(width, height))
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(rgb_frame.shape[1], x + width + margin)
                        y2 = min(rgb_frame.shape[0], y + height + margin)
                        
                        face = rgb_frame[y1:y2, x1:x2]
                        if face.size == 0: continue
                        
                        # Reconocer
                        face_embedding = self.get_face_embedding(face)
                        name, distance = self.recognize_face(face_embedding, use_cosine)
                        
                        # Guardar en lista temporal
                        current_detected_faces.append({
                            'detection': detection,
                            'name': name,
                            'distance': distance
                        })
                    
                    # --- LÓGICA DE PERSISTENCIA ---
                    if len(current_detected_faces) > 0:
                        active_faces = current_detected_faces
                        frames_without_detection = 0
                    else:
                        pass
                        
                except Exception as e:
                    print(f"⚠️ Error: {str(e)}")

            # --- FASE 2: LIMPIEZA ---
            # Si han pasado demasiados frames sin ver un rostro, olvidamos la memoria
            if frames_without_detection > PERSISTENCE_LIMIT:
                active_faces = []

            # --- FASE 3: DIBUJAR ---
            # Dibujamos lo que haya en memoria (sea nuevo o persistente)
            for face_data in active_faces:
                self.draw_face_box(
                    display_frame, 
                    face_data['detection'], 
                    face_data['name'], 
                    face_data['distance']
                )
            
            # Info en pantalla
            status_color = (0, 255, 0) if frames_without_detection < PERSISTENCE_LIMIT else (0, 0, 255)
            cv2.circle(display_frame, (30, 30), 10, status_color, -1) # Indicador de estado
            
            cv2.imshow('Reconocimiento Facial', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): use_cosine = not use_cosine
        
        cap.release()
        cv2.destroyAllWindows()

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
