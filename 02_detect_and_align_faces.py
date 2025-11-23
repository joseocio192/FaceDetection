"""
Script para detectar y alinear rostros usando MTCNN.
Procesa las imágenes raw y guarda rostros alineados listos para generar embeddings.

MTCNN (Multi-task Cascaded Convolutional Networks) es superior a Haar Cascades
porque detecta rostros con mayor precisión y proporciona puntos faciales clave.

Uso:
    python 02_detect_and_align_faces.py
"""

import cv2
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from config import RAW_IMAGES_DIR, ALIGNED_FACES_DIR, IMAGE_SIZE, CONFIDENCE_THRESHOLD

def align_face(image, detection, output_size=IMAGE_SIZE):
    """
    Alinea y recorta un rostro basándose en los puntos clave detectados.
    
    Args:
        image (np.array): Imagen original
        detection (dict): Diccionario con información de detección de MTCNN
        output_size (int): Tamaño de salida de la imagen alineada
        
    Returns:
        np.array: Rostro alineado y redimensionado
    """
    # Extraer bounding box
    x, y, width, height = detection['box']
    
    # Asegurar que las coordenadas estén dentro de los límites
    x, y = abs(x), abs(y)
    x2, y2 = x + width, y + height
    
    # Agregar margen (15% adicional)
    margin = 0.15
    x_margin = int(width * margin)
    y_margin = int(height * margin)
    
    x = max(0, x - x_margin)
    y = max(0, y - y_margin)
    x2 = min(image.shape[1], x2 + x_margin)
    y2 = min(image.shape[0], y2 + y_margin)
    
    # Extraer rostro
    face = image[y:y2, x:x2]
    
    if face.size == 0:
        return None
    
    # Redimensionar a tamaño estándar
    face_resized = cv2.resize(face, (output_size, output_size))
    
    return face_resized

def process_images():
    """
    Procesa todas las imágenes raw, detecta rostros y guarda versiones alineadas.
    """
    print("=" * 60)
    print("🔍 DETECCIÓN Y ALINEACIÓN DE ROSTROS CON MTCNN")
    print("=" * 60)
    
    # Inicializar detector MTCNN
    print("\n⏳ Inicializando detector MTCNN...")
    detector = MTCNN()
    print("✅ Detector inicializado")
    
    # Obtener lista de personas
    persons = [d for d in os.listdir(RAW_IMAGES_DIR) 
               if os.path.isdir(os.path.join(RAW_IMAGES_DIR, d))]
    
    if not persons:
        print("\n❌ No se encontraron carpetas de personas en:", RAW_IMAGES_DIR)
        print("💡 Primero ejecuta: 01_capture_images.py")
        return
    
    print(f"\n👥 Personas encontradas: {len(persons)}")
    for person in persons:
        print(f"   - {person}")
    
    total_processed = 0
    total_faces_detected = 0
    
    # Procesar cada persona
    for person_name in persons:
        print(f"\n{'='*60}")
        print(f"👤 Procesando: {person_name}")
        print(f"{'='*60}")
        
        # Directorios
        person_raw_dir = os.path.join(RAW_IMAGES_DIR, person_name)
        person_aligned_dir = os.path.join(ALIGNED_FACES_DIR, person_name)
        os.makedirs(person_aligned_dir, exist_ok=True)
        
        # Obtener imágenes
        image_files = [f for f in os.listdir(person_raw_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📊 Imágenes a procesar: {len(image_files)}")
        
        processed = 0
        faces_detected = 0
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(person_raw_dir, image_file)
            
            try:
                # Leer imagen
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️  [{idx}/{len(image_files)}] No se pudo leer: {image_file}")
                    continue
                
                # Convertir BGR a RGB (MTCNN espera RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detectar rostros
                detections = detector.detect_faces(image_rgb)
                
                if not detections:
                    print(f"⚠️  [{idx}/{len(image_files)}] No se detectó rostro en: {image_file}")
                    continue
                
                # Filtrar por confianza
                valid_detections = [d for d in detections 
                                   if d['confidence'] >= CONFIDENCE_THRESHOLD]
                
                if not valid_detections:
                    print(f"⚠️  [{idx}/{len(image_files)}] Confianza baja en: {image_file}")
                    continue
                
                # Usar la detección con mayor confianza
                best_detection = max(valid_detections, key=lambda x: x['confidence'])
                
                # Alinear rostro
                aligned_face = align_face(image_rgb, best_detection)
                
                if aligned_face is None:
                    print(f"⚠️  [{idx}/{len(image_files)}] Error al alinear: {image_file}")
                    continue
                
                # Guardar rostro alineado
                output_filename = f"aligned_{os.path.splitext(image_file)[0]}.jpg"
                output_path = os.path.join(person_aligned_dir, output_filename)
                
                # Convertir RGB a BGR para guardar con OpenCV
                aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, aligned_face_bgr)
                
                processed += 1
                faces_detected += 1
                
                print(f"✅ [{idx}/{len(image_files)}] Procesado: {image_file} "
                      f"(confianza: {best_detection['confidence']:.2f})")
                
            except Exception as e:
                print(f"❌ [{idx}/{len(image_files)}] Error en {image_file}: {str(e)}")
        
        print(f"\n📊 Resumen para {person_name}:")
        print(f"   - Procesadas exitosamente: {processed}/{len(image_files)}")
        print(f"   - Rostros detectados: {faces_detected}")
        print(f"   - Guardados en: {person_aligned_dir}")
        
        total_processed += processed
        total_faces_detected += faces_detected
    
    print(f"\n{'='*60}")
    print(f"✅ PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"📊 Total de imágenes procesadas: {total_processed}")
    print(f"👥 Total de rostros detectados: {total_faces_detected}")
    print(f"📁 Rostros guardados en: {ALIGNED_FACES_DIR}")
    print(f"\n🚀 Siguiente paso: Ejecuta 03_generate_embeddings.py")
    print("=" * 60)

if __name__ == "__main__":
    process_images()
