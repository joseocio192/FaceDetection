"""
Script para generar embeddings de rostros usando FaceNet.
Los embeddings son vectores de 128 dimensiones que representan características únicas de cada rostro.

FaceNet es una red neuronal entrenada que convierte rostros en vectores de características.
Rostros similares tendrán embeddings cercanos en el espacio vectorial.

Uso:
    python 03_generate_embeddings.py
"""

import os
import numpy as np
import pickle
from keras_facenet import FaceNet
import cv2
from config import ALIGNED_FACES_DIR, EMBEDDINGS_DIR, IMAGE_SIZE

def load_facenet_model():
    """
    Carga el modelo FaceNet pre-entrenado.
    
    Returns:
        FaceNet: Modelo FaceNet cargado
    """
    print("⏳ Cargando modelo FaceNet...")
    embedder = FaceNet()
    print("✅ Modelo FaceNet cargado correctamente")
    print(f"📊 Dimensión de embeddings: 128")
    return embedder

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesa una imagen para FaceNet.
    
    Args:
        image_path (str): Ruta a la imagen
        target_size (int): Tamaño objetivo
        
    Returns:
        np.array: Imagen preprocesada
    """
    # Leer imagen
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convertir BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar si es necesario
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size))
    
    return image

def generate_embeddings():
    """
    Genera embeddings para todas las imágenes alineadas y los guarda en archivo.
    """
    print("=" * 60)
    print("🧠 GENERACIÓN DE EMBEDDINGS CON FACENET")
    print("=" * 60)
    
    # Cargar modelo
    embedder = load_facenet_model()
    
    # Verificar que existan rostros alineados
    if not os.path.exists(ALIGNED_FACES_DIR):
        print(f"\n❌ No se encontró el directorio: {ALIGNED_FACES_DIR}")
        print("💡 Primero ejecuta: 02_detect_and_align_faces.py")
        return
    
    # Obtener lista de personas
    persons = [d for d in os.listdir(ALIGNED_FACES_DIR) 
               if os.path.isdir(os.path.join(ALIGNED_FACES_DIR, d))]
    
    if not persons:
        print(f"\n❌ No se encontraron carpetas de personas en: {ALIGNED_FACES_DIR}")
        return
    
    print(f"\n👥 Personas encontradas: {len(persons)}")
    for person in persons:
        print(f"   - {person}")
    
    # Estructuras para guardar embeddings
    embeddings_database = {
        'embeddings': [],  # Lista de vectores de embeddings
        'labels': [],      # Lista de nombres correspondientes
        'image_paths': []  # Rutas de las imágenes (para referencia)
    }
    
    total_embeddings = 0
    
    # Procesar cada persona
    for person_name in persons:
        print(f"\n{'='*60}")
        print(f"👤 Procesando embeddings para: {person_name}")
        print(f"{'='*60}")
        
        person_dir = os.path.join(ALIGNED_FACES_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📊 Imágenes a procesar: {len(image_files)}")
        
        person_embeddings = []
        processed = 0
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(person_dir, image_file)
            
            try:
                # Cargar y preprocesar imagen
                image = preprocess_image(image_path)
                
                if image is None:
                    print(f"⚠️  [{idx}/{len(image_files)}] No se pudo cargar: {image_file}")
                    continue
                
                # Generar embedding (FaceNet espera batch de imágenes)
                # Expandir dimensiones: (160, 160, 3) -> (1, 160, 160, 3)
                image_batch = np.expand_dims(image, axis=0)
                
                # Obtener embedding
                embedding = embedder.embeddings(image_batch)[0]
                
                # Normalizar embedding (importante para comparación con distancia coseno)
                embedding = embedding / np.linalg.norm(embedding)
                
                # Guardar
                embeddings_database['embeddings'].append(embedding)
                embeddings_database['labels'].append(person_name)
                embeddings_database['image_paths'].append(image_path)
                
                person_embeddings.append(embedding)
                processed += 1
                
                print(f"✅ [{idx}/{len(image_files)}] Embedding generado: {image_file}")
                
            except Exception as e:
                print(f"❌ [{idx}/{len(image_files)}] Error en {image_file}: {str(e)}")
        
        # Calcular estadísticas para esta persona
        if person_embeddings:
            person_embeddings_array = np.array(person_embeddings)
            mean_embedding = np.mean(person_embeddings_array, axis=0)
            std_embedding = np.std(person_embeddings_array, axis=0)
            
            print(f"\n📊 Resumen para {person_name}:")
            print(f"   - Embeddings generados: {processed}/{len(image_files)}")
            print(f"   - Dimensión del embedding: {len(mean_embedding)}")
            print(f"   - Desviación estándar promedio: {np.mean(std_embedding):.4f}")
            
            total_embeddings += processed
        else:
            print(f"\n⚠️  No se generaron embeddings para {person_name}")
    
    # Convertir listas a arrays de NumPy
    embeddings_database['embeddings'] = np.array(embeddings_database['embeddings'])
    embeddings_database['labels'] = np.array(embeddings_database['labels'])
    embeddings_database['image_paths'] = np.array(embeddings_database['image_paths'])
    
    # Guardar base de datos de embeddings
    database_path = os.path.join(EMBEDDINGS_DIR, 'face_embeddings.pkl')
    
    with open(database_path, 'wb') as f:
        pickle.dump(embeddings_database, f)
    
    print(f"\n{'='*60}")
    print(f"✅ EMBEDDINGS GENERADOS EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"📊 Total de embeddings generados: {total_embeddings}")
    print(f"👥 Total de personas: {len(persons)}")
    print(f"💾 Base de datos guardada en: {database_path}")
    print(f"📦 Tamaño de la base de datos: {embeddings_database['embeddings'].shape}")
    print(f"\n🚀 Siguiente paso: Ejecuta 04_recognition_realtime.py")
    print("=" * 60)

def verify_embeddings():
    """
    Verifica la base de datos de embeddings cargándola y mostrando estadísticas.
    """
    database_path = os.path.join(EMBEDDINGS_DIR, 'face_embeddings.pkl')
    
    if not os.path.exists(database_path):
        print("❌ No se encontró la base de datos de embeddings")
        return
    
    print("\n" + "=" * 60)
    print("🔍 VERIFICACIÓN DE BASE DE DATOS")
    print("=" * 60)
    
    with open(database_path, 'rb') as f:
        db = pickle.load(f)
    
    print(f"\n📊 Estadísticas:")
    print(f"   - Total de embeddings: {len(db['embeddings'])}")
    print(f"   - Dimensión: {db['embeddings'].shape[1]}")
    print(f"   - Personas únicas: {len(np.unique(db['labels']))}")
    
    print(f"\n👥 Distribución por persona:")
    unique_labels, counts = np.unique(db['labels'], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   - {label}: {count} embeddings")
    
    print("\n✅ Base de datos verificada correctamente")

if __name__ == "__main__":
    # Generar embeddings
    generate_embeddings()
    
    # Verificar resultados
    verify_embeddings()
