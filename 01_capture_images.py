"""
Script para capturar imágenes de rostros desde la webcam.
Guarda las imágenes organizadas por carpetas con el nombre de cada persona.

Uso:
    python 01_capture_images.py

Controles:
    - Presiona 'c' para capturar una imagen
    - Presiona 'q' para salir
"""

import cv2
import os
from datetime import datetime
from config import RAW_IMAGES_DIR, MIN_IMAGES_PER_PERSON

def capture_images_for_person(person_name):
    """
    Captura imágenes de una persona usando la webcam.
    
    Args:
        person_name (str): Nombre de la persona a registrar
    """
    # Crear directorio para esta persona
    person_dir = os.path.join(RAW_IMAGES_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo acceder a la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Cargar detector de rostros Haar Cascade (para preview)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    image_count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
    print(f"\n📸 Capturando imágenes para: {person_name}")
    print(f"📊 Imágenes actuales: {image_count}")
    print(f"🎯 Objetivo mínimo: {MIN_IMAGES_PER_PERSON} imágenes")
    print("\n⌨️  Controles:")
    print("   - Presiona 'c' para capturar")
    print("   - Presiona 'q' para salir")
    print("\n💡 Consejos:")
    print("   - Varía tus expresiones faciales")
    print("   - Cambia el ángulo de tu rostro")
    print("   - Prueba con diferentes iluminaciones")
    print("   - Mantén el rostro centrado en el recuadro verde\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer el frame")
            break
        
        # Crear copia para mostrar
        display_frame = frame.copy()
        
        # Detectar rostros para feedback visual
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Dibujar rectángulos alrededor de rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Rostro detectado", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar contador de imágenes
        cv2.putText(display_frame, f"Imagenes: {image_count}/{MIN_IMAGES_PER_PERSON}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Persona: {person_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Captura de Rostros - Presiona C para capturar, Q para salir', 
                   display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capturar imagen
        if key == ord('c'):
            if len(faces) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{person_name}_{timestamp}.jpg"
                filepath = os.path.join(person_dir, filename)
                
                cv2.imwrite(filepath, frame)
                image_count += 1
                print(f"✅ Imagen {image_count} capturada: {filename}")
                
                if image_count >= MIN_IMAGES_PER_PERSON:
                    print(f"🎉 ¡Objetivo alcanzado! {image_count} imágenes capturadas")
            else:
                print("⚠️  No se detectó ningún rostro. Intenta de nuevo.")
        
        # Salir
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Sesión finalizada")
    print(f"📊 Total de imágenes capturadas: {image_count}")
    print(f"📁 Guardadas en: {person_dir}")

def main():
    """Función principal para capturar imágenes."""
    print("=" * 60)
    print("🎭 SISTEMA DE CAPTURA DE ROSTROS")
    print("=" * 60)
    
    # Solicitar nombre de la persona
    person_name = input("\n👤 Ingresa el nombre de la persona: ").strip()
    
    if not person_name:
        print("❌ Error: Debes ingresar un nombre válido")
        return
    
    # Validar que el nombre no contenga caracteres especiales
    if not person_name.replace(" ", "").replace("_", "").isalnum():
        print("❌ Error: El nombre solo puede contener letras, números, espacios y guiones bajos")
        return
    
    # Reemplazar espacios con guiones bajos
    person_name = person_name.replace(" ", "_")
    
    # Capturar imágenes
    capture_images_for_person(person_name)
    
    print("\n" + "=" * 60)
    print("🚀 Siguiente paso: Ejecuta 02_detect_and_align_faces.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
