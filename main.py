"""
Sistema de Reconocimiento Facial - Menú Principal
Ejecuta todos los pasos del sistema desde una interfaz de menú única.

Uso:
    python main.py
"""

import os
import sys

def print_header():
    """Imprime el encabezado del sistema."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print("🎭 SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 70)
    print()

def print_menu():
    """Muestra el menú principal."""
    print("\n📋 MENÚ PRINCIPAL")
    print("-" * 70)
    print("1️⃣  Capturar imágenes de rostros")
    print("2️⃣  Detectar y alinear rostros (MTCNN)")
    print("3️⃣  Generar embeddings (FaceNet)")
    print("4️⃣  Reconocimiento facial en tiempo real")
    print()
    print("5️⃣  Ejecutar pipeline completo (pasos 2-4)")
    print("6️⃣  Ver estado del sistema")
    print()
    print("0️⃣  Salir")
    print("-" * 70)

def check_system_status():
    """Verifica y muestra el estado del sistema."""
    from config import RAW_IMAGES_DIR, ALIGNED_FACES_DIR, EMBEDDINGS_DIR
    import pickle
    
    print("\n📊 ESTADO DEL SISTEMA")
    print("=" * 70)
    
    # Verificar imágenes raw
    raw_count = 0
    raw_persons = []
    if os.path.exists(RAW_IMAGES_DIR):
        for person in os.listdir(RAW_IMAGES_DIR):
            person_dir = os.path.join(RAW_IMAGES_DIR, person)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    raw_persons.append(person)
                    raw_count += len(images)
    
    print(f"\n1️⃣  Imágenes capturadas:")
    if raw_count > 0:
        print(f"   ✅ {raw_count} imágenes de {len(raw_persons)} persona(s)")
        for person in raw_persons:
            person_dir = os.path.join(RAW_IMAGES_DIR, person)
            count = len([f for f in os.listdir(person_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"      - {person}: {count} imágenes")
    else:
        print(f"   ⚠️  No hay imágenes capturadas")
    
    # Verificar rostros alineados
    aligned_count = 0
    aligned_persons = []
    if os.path.exists(ALIGNED_FACES_DIR):
        for person in os.listdir(ALIGNED_FACES_DIR):
            person_dir = os.path.join(ALIGNED_FACES_DIR, person)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    aligned_persons.append(person)
                    aligned_count += len(images)
    
    print(f"\n2️⃣  Rostros alineados:")
    if aligned_count > 0:
        print(f"   ✅ {aligned_count} rostros de {len(aligned_persons)} persona(s)")
        for person in aligned_persons:
            person_dir = os.path.join(ALIGNED_FACES_DIR, person)
            count = len([f for f in os.listdir(person_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"      - {person}: {count} rostros")
    else:
        print(f"   ⚠️  No hay rostros alineados")
    
    # Verificar embeddings
    embeddings_file = os.path.join(EMBEDDINGS_DIR, 'face_embeddings.pkl')
    print(f"\n3️⃣  Base de datos de embeddings:")
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                db = pickle.load(f)
            import numpy as np
            print(f"   ✅ {len(db['embeddings'])} embeddings generados")
            unique_labels, counts = np.unique(db['labels'], return_counts=True)
            print(f"   👥 {len(unique_labels)} persona(s) registradas:")
            for label, count in zip(unique_labels, counts):
                print(f"      - {label}: {count} embeddings")
        except Exception as e:
            print(f"   ❌ Error al leer base de datos: {str(e)}")
    else:
        print(f"   ⚠️  Base de datos no generada")
    
    # Estado del sistema
    print(f"\n🎯 Estado general:")
    if raw_count > 0 and aligned_count > 0 and os.path.exists(embeddings_file):
        print(f"   ✅ Sistema listo para reconocimiento facial")
    elif raw_count > 0:
        print(f"   ⚠️  Ejecuta el paso 2 para alinear rostros")
    else:
        print(f"   ⚠️  Ejecuta el paso 1 para capturar imágenes")
    
    print("=" * 70)

def run_capture_images():
    """Ejecuta el script de captura de imágenes."""
    print_header()
    print("🎬 Iniciando captura de imágenes...\n")
    try:
        from _01_capture_images import main as capture_main
        capture_main()
    except ImportError:
        # Si el import con guión bajo falla, intentar sin él
        import importlib.util
        spec = importlib.util.spec_from_file_location("capture", "01_capture_images.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

def run_detect_align():
    """Ejecuta el script de detección y alineación."""
    print_header()
    print("🔍 Iniciando detección y alineación de rostros...\n")
    try:
        from _02_detect_and_align_faces import process_images
        process_images()
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("detect", "02_detect_and_align_faces.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.process_images()

def run_generate_embeddings():
    """Ejecuta el script de generación de embeddings."""
    print_header()
    print("🧠 Iniciando generación de embeddings...\n")
    try:
        from _03_generate_embeddings import generate_embeddings, verify_embeddings
        generate_embeddings()
        verify_embeddings()
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("embeddings", "03_generate_embeddings.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.generate_embeddings()
        module.verify_embeddings()

def run_realtime_recognition():
    """Ejecuta el sistema de reconocimiento en tiempo real."""
    print_header()
    print("🎥 Iniciando reconocimiento en tiempo real...\n")
    try:
        from _04_recognition_realtime import main as recognition_main
        recognition_main()
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("recognition", "04_recognition_realtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

def run_full_pipeline():
    """Ejecuta el pipeline completo (pasos 2-4)."""
    print_header()
    print("🚀 EJECUTANDO PIPELINE COMPLETO")
    print("=" * 70)
    
    print("\n📋 Se ejecutarán los siguientes pasos:")
    print("   1. Detectar y alinear rostros")
    print("   2. Generar embeddings")
    print("   3. Reconocimiento en tiempo real")
    
    confirm = input("\n¿Continuar? (s/n): ").strip().lower()
    if confirm != 's':
        print("❌ Pipeline cancelado")
        return
    
    # Paso 2
    print("\n" + "=" * 70)
    print("PASO 1/3: Detección y alineación")
    print("=" * 70)
    input("\nPresiona Enter para continuar...")
    run_detect_align()
    
    # Paso 3
    print("\n" + "=" * 70)
    print("PASO 2/3: Generación de embeddings")
    print("=" * 70)
    input("\nPresiona Enter para continuar...")
    run_generate_embeddings()
    
    # Paso 4
    print("\n" + "=" * 70)
    print("PASO 3/3: Reconocimiento en tiempo real")
    print("=" * 70)
    input("\nPresiona Enter para continuar...")
    run_realtime_recognition()
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 70)

def main():
    """Función principal del menú."""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Selecciona una opción: ").strip()
            
            if choice == '1':
                run_capture_images()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '2':
                run_detect_align()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '3':
                run_generate_embeddings()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '4':
                run_realtime_recognition()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '5':
                run_full_pipeline()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '6':
                print_header()
                check_system_status()
                input("\n✅ Presiona Enter para volver al menú...")
                
            elif choice == '0':
                print_header()
                print("👋 ¡Hasta luego!")
                print("=" * 70)
                sys.exit(0)
                
            else:
                print("\n❌ Opción no válida. Por favor, selecciona una opción del menú.")
                input("Presiona Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            print("=" * 70)
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            input("\nPresiona Enter para volver al menú...")

if __name__ == "__main__":
    main()
