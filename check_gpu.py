"""
Script para verificar si TensorFlow puede acceder a la GPU.
Muestra información sobre los dispositivos disponibles y configuración.

Uso:
    python check_gpu.py
"""

import tensorflow as tf
import sys

def check_gpu():
    """Verifica la disponibilidad y configuración de GPU."""
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE GPU PARA TENSORFLOW")
    print("=" * 60)
    
    # Versión de TensorFlow
    print(f"\n📦 TensorFlow versión: {tf.__version__}")
    
    # Verificar si TensorFlow fue compilado con soporte CUDA
    print(f"\n🔧 Compilado con CUDA: {tf.test.is_built_with_cuda()}")
    
    # Listar dispositivos físicos
    print("\n💻 Dispositivos físicos disponibles:")
    physical_devices = tf.config.list_physical_devices()
    if not physical_devices:
        print("   ❌ No se encontraron dispositivos")
    else:
        for device in physical_devices:
            print(f"   - {device.device_type}: {device.name}")
    
    # Verificar GPUs disponibles
    print("\n🎮 GPUs disponibles:")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("   ❌ No se encontraron GPUs")
        print("\n💡 Para habilitar GPU necesitas:")
        print("   1. GPU NVIDIA con soporte CUDA")
        print("   2. Controladores NVIDIA actualizados")
        print("   3. CUDA Toolkit 11.8+ y cuDNN 8.6+")
        print("   4. TensorFlow con soporte GPU:")
        print("      pip uninstall tensorflow")
        print("      pip install tensorflow[and-cuda]==2.17.1")
        return False
    else:
        for i, gpu in enumerate(gpus):
            print(f"   ✅ GPU {i}: {gpu.name}")
            try:
                # Obtener detalles de la GPU
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"      Detalles: {gpu_details}")
            except:
                pass
    
    # Test de GPU
    print("\n🧪 Realizando test de GPU...")
    try:
        with tf.device('/GPU:0'):
            # Crear tensores de prueba
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"   ✅ Operación matricial en GPU exitosa")
            print(f"   Resultado shape: {c.shape}")
    except RuntimeError as e:
        print(f"   ❌ Error al usar GPU: {e}")
        return False
    
    # Configuración de memoria
    print("\n💾 Configuración de memoria GPU:")
    for gpu in gpus:
        try:
            # Habilitar crecimiento de memoria (evita reservar toda la GPU)
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ Crecimiento de memoria habilitado para {gpu.name}")
        except Exception as e:
            print(f"   ⚠️  No se pudo configurar memoria: {e}")
    
    # Información de asignación de dispositivos
    print("\n🗺️  Dispositivos lógicos:")
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        print(f"   - {device.device_type}: {device.name}")
    
    print("\n" + "=" * 60)
    print("✅ GPU CONFIGURADA CORRECTAMENTE")
    print("=" * 60)
    print("\n💡 TensorFlow usará GPU automáticamente")
    print("   El reconocimiento facial será ~10-50x más rápido")
    
    return True

def show_cuda_info():
    """Muestra información adicional sobre CUDA si está disponible."""
    print("\n" + "=" * 60)
    print("🔧 INFORMACIÓN DE CUDA")
    print("=" * 60)
    
    try:
        import subprocess
        
        # Intentar obtener versión de nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu',
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("\n📊 Información de GPU (nvidia-smi):")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 5:
                    print(f"\n   GPU {i}:")
                    print(f"   - Nombre: {parts[0]}")
                    print(f"   - Driver: {parts[1]}")
                    print(f"   - Memoria Total: {parts[2]}")
                    print(f"   - Memoria Libre: {parts[3]}")
                    print(f"   - Utilización: {parts[4]}")
        else:
            print("\n⚠️  nvidia-smi no disponible")
            print("   Asegúrate de tener los drivers NVIDIA instalados")
            
    except FileNotFoundError:
        print("\n⚠️  nvidia-smi no encontrado")
        print("   Instala los controladores NVIDIA desde:")
        print("   https://www.nvidia.com/Download/index.aspx")
    except Exception as e:
        print(f"\n⚠️  Error al obtener información de CUDA: {e}")

def benchmark_cpu_vs_gpu():
    """Compara rendimiento CPU vs GPU."""
    print("\n" + "=" * 60)
    print("⚡ BENCHMARK: CPU vs GPU")
    print("=" * 60)
    
    import time
    
    # Crear datos de prueba
    size = 5000
    iterations = 10
    
    print(f"\nGenerando matrices {size}x{size} para benchmark...")
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    
    # Test en CPU
    print("\n🖥️  Probando en CPU...")
    with tf.device('/CPU:0'):
        start = time.time()
        for _ in range(iterations):
            c = tf.matmul(a, b)
        cpu_time = (time.time() - start) / iterations
        print(f"   Tiempo promedio: {cpu_time*1000:.2f} ms")
    
    # Test en GPU (si está disponible)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\n🎮 Probando en GPU...")
        with tf.device('/GPU:0'):
            # Warmup
            _ = tf.matmul(a, b)
            
            start = time.time()
            for _ in range(iterations):
                c = tf.matmul(a, b)
            gpu_time = (time.time() - start) / iterations
            print(f"   Tiempo promedio: {gpu_time*1000:.2f} ms")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 Aceleración GPU: {speedup:.2f}x más rápido que CPU")
        
        if speedup < 2:
            print("\n⚠️  La aceleración es menor de lo esperado.")
            print("   Esto puede ser normal para operaciones pequeñas.")
            print("   En reconocimiento facial real la mejora será mayor.")
    else:
        print("\n❌ No hay GPU disponible para comparar")

def main():
    """Función principal."""
    # Verificar GPU
    gpu_available = check_gpu()
    
    # Mostrar información de CUDA
    show_cuda_info()
    
    # Benchmark si hay GPU
    if gpu_available:
        try:
            benchmark_cpu_vs_gpu()
        except Exception as e:
            print(f"\n⚠️  No se pudo ejecutar benchmark: {e}")
    
    print("\n" + "=" * 60)
    if gpu_available:
        print("🎉 Sistema listo para usar GPU")
        print("   El reconocimiento facial usará CUDA automáticamente")
    else:
        print("ℹ️  Sistema funcionará con CPU")
        print("   Considera instalar soporte GPU para mejor rendimiento")
    print("=" * 60)

if __name__ == "__main__":
    main()
