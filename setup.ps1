# 🚀 Script de Instalación y Setup - Face Recognition

# Este script automatiza la instalación completa del proyecto

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59)
Write-Host "🎭 INSTALACIÓN DEL SISTEMA DE RECONOCIMIENTO FACIAL" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host ""

# Verificar Python
Write-Host "🔍 Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python no está instalado o no está en PATH" -ForegroundColor Red
    Write-Host "💡 Descarga Python desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Verificar versión de Python (mínimo 3.8)
$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([float]$version -lt 3.8) {
    Write-Host "⚠️  Advertencia: Python $version detectado. Se recomienda Python 3.8 o superior" -ForegroundColor Yellow
}

Write-Host ""

# Crear entorno virtual
Write-Host "📦 Creando entorno virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  El entorno virtual ya existe. ¿Deseas recrearlo? (S/N)" -ForegroundColor Yellow
    $respuesta = Read-Host
    if ($respuesta -eq "S" -or $respuesta -eq "s") {
        Remove-Item -Recurse -Force venv
        python -m venv venv
        Write-Host "✅ Entorno virtual recreado" -ForegroundColor Green
    } else {
        Write-Host "⏭️  Usando entorno virtual existente" -ForegroundColor Cyan
    }
} else {
    python -m venv venv
    Write-Host "✅ Entorno virtual creado" -ForegroundColor Green
}

Write-Host ""

# Activar entorno virtual
Write-Host "🔌 Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  No se pudo activar el entorno. Intentando cambiar política de ejecución..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "✅ Entorno virtual activado" -ForegroundColor Green
Write-Host ""

# Actualizar pip
Write-Host "⬆️  Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✅ pip actualizado" -ForegroundColor Green
Write-Host ""

# Instalar dependencias
Write-Host "📥 Instalando dependencias..." -ForegroundColor Yellow
Write-Host "⏳ Este proceso puede tardar varios minutos (TensorFlow es ~500 MB)..." -ForegroundColor Cyan
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Todas las dependencias instaladas correctamente" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Error al instalar dependencias" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Verificar instalación
Write-Host "🔍 Verificando instalación..." -ForegroundColor Yellow

$verificacion = python -c @"
try:
    import tensorflow as tf
    import cv2
    import mtcnn
    import numpy as np
    from keras_facenet import FaceNet
    print('✅ TODAS LAS BIBLIOTECAS IMPORTADAS CORRECTAMENTE')
    print(f'   - TensorFlow: {tf.__version__}')
    print(f'   - OpenCV: {cv2.__version__}')
    print(f'   - NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"@

Write-Host $verificacion

Write-Host ""

# Crear directorios
Write-Host "📁 Creando estructura de directorios..." -ForegroundColor Yellow

$directorios = @(
    "data",
    "data\raw_images",
    "data\aligned_faces",
    "data\embeddings",
    "models"
)

foreach ($dir in $directorios) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ Creado: $dir" -ForegroundColor Green
    } else {
        Write-Host "   ⏭️  Ya existe: $dir" -ForegroundColor Cyan
    }
}

Write-Host ""

# Verificar webcam
Write-Host "📸 Verificando acceso a webcam..." -ForegroundColor Yellow

$webcamTest = python -c @"
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Webcam detectada y accesible')
    cap.release()
else:
    print('⚠️  No se pudo acceder a la webcam')
    print('💡 Verifica permisos en: Configuración > Privacidad > Cámara')
"@

Write-Host $webcamTest
Write-Host ""

# Resumen final
Write-Host ("=" * 60)
Write-Host "🎉 ¡INSTALACIÓN COMPLETADA!" -ForegroundColor Green
Write-Host ("=" * 60)
Write-Host ""

Write-Host "📋 SIGUIENTE PASO:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1️⃣  Captura imágenes de rostros:" -ForegroundColor White
Write-Host "   python 01_capture_images.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "2️⃣  Detecta y alinea rostros:" -ForegroundColor White
Write-Host "   python 02_detect_and_align_faces.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3️⃣  Genera embeddings:" -ForegroundColor White
Write-Host "   python 03_generate_embeddings.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "4️⃣  Reconocimiento en tiempo real:" -ForegroundColor White
Write-Host "   python 04_recognition_realtime.py" -ForegroundColor Yellow
Write-Host ""

Write-Host "📚 Documentación completa disponible en:" -ForegroundColor Cyan
Write-Host "   - README.md (guía de uso)" -ForegroundColor White
Write-Host "   - GUIDE.md (guía técnica)" -ForegroundColor White
Write-Host ""

Write-Host ("=" * 60)
Write-Host "🚀 ¡Listo para empezar! Happy coding!" -ForegroundColor Green
Write-Host ("=" * 60)
