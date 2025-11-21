@echo off
echo ========================================
echo Compilando DQN con CUDA
echo ========================================

REM Verificar que estamos en el directorio correcto
if not exist "src\main_dqn.cu" (
    echo Error: No se encuentra src\main_dqn.cu
    echo Asegurate de estar en el directorio raiz del proyecto
    pause
    exit /b 1
)

REM Crear directorio bin si no existe
if not exist "bin" mkdir bin

echo.
echo Compilando...
echo.

REM Compilar usando nvcc directamente
nvcc -arch=sm_50 ^
    -I include ^
    -o bin\dqn_train.exe ^
    src\main_dqn.cu ^
    src\neural_network.cu ^
    src\replay_buffer.cu ^
    src\dqn.cu ^
    -lcurand

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Error en la compilacion!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Compilacion exitosa!
echo ========================================
echo.
echo Ejecutando DQN Training...
echo.

bin\dqn_train.exe

echo.
echo ========================================
echo Programa terminado
echo ========================================
pause
