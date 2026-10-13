@echo off
chcp 65001 >nul
title Construir TROPI SNACKS EXE
echo Este archivo es para COMPILAR la aplicacion en Windows.
echo Requiere Python instalado solo en la computadora que compila.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean --onefile --windowed --name Tropi_Snacks tropi_snacks.py
if errorlevel 1 (
  echo.
  echo Hubo un error al compilar. Revisa los mensajes anteriores.
  pause
  exit /b 1
)
echo.
echo EXE creado en la carpeta dist: dist\Tropi_Snacks.exe
pause
