#!/bin/bash
# Instalar distutils
apt-get update && apt-get install -y python3-distutils

# Instalar las dependencias del proyecto
pip install -r requirements.txt