#!/bin/bash

echo "================================================================================"
echo "         EMAIL SPAM DETECTION - AUTOMATED SETUP SCRIPT"
echo "================================================================================"
echo ""

echo "[Step 1/3] Installing Python packages..."
echo ""
pip3 install flask scikit-learn pandas numpy matplotlib seaborn nltk joblib --break-system-packages
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install packages!"
    echo "Try running: pip3 install -r requirements.txt --break-system-packages"
    exit 1
fi

echo ""
echo "================================================================================"
echo "[Step 2/3] Training machine learning models..."
echo "This will take 2-3 minutes. Please wait..."
echo "================================================================================"
echo ""
python3 train_and_save_model.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Model training failed!"
    echo "Make sure spam.csv is in the same folder."
    exit 1
fi

echo ""
echo "================================================================================"
echo "[Step 3/3] Setup complete! Starting web application..."
echo "================================================================================"
echo ""
echo "Your spam detection system is ready!"
echo ""
echo "Open your browser and go to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server when done."
echo "================================================================================"
echo ""

python3 app.py
