# 🧠 Brain Tumor Detection (PyTorch + CustomTkinter)

CNN-based binary classifier that detects brain tumors from MRI scans. Includes a PyTorch training script and a CustomTkinter desktop GUI.

## Features
- **Custom CNN Architecture:** Built from scratch using PyTorch.
- **High Accuracy:** Achieved ~96.6% validation accuracy during training.
- **Modern GUI:** A sleek, dark-themed desktop application for easy image uploading and prediction.
- **Binary Classification:** Predicts between "Tumor Detected" and "Healthy".

## Model Performance
The model was trained for 50 epochs with early stopping (patience=5). The best validation accuracy achieved was **96.67% at Epoch 15**. Full training logs are available in `epochs.txt`.

## Files
- `brain_tumor_model.py` — trains the CNN and saves `best_model.pth`
- `brain_tumor_gui.py` — GUI app for testing images
- `metadata.csv`, `metadata_rgb_only.csv` — dataset metadata
- `epochs.txt` — training logs

## Setup
```bash
pip install torch torchvision pillow customtkinter opencv-python scikit-learn
python brain_tumor_model.py   # creates best_model.pth
python brain_tumor_gui.py

## Dataset Layout
```text
Brain_Tumor_Data_Set/
└── Brain_Tumor_Data_Set/
    ├── Healthy/
    └── Brain_Tumor/
```

## Train
```bash
python brain_tumor_model.py
```
This creates `best_model.pth` in the project root.

## Run GUI
```bash
python brain_tumor_gui.py
```
Upload an MRI image (`.jpg`, `.png`, etc.) to get a prediction.

## Note
`best_model.pth` is not included due to GitHub file size limits. Train the model first.

## Disclaimer
For educational and research purposes only. Not a substitute for professional medical advice, diagnosis, or treatment.