# Acoustic-based-object-detection
Developed an audio-based object detection system capable of identifying sound-emitting objects (Drone / TV / Fan) using uploaded audio files instead of cameras.
dataset:https://drive.google.com/drive/folders/1bt0wGgQOHOJqAjn0mqg7vE3PTAKAp0Ac?usp=sharing


Designed a complete pipeline: audio upload → preprocessing → STFT/Mel spectrogram generation → YOLOv8 inference → output visualization.

Self-recorded fan and TV datasets and augmented them with external drone recordings, producing 1-second normalized training segments and mixed-source samples for noise-robust learning.

Trained a custom YOLOv8 model on spectrogram images to detect and classify sound patterns with bounding-box localization, achieving 0.995 mAP50 across classes.

Implemented denoising and augmentation techniques (spectral gating, frequency & time masking, mixed-audio overlap) to improve detection performance under noisy and overlapping acoustic conditions.

Built an inference module that automatically converts an uploaded audio file into a spectrogram and predicts the sound class with confidence scores, eliminating dependency on camera-based detection.

Tech Stack: Python, Librosa, NumPy, SciPy, Matplotlib, PyTorch, Ultralytics YOLOv8, Google Colab
