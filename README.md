# 🧠 A Unified Deep Learning Framework for Lip Reading and Deep Fake Audio Classification

## 📌 Project Overview
This project introduces a **deep learning-based multimodal framework** that combines **video-based lip reading** and **audio-based deepfake detection**.  
It leverages a **LipNet model** trained on the **GRID dataset** for visual speech recognition, and a **spectrogram-based CNN classifier** trained on the **Deep Voice Kaggle dataset** for detecting fake audio.  
The system is packaged into an interactive **Streamlit web app** for real-time, side-by-side video and prediction analysis.

---

## 🗂️ Dataset

### 🎥 Lip Reading
- **Dataset**: [GRID Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- **Content**: Over 30,000 video utterances from 34 speakers
- **Pretrained Model**: [LipNet](https://github.com/rizkiarm/LipNet)
- **Preprocessing**: Mouth ROI extraction, landmark detection, frame normalization

### 🔊 Deepfake Audio Detection
- **Dataset**: [Deep Voice: Deepfake Voice Recognition (Kaggle)](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
- **Content**: Real vs. synthetic audio samples across multiple speakers
- **Preprocessing**: Mel-spectrogram extraction, normalization, padding

---

## 🧠 Model Architecture

### 🎥 Lip Reading (Visual)
- **Backbone**: CNN → Bi-GRU → Dense + CTC
- **Input**: Sequence of 75 lip landmark frames
- **Output**: Decoded character tokens (CTC)

### 🔊 Deepfake Detection (Audio)
- **Backbone**: 2D CNN on spectrograms
- **Input**: 128×128 Mel-spectrogram images
- **Output**: Binary classification (Real or Fake)

### 🔁 Multimodal Fusion
- Visual and audio predictions are fused using a **late decision-level fusion strategy** to enhance reliability in cross-modal authentication.

---

## ✅ Evaluation Metrics

### Lip Reading
- **Word Error Rate (WER)**: 1.74%
- **Character Error Rate (CER)**: 0.65%

### Deepfake Detection
- **Accuracy**: 98.1%
- **Precision / Recall / F1-Score**: Evaluated per class

---

## ⚙️ Features
- 🎬 Upload and play GRID-style video samples
- 📸 Visualize extracted lip landmarks (GIF)
- 🧠 Real-time lip reading using LipNet
- 🔊 Audio classifier for deepfake detection
- 🧪 CTC decoding and raw token visualization
- 🌐 Built entirely in **Streamlit** for interactivity

---

## 📊 Visual Output
- Side-by-side original video & lip-only input
- Prediction tokens and final decoded sentence
- Deepfake audio prediction (Real/Fake)
- Lip motion GIF generation for model visualization

---

## 📝 Research Publication

This project is part of the **research paper** titled:  
**"A Unified Deep Learning Framework for Lip Reading and Deep Fake Audio Classification"**  
🧾 Presented at **IEEE ISPCC 2025**  
🔗 [IEEE Xplore Link](https://ieeexplore.ieee.org/document/11039390)

---

## 🧠 Patent

**Patent Title**: *System and Method for Real-Time Audiovisual Deepfake Detection Using Lip Reading and Audio Feature Fusion*  
- 📄 **Application No.**: 202541059722  
- 🗓️ **Filing Date**: June 14, 2025  
- 📅 **Publication Date**: July 04, 2025  
- 🏢 **Patent Office**: India  

---

## 📚 Author
- **Developed by:** Omkar Bhosale

## 📫 Contact

- 📧 **Email**: omkarbhosale1623@gmail.com  
- 🔗 **LinkedIn**: [linkedin.com/in/omkar-bhosale-75a18122a](https://www.linkedin.com/in/omkar-bhosale-75a18122a/)  
- 💻 **GitHub**: [github.com/omkarbhosale1623](https://github.com/omkarbhosale1623)

