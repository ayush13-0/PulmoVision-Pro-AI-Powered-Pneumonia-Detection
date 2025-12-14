<!-- PROJECT BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/PulmoVision%20Pro-AI%20Powered%20Pneumonia%20Detection-blueviolet?style=for-the-badge&logo=python&logoColor=white" />
</p>

<!-- TITLE -->
<h1 align="center">🩺🧠 PulmoVision Pro — AI-Powered Pneumonia Detection</h1>

<!-- TAGLINE -->
<p align="center">
  <b>Deep Learning–based pneumonia detection from chest X-rays with Grad-CAM interpretability</b>
</p>

<!-- CORE TECH BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNNs%20%7C%20Transfer%20Learning-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-DenseNet121%20%7C%20ResNet50-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Medical%20AI-X--Ray%20Classification-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%202.14-blue?style=for-the-badge&logo=tensorflow" />
</p>

<!-- DATA / LANGUAGE BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Data-Medical%20Imaging-lightgrey?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/UI-Streamlit-success?style=for-the-badge&logo=streamlit" />
</p>

<!-- ADVANCED / SYSTEM BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20Dashboard-black?style=for-the-badge" />
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" />
</p>

<!-- PROJECT TYPE / LICENSE -->
<p align="center">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20AI-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge" />
</p>

<!-- OPTIONAL DEMO BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/Live%20Demo-Streamlit%20App-success?style=for-the-badge&logo=streamlit" />
</p>


# 🩺 PulmoVision Pro — AI-Powered Pneumonia Detection
Tagline:


# 🔍 Project Overview
Pneumonia is a lung infection causing inflammation in the air sacs, making early detection critical. Manual diagnosis using chest X-rays can be slow and error-prone.
PulmoVision Pro automates this process using CNNs to classify Normal vs Pneumonia X-ray images, providing interpretable results with Grad-CAM heatmaps.
**Goals:**
- Automate pneumonia detection from chest X-rays 🖼️
- Provide interpretable results for clinicians 🔥
- Deploy predictions on a professional Streamlit dashboard 💻

# 🎯 Objectives
- Preprocess chest X-ray images 🖼️
- Train & fine-tune CNNs (DenseNet121, ResNet50) 🧠

**Evaluate models using:**
- Accuracy ✅
- Precision 🎯
- Recall 📊
- AUC 📈

**📊Visualize results:**
- Training/Validation curves
- Confusion Matrix heatmaps
- ROC curves
- Grad-CAM overlays 🌟
- Compare DenseNet vs ResNet performance ⚖️
- Deploy predictions on interactive Streamlit dashboard

# 📁 Dataset
- **Kermany Chest X-Ray Pneumonia Dataset** (~5 GB)
- Balanced classes: Normal & Pneumonia ⚖️
- Preprocessed and ready for CNN training/testing

**Kaggle link: Chest X-Ray Images (Pneumonia)**
Folder Structure Example:

<pre> chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/ </pre>


# 🧠 Model Architecture
**Model	Description**
- DenseNet121 🔹	Dense connections for feature reuse; excellent for detecting X-ray textures
- ResNet50 🔹	Residual connections prevent vanishing gradients; performs well on smaller datasets

Custom Classifier:
<pre> GlobalAveragePooling2D → Dense(128, ReLU) → Dense(1, Sigmoid) </pre>

# 🛠️ Streamlit Dashboard Features
Upload single or multiple X-ray images
Select DenseNet121 / ResNet50 models
Display prediction label + probability
Grad-CAM heatmap overlay for interpretability 🔥
Batch prediction with CSV download
Interactive training/validation curves, ROC curve, confusion matrix
Export predictions & heatmaps as PDF reports 📄

# 📊 Evaluation & Expected Results
- Model	Accuracy	Precision	Recall	AUC
- ResNet50	92–95%	High	High	0.95+
- DenseNet121	95–97%	Very High	Very High	0.97+

**Notes:**
- DenseNet121 generally outperforms ResNet50 due to better feature reuse
- Grad-CAM provides visual interpretability 🔥

# 📈 Visualizations
- Training & Validation Curves — Monitor overfitting/underfitting 📈
- Confusion Matrix Heatmap — Professional view of True vs Predicted ✅
- ROC Curve & AUC — Evaluate model performance 📊
- Grad-CAM Overlay — Highlights regions contributing most to predictions 🌟
- Sample Prediction Gallery — Multiple images with predicted labels 🖼️

🎥 Demo (GIF)
<p align="center"> <img src="https://media.giphy.com/media/your-demo-gif-url.gif" alt="PulmoVision Streamlit Demo" width="600"/> </p> > Replace the GIF URL above with your **actual Streamlit app recording** for portfolio-ready visualization.


# 💻 Installation & Setup

**1️⃣ Clone repository**
<pre> git clone https://github.com/ayush13-0/PulmoVision-Pro-AI-Powered-Pneumonia-Detection/tree/main
    cd PulmoVision-Pro </pre>

**2️⃣ Create virtual environment**
<pre> python -m venv venv
    # Linux/Mac
    source venv/bin/activate
    # Windows
    venv\Scripts\activate </pre>

**3️⃣ Install dependencies**
<pre> pip install -r requirements.txt </pre>

**4️⃣ Run Streamlit App**
<pre> streamlit run PulmoVision-Pro.py </pre>

**5️⃣ Download Dataset**

- Kaggle Chest X-Ray Pneumonia
- Organize folder structure as shown above

**🔗 Pre-trained Models**

<pre> 
- DenseNet121: models/densenet_pulmovision.h5
- ResNet50: models/resnet_pulmovision.h5
- Pre-trained on Kermany X-Ray Pneumonia dataset </pre>
:- Ready for inference & Grad-CAM visualization 🔥

# 🏁 Conclusion
**PulmoVision Pro demonstrates:**
- Automated pneumonia detection using CNNs 🧠
- Transfer learning improves medical imaging performance 🚀
- Professional evaluation using accuracy, precision, recall, F1-score, AUC 📊
- Grad-CAM visualization for clinician-friendly interpretability 🔥
- Fully professional, interactive Streamlit dashboard for deployment 🩺

# 📖 References
**Kermany Chest X-Ray Pneumonia Dataset – Kaggle**
- He, K. et al. "Deep Residual Learning for Image Recognition", 2015
- Huang, G. et al. "Densely Connected Convolutional Networks", 2017

# 📌 License
**This project is for educational and research purposes.**
