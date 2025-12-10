<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/TensorFlow-2.14-orange?style=for-the-badge&logo=tensorflow" /> <img src="https://img.shields.io/badge/Streamlit-1.28-success?style=for-the-badge&logo=streamlit" /> <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge" /> </p>

# 🩺 PulmoVision Pro — AI-Powered Pneumonia Detection
Tagline:
**"Leveraging Deep Learning to detect Pneumonia with high accuracy, interpretability, and professional visualizations."**

🔍 Project Overview
Pneumonia is a lung infection causing inflammation in the air sacs, making early detection critical. Manual diagnosis using chest X-rays can be slow and error-prone.
PulmoVision Pro automates this process using CNNs to classify Normal vs Pneumonia X-ray images, providing interpretable results with Grad-CAM heatmaps.
Goals:
Automate pneumonia detection from chest X-rays 🖼️
Provide interpretable results for clinicians 🔥
Deploy predictions on a professional Streamlit dashboard 💻

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

<h2> Kaggle link: Chest X-Ray Images (Pneumonia) <h2>
Folder Structure Example:

<pre>chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/</pre>


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

<h2>1️⃣ Clone repository<h2>
<pre>
git clone https://github.com/ayush13-0/PulmoVision-Pro-AI-Powered-Pneumonia-Detection/tree/main
cd PulmoVision-Pro
</pre>

<h2>2️⃣ Create virtual environment<h2>
<pre>
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
</pre>

<h2>3️⃣ Install dependencies<h2>
<pre>
pip install -r requirements.txt
</pre>

<h2>4️⃣ Run Streamlit App<h2>
<pre>
streamlit run PulmoVision-Pro.py
</pre>

<h2>5️⃣ Download Dataset<h2>

# Kaggle Chest X-Ray Pneumonia
Organize folder structure as shown above
<h2>🔗 Pre-trained Models <h2>

<pre>DenseNet121: models/densenet_pulmovision.h5
ResNet50: models/resnet_pulmovision.h5
Pre-trained on Kermany X-Ray Pneumonia dataset </pre>
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
