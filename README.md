# 🧠 SupraScan: ML-Powered Early PSP Detection  

## 📌 Overview  
**SupraScan** is an innovative ML-driven solution for the **early detection of Progressive Supranuclear Palsy (PSP)** using **PET and MRI scans**. The model is built using a **3D Convolutional Neural Network (CNN)** to detect early biomarkers such as **amyloid-beta buildup and tau protein aggregation**, allowing for early intervention and improved patient outcomes.  

## Link To Research Abstract Presented @ CSHL
https://meetings.cshl.edu/posters/galaxy25/gbcc25_AbstractBook.pdf#page117

### 🔹 Key Features:  
- **Deep Learning Model** trained on PSP and non-PSP patient data.  
- **Streamlit-based UI** for easy interaction.  
- **Real-time PET/MRI scan analysis** to detect early signs of PSP.  
- **Interactive 3D Brain Model** for visualization of affected regions.  
- **Seamless integration potential** with hospital systems like **EPIC & Cerner**.  

---

## 🛠️ Features & Functionality  
✅ **Medical Image Preprocessing**: Converts PET/MRI scans into structured inputs for analysis.  
✅ **3D CNN for Feature Extraction**: Identifies early PSP indicators from medical scans.  
✅ **Live Model Training & Prediction**: Users can train and test the model through the UI.  
✅ **Performance Metrics Visualization**: Training accuracy, validation accuracy, and prediction confidence.  
✅ **3D Brain Model Interaction**: Allows visualization of affected brain regions.  
✅ **API & EHR Integration**: Designed for seamless integration with clinical software.  

---

## 🚀 Getting Started  

### 1️⃣ Installation  
#### Clone the Repository  
```bash
git clone https://github.com/arishs24/SupraScan.git
cd SupraScan
```

## Install Dependencies
pip install -r requirements.txt

streamlit run PSPDetection/main.py

