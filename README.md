# ConsBASE: Construction Document Analysis Framework

ConsBASE is an end-to-end document analysis system designed to convert scanned construction documents into structured JSON outputs.  
The framework integrates OCR, table detection, image detection, and text structuring into a unified ETL pipeline.

---

## 📌 Overview

ConsBASE processes construction documents (e.g., design reports, budget tables) and extracts structured information such as:

- Tables  
- Figures (images)  
- Text paragraphs  
- Table of contents (index)  

---

## 🏗️ System Architecture

![System Architecture](image/flowchart.png)

The system follows a modular pipeline:

1. Table Detection (OpenCV)  
2. OCR (PaddleOCR)  
3. Image Detection (Mask R-CNN)  
4. Text Structuring  
5. JSON Construction  

---

## 🔄 Processing Pipeline

![Pipeline](image/Framework.png)

---

## 📁 Project Structure

cals/
├── environment/
├── code/
├── data/
└── README.md

---

## 🚀 Installation

git clone https://github.com/SeoDSeok/cals.git  
cd cals  

cd environment  
docker build -t consbase .  

---

## 📥 Download Weights

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1g8KdJ9PDYQJOzxc2HAJa-SoxwI4RHukB/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g8KdJ9PDYQJOzxc2HAJa-SoxwI4RHukB" -O mask_custom.h5  

mv mask_custom.h5 ../code/

---

## ▶️ Run

cd ../code  

docker run -it --rm -v $(pwd):/workspace -v $(pwd):/data -w /workspace consbase bash ./run  

---

## ⚙️ Mode

Edit run file:

Table:
python -u main.py table.png 0 <API_KEY> <OCR_URL>

Document:
python -u main.py document.png 1 <API_KEY> <OCR_URL>

---

## 📄 Output

/results/

---

## ⚠️ Notes

GPU optional  
Runs on CPU if GPU unavailable  
