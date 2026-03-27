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

1. **Table Detection (OpenCV)**
2. **OCR (PaddleOCR)**
3. **Image Detection (Mask R-CNN)**
4. **Text Structuring**
5. **JSON Construction**

---

## 🔄 Processing Pipeline

![Pipeline](image/framework.png)

The workflow is:

1. Input document (PDF/Image)
2. Detect tables using OpenCV
3. Extract text using OCR
4. Detect images using Mask R-CNN
5. Organize extracted content
6. Generate structured JSON output

---

## 📁 Project Structure
