# Laboratory 3 — Transforming Textual and Image Data into a Machine-Understandable Format

**Course:** Introduction to Artificial Intelligence (AI–101L)  
**Instructor:** Dr. Muhammad Awais  
**Duration:** 3 hours · **Block:** Phase A — Environment and Data · **Mapped CLO:** CLO–2  

---

## 📌 Project Overview
Machine learning algorithms cannot ingest raw text or arbitrary images directly; they require **fixed-length numeric vector representations**. This lab demonstrates two foundational transformation pipelines:

1. **Text Data Vectorization (Bag-of-Words)**: Transforming variable-length text strings into a document-term matrix.
2. **Image Data Feature Engineering**: Converting high-dimensional multichannel color images into standardized, flattened numeric feature vectors.

```
Part A (Text)  : Documents ──► Tokenization ──► Vocabulary Building ──► Document-Term Matrix (BoW)
Part B (Images): RGB Image ──► Resize (WxH) ──► Grayscale ('L' mode) ──► 1D Flattened Vector
```

---

## 🎯 Learning Objectives
- Convert text documents into numerical feature vectors using `CountVectorizer`.
- Analyze vocabulary generation, token patterns, lowercase conversion, and document-term sparsity.
- Compare unigram vs bigram vocabulary growth (`ngram_range=(1,2)`).
- Perform image grayscale conversion ($3	imes$ feature reduction factor) discarding chrominance while retaining luminance and spatial geometry.
- Flatten 2D matrices into 1D feature vectors suitable for classical ML algorithms (Logistic Regression, SVM).
- Build a reusable image preprocessing pipeline (`image_to_features`) guaranteeing constant vector output dimension across heterogeneous input image resolutions.

---

## 📂 Repository Structure
```
├── Lab_03_Text_and_Image_Features.ipynb  # Interactive Jupyter Notebook
├── image_utils.py                       # Reusable image_to_features function
├── requirements.txt                     # Dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Documentation
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Notebook
```bash
jupyter notebook Lab_03_Text_and_Image_Features.ipynb
```

### 3. Using `image_utils.py`
```python
from image_utils import image_to_features
from PIL import Image

vec = image_to_features("path/to/image.jpg", size=(64, 64))
print("Vector length:", vec.shape)  # Output: (4096,)
```
