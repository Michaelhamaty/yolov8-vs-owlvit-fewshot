# Sample Efficiency of YOLOv8 vs. Zero-Shot OWL-ViT

**Author:** Michael Hamaty  

## 📄 [Read the Full Research Report (PDF)](./Michael_Hamaty-Final_Report_CS-171.pdf)

### Project Overview

Object detection models typically require massive datasets to perform well. This project investigates a critical question for resource-constrained applications: **What is the "price" of matching a massive Foundation Model using a tiny, fine-tuned CNN?**

This repository contains the experimental code used to benchmark the sample efficiency of **YOLOv8 Nano** (fine-tuned on $k$-shot subsets) against two baselines:
1.  **OWL-ViT (Zero-Shot):** A Vision Transformer that uses open-vocabulary text prompts.
2.  **YOLOv8 (COCO Pretrained):** A standard off-the-shelf object detector.

The study focuses on a specific domain: **Kitchen Utensil Detection** (10 classes), comparing performance across dataset sizes of $k=1, 4, 16, \text{and } 32$ images per class.

These notebooks were designed to run in **Google Colab** with T4 GPU acceleration.


### Repository Structure

* `Michael_Hamaty-Final_Report_CS-171.pdf`
* `01_YOLOv8_FewShot_FineTuning.ipynb`: The primary pipeline. Handles data preprocessing, stratified $k$-shot sampling, and training the YOLOv8n model on restricted subsets.
* `02_ZeroShot_and_Baseline_Evaluation.ipynb`: The benchmark suite. Evaluates the zero-shot capabilities of OWL-ViT (using text prompts like "a photo of a spatula") and the transfer learning performance of standard COCO weights.

**Key Libraries:**
* `ultralytics` (YOLOv8)
* `transformers` (Hugging Face OWL-ViT)
* `torch`, `pandas`, `scikit-learn`

### Methodology Highlights

The code implements a rigorous **stratified sampling strategy**. Unlike standard train/test splits, this project creates custom subsets to strictly control the number of examples the model sees (10, 40, 160, and 320 total images).

For the detailed breakdown of how the **32-shot YOLO model** managed to tie with the zero-shot Transformer, please refer to the [Final Report](./Michael_Hamaty-Final_Report_CS-171.pdf).
