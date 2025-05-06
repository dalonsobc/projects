### Mass Spectra Comparison with Siamese Neural Network

This project implements a Siamese neural network combining 1D-CNN and LSTM layers to perform binary comparison of compound mass spectra—distinguishing whether two spectra belong to the same or different chemical compounds. The project focuses on forensic applications, such as identifying drug family relationships.


## Overview

*Goal*: Learn deep embeddings of 1D mass-spectrometry data for compound comparison using a Siamese architecture.

*Approach*:
- Data preprocessing: normalization, rational transformation, m/z binning
- Pair generation: compound-level split into same/different spectrum pairs
- Model: Siamese network with CNN + LSTM branches
- Loss: Contrastive loss
- Threshold calibration: ROC, AUC, F1-score
- Evaluation: Visualizations (t-SNE), confusion matrices (classifier-based)


## Highlights

- Achieved 99% accuracy on unseen compound pairs using a small NN distance-based classifier.
- Embeddings clearly cluster compound families (validated via t-SNE & Random Forest confusion matrix).
- Designed for generalization to real-world forensic applications.


