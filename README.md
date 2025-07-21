# EEG-BCI-transfer-learning
# Brain-Computer Interface with Transfer Learning for Motor Imagery EEG Classification

This project explores the use of transfer learning, data augmentation, and ensemble models to improve motor imagery classification in EEG-based Brain-Computer Interfaces (BCIs). It was developed as part of my MSc Robotics dissertation at the University of Sheffield.

## 🧠 Project Highlights
- **Domain**: EEG-based Brain-Computer Interfaces (BCIs)
- **Models**: SVM, LDA, Ensemble Learning
- **Techniques**: Time-Shifting, Gaussian Noise Injection, CORAL Domain Adaptation
- **Peak Accuracy**: 88.19% with ensemble model + time-shifting augmentation
- **Framework**: MIND – Measure, Interpret, Encode, Deploy

## 📁 Key Components
- `preprocessing`: ICA filtering, bandpass filtering, artifact removal
- `feature_extraction`: Common Spatial Patterns (CSP), Bandpower
- `classification`: SVM and LDA with hyperparameter tuning
- `ensemble`: Weighted voting model combining SVM (70%) and LDA (30%)
- `transfer_learning`: Domain adaptation with CORAL
- `augmentation`: Time shifting, Gaussian noise
- `evaluation`: Accuracy, confusion matrix, cross-validation

## 🔧 Technologies Used
- MATLAB
- (Optional) Python (if converting scripts later)
- EEG 10-20 channel layout
- Cross-validation, Grid Search

## 📄 Report
A detailed academic dissertation is included in the [`report/`](report/) folder.

## 👤 Author
**Animesh Sandhu**  
MSc Robotics, University of Sheffield  
LinkedIn: https://www.linkedin.com/in/animesh-sandhu-9a0586174?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BWLyZbZnDQv6jZcXJWaHhUw%3D%3D

---

