# Quantitative Analysis of Upper Limb Clonic Movements Using Optical Flow

## Overview
This repository provides Python code for quantitative analysis of upper-limb clonic movements
in bilateral tonic–clonic seizures (BTCS) and psychogenic nonepileptic seizures (PNES)
using optical flow analysis and principal component analysis (PCA).

The code was developed to extract interpretable motion-based metrics from routine
video-EEG monitoring (VEEG) recordings and corresponds to the analyses reported in our manuscript:

> *Quantitative Analysis of Upper Limb Clonic Movements in Tonic–Clonic Seizures and  
> Psychogenic Nonepileptic Seizures Using Optical Flow*

---

## Key Concepts
- Dense optical flow (Farnebäck method) for motion quantification
- Body-centered coordinate projection using shoulder landmarks
- PCA-based dimensionality reduction of two-dimensional motion
- Quantification of temporal changes in clonic frequency and amplitude

---

## Analysis Pipeline
1. **Manual ROI selection**  
   A trapezoidal region of interest (ROI) covering one upper limb is manually defined
   and fixed throughout the analysis window.

2. **Optical flow estimation**  
   Dense optical flow vectors are computed within the ROI using the Farnebäck method.

3. **Body-axis projection**  
   Optical flow vectors are projected onto a body-centered coordinate system
   defined by shoulder landmarks.

4. **Principal component analysis (PCA)**  
   Short-time PCA is applied to velocity vectors to extract the dominant movement
   direction as a one-dimensional waveform (PC1).

5. **Metric extraction**  
   Frequency- and amplitude-related metrics are computed from the PC1 waveform.

---

## Metrics
The following metrics are implemented and used in the manuscript.

- **Kendall’s τ**  
  Quantifies the monotonic temporal change in clonic frequency.
  Positive values indicate progressive slowing of clonic movements.

- **Amplitude Decay Slope (ADS)**  
  Log-linear slope of the smoothed |PC1| waveform,
  representing temporal changes in movement amplitude.

- **Area Under the Curve (AUC)**  
  Time integral of |PC1|, reflecting overall movement intensity
  during the analysis window.

These metrics were selected to capture **distinct but complementary aspects**
of clonic motor behavior.

---

## Citation
If you use this code, please cite the corresponding manuscript.
Saito S, Kuramochi I, Taniguchi G, et al.
Quantitative Analysis of Upper Limb Clonic Movements in Tonic–Clonic Seizures
and Psychogenic Nonepileptic Seizures Using Optical Flow.

---

## Author
Satoshi Saito, MD
Department of Epileptology
National Center Hospital, National Center of Neurology and Psychiatry, Japan
