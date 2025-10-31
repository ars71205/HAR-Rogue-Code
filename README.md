# HAR-Rogue-Code

# Human Activity Recognition (HAR) System using LSTM

This project is a **Human Activity Recognition (HAR) model** built using the **UCI HAR Dataset**.  
It classifies **six human fitness activities** based on time-series smartphone sensor data collected from 30 individuals.  
The aim of this project is to understand and implement a **multi-class classification** problem using Deep Learning (LSTM).

---

## 🚀 Project Overview

The model uses **9-channel inertial sensor data** (accelerometer & gyroscope) to recognize the following activities:

| Label | Activity |
|--------|-------------|
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

---

## 🧠 What This Project Demonstrates

- Handling & preprocessing time-series data
- Multi-class classification using Deep Learning (LSTM)
- Evaluating classification performance using ML metrics
- Predicting activity for a single input sample

---

## 📂 Dataset

This project uses the **UCI HAR Dataset**, which includes:

- Sensor data from smartphones worn on the waist
- Data from **30 participants**
- **9 inertial signal features** per time step
- Pre-split into train & test sets

Dataset Folder Structure Required:

