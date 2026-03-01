# Driver Drowsiness — Raw Data

This folder contains the **raw, unprocessed data** required for the Driver Drowsiness Detection subtask.

## 🔽 How to Get the Data

You can download the required datasets from the following official sources:

### 🔹 YawDD Dataset (Yawning Detection Dataset)
- **Official Link:** [IEEE Dataport – YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset)
- **Kaggle Mirror:** [Kaggle – YawDD Dataset](https://www.kaggle.com/datasets/enider/yawdd-dataset)
- **Description:**  
  Contains dashcam videos annotated for yawning behavior. Useful for detecting early signs of fatigue based on mouth movements.

### 🔹 NTHU Drowsiness Detection Dataset
- **Official Link:** [NTHU Dataset Page](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
- **Kaggle Mirror:** [Kaggle – NTHU DDD Dataset](https://www.kaggle.com/datasets/banudeep/nthuddd2)
- **Description:**  
  Provides video sequences with simulated drowsiness under diverse conditions (nighttime, sunglasses, glasses, etc.). Ideal for fine-tuning and evaluating drowsiness models.

---

## 📂 Folder Usage Instructions

1. Download and extract the dataset(s) into this directory:  
   `src/data/raw/driver_drowsiness/`

2. **Do not modify** the contents manually. Any cleaning or transformations should be done using the corresponding preprocessing script: src/preprocessing_data/driver_drowsiness/pre_processing_driver_drowsiness.py
