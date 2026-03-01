# Driver Attentiveness — Raw Data

This folder contains the raw, unprocessed data required for the Driver Attentiveness Detection subtask.

## 🔽 How to Get the Data

You can download the required dataset from the following official source:

### 🔹 DAD (Driver Attention Dataset)  
**Official Link:** [TUM DAD Dataset](https://www.ce.cit.tum.de/mmk/dad/)  
**Description:**  
The DAD dataset provides real driving video data with frame-wise annotations indicating whether the driver is attentive or distracted. It includes multiple camera angles, lighting conditions, and realistic scenarios for attentiveness analysis.

## 📂 Folder Usage Instructions

Download and extract the dataset into the following directory:  
`src/data/raw/driver_attentiveness/`

> ⚠️ **Note:** Do not modify the raw files manually. All cleaning, processing, and formatting should be performed using the preprocessing script:  
> `src/preprocessing_data/driver_attentiveness/pre_processing_driver_attentiveness.py`
