# Facial Expression — Raw Data

This folder contains the raw, unprocessed data required for the Facial Expression Recognition subtask.

## 🔽 How to Get the Data

Use datasets that consist of facial images labeled with emotion or expression categories (e.g., neutral, happy, sad, angry, etc.).

- The dataset should be in the form of **single images** per instance.
- Example formats: JPEG, PNG with accompanying label files or folder-structured labels.

## 📂 Folder Usage Instructions

Download and extract the dataset into the following directory:  
`src/data/raw/facial_micro_expression/`

> ⚠️ **Note:** Do not modify the contents manually. All processing and transformations should be done using the preprocessing script:  
> `src/preprocessing_data/facial_micro_expression/pre_processing_facial_micro_expression.py`
