# Driver Attentiveness — Processed Data

This folder contains the **cleaned, transformed, and structured data** ready for use in model training, and testing.

## ⚙️ How to Generate Processed Data

1. Ensure that the raw data has been downloaded and placed correctly in:  
   `src/data/raw/driver_attentiveness/`

2. Run the preprocessing script:  
   `python src/preprocessing_data/driver_attentiveness/pre_processing_driver_attentiveness.py`

3. The script will perform the following operations:
   - Frame extraction from videos (if applicable)
   - Head pose or gaze feature extraction
   - Image resizing and normalization
   - Attentiveness label generation and encoding
   - Dataset splitting into train/test sets

4. The generated outputs will be automatically saved in this directory:  
   `src/data/processed/driver_attentiveness/`
