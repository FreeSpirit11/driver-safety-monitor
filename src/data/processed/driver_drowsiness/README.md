# Driver Drowsiness — Processed Data

This folder contains the **cleaned, transformed, and structured data** ready for use in model training, validation, and testing.

## ⚙️ How to Generate Processed Data

1. Ensure that the raw data has been downloaded and placed correctly in:
   src/data/raw/driver_drowsiness/

2. Run the preprocessing script:
   python src/preprocessing_data/driver_drowsiness/pre_processing_driver_drowsiness.py

3. The script will perform the following operations:
   - Frame extraction from videos (if needed)
   - Image resizing and normalization
   - Label generation and encoding
   - Dataset splitting into train/validation/test sets

4. The generated outputs will be automatically saved in this directory:
   src/data/processed/driver_drowsiness/
