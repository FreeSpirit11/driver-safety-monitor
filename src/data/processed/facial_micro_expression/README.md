# Facial Expression — Processed Data

This folder contains the **cleaned, transformed, and structured data** ready for use in model training, validation, and testing.

## ⚙️ How to Generate Processed Data

1. Ensure that the raw data has been downloaded and placed correctly in:  
   src/data/raw/facial_micro_expression/

2. Run the preprocessing script:  
   python src/preprocessing_data/facial_micro_expression/pre_processing_facial_micro_expression.py

3. The script will perform the following operations:
   - Image loading and resizing
   - Normalization of pixel values
   - Emotion label generation and encoding
   - Dataset splitting into train/validation/test sets

4. The generated outputs will be automatically saved in this directory:  
   src/data/processed/facial_micro_expression/
