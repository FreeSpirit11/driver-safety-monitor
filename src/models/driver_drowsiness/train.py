import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
PROCESSED_YAWDD = 'src/data/processed/driver_drowsiness/YawDD_sequences'
PROCESSED_NTHU = 'src/data/processed/driver_drowsiness/NTHU_sequences'
TEMP_MODEL_PATH = 'models/mobilenetv2_yawdd.h5'
FINETUNE_MODEL_PATH = 'models/mobilenetv2_nthu_finetuned.h5'

# Training params
IMG_SHAPE = (224, 224, 3)    
TIME_STEPS = 10            
BATCH_SIZE = 16
LR = 1e-4
EPOCHS_STAGE1 = 22
EPOCHS_STAGE2 = 12

def build_model(input_shape):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    model = Sequential([
        TimeDistributed(base),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_sequences_and_labels(seq_dir, time_steps):
    import glob
    paths = glob.glob(os.path.join(seq_dir, '*.npy'))
    X, y = [], []
    for p in paths:
        arr = np.load(p)
        if arr.shape[0] != time_steps:
            continue
        X.append(arr)
        label = int(os.path.basename(p).split('_')[1])
        y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

def train_on_yawdd():
    X, y = load_sequences_and_labels(PROCESSED_YAWDD, TIME_STEPS)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((TIME_STEPS,)+IMG_SHAPE)
    ckpt = ModelCheckpoint(TEMP_MODEL_PATH, save_best_only=True)
    stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS_STAGE1, batch_size=BATCH_SIZE,
              callbacks=[ckpt, stop])
    return model

def finetune_on_nthu():
    # reload stage1 model
    model = load_model(TEMP_MODEL_PATH)
    # unfreeze some layers
    for layer in model.layers[0].layer.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=Adam(LR/10), loss='binary_crossentropy', metrics=['accuracy'])

    X, y = load_sequences_and_labels(PROCESSED_NTHU, TIME_STEPS)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ckpt = ModelCheckpoint(FINETUNE_MODEL_PATH, save_best_only=True)
    stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS_STAGE2, batch_size=BATCH_SIZE,
              callbacks=[ckpt, stop])
    return model

def main():
    os.makedirs('models', exist_ok=True)
    print("Stage 1: Training on YawDD dataset")
    train_on_yawdd()
    print("✅ Completed YawDD training")

    print("\nStage 2: Fine-tuning on NTHU dataset")
    finetune_on_nthu()
    print("✅ Completed NTHU fine-tuning — model saved to", FINETUNE_MODEL_PATH)

if __name__ == "__main__":
    main()
