def plot_training_history(history):
    """Plot accuracy and loss curves"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with Seaborn"""
    cm = confusion_matrix(y_true, y_pred > 0.5)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Alert', 'Drowsy'],
                yticklabels=['Alert', 'Drowsy'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred > 0.5, target_names=['Alert', 'Drowsy']))


def visualize_predictions(model, generator, n_samples=5):
    """Show sample predictions with frames and color-coded correctness"""

    batch_index = random.randint(0, len(generator) - 1)
    X, y = generator[batch_index]

    indices = random.sample(range(len(X)), min(n_samples, len(X)))
    X_sample = X[indices]
    y_sample = y[indices]

    preds = model.predict(X_sample)

    plt.figure(figsize=(3 * n_samples, 3))
    for i in range(len(indices)):
        plt.subplot(1, n_samples, i + 1)

        img = X_sample[i][SEQUENCE_LENGTH // 2]

        if img.max() <= 1.0:
            img = (img * 255).astype('uint8')
        else:
            img = img.astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        plt.imshow(img)

        pred_label = 1 if preds[i][0] >= 0.5 else 0
        true_label = y_sample[i]
        color = 'green' if pred_label == true_label else 'red'

        plt.title(f"T:{true_label} P:{pred_label}\n{preds[i][0]:.2f}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
