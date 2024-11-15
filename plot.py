import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc


def plot_logistic_regression_evaluation(logistic_model, X_test, Y_test):
    # Get predicted probabilities
    Y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ROC Curve
    plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') #Plot the diagonal line which represents a random model
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    plt.subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")


def plot_training_history(history, save_path=None):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Keras history object
        save_path: Optional path to save the plot
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Training History', fontsize=16)
    
    # Plot training & validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('AUC Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Calculate and plot learning rate if available
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    # Convert probabilities to class predictions if necessary
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save the plot
    """
    # Convert one-hot encoded labels if necessary
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
    
    # Calculate area under curve
    pr_auc = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save the plot
    """
    # Convert one-hot encoded labels if necessary
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_predictions(original_image, true_mask, predicted_mask, save_path=None):
    """
    Visualize original image, true mask, and predicted mask side by side.
    
    Args:
        original_image: Original input image
        true_mask: True segmentation mask
        predicted_mask: Predicted segmentation mask
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot true mask
    axes[1].imshow(true_mask, cmap='binary')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(predicted_mask, cmap='binary')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_metrics(y_true, y_pred):
    """
    Print detailed classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Convert probabilities to class predictions if necessary
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
