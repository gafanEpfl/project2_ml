import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os
import sys
from tensorflow.keras import layers, Model
import pathlib
from plot import *

# Constants (keep your existing constants)
TRAINING_SIZE = 100
BATCH_SIZE = 16
NUM_CHANNELS = 3
NUM_LABELS = 2
NUM_EPOCHS = 100
IMG_PATCH_SIZE = 16
PIXEL_DEPTH = 255.0

# Helper functions
def img_crop(im, w, h):
    """Helper function for cropping an image."""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im_patch = im[j:j + w, i:i + h]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images, patch_size):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    
    return np.asarray(data)

def value_to_class_cnn(v):
    """Assign a label to a patch."""
    foreground_threshold = 0.25
    df = np.sum(v)
    return [0, 1] if df > foreground_threshold else [1, 0]

def extract_labels_cnn(filename, num_images, patch_size):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) 
                      for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class_cnn(np.mean(data[i])) for i in range(len(data))])
    
    return labels.astype(np.float32)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
                    / predictions.shape[0])

def label_to_img_cnn(imgwidth, imgheight, w, h, labels):
    """Convert array of labels to an image."""
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            l = 0 if labels[idx][0] > 0.5 else 1
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + " " + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + " " + str(max_predictions))


class RoadSegmentationModel(tf.keras.Model):
    def __init__(self):
        super(RoadSegmentationModel, self).__init__()
        
        # Define layers
        self.conv1 = layers.Conv2D(32, 5, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv2 = layers.Conv2D(64, 5, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.ReLU()
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(NUM_LABELS)
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        if training:
            x = self.dropout(x)
        return self.fc2(x)

def train_model():
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Prepare data using your helper functions
    data_dir = "training/"
    train_data = extract_data(data_dir + "images/", TRAINING_SIZE, IMG_PATCH_SIZE)
    train_labels = extract_labels_cnn(data_dir + "groundtruth/", TRAINING_SIZE, IMG_PATCH_SIZE)
    
    # Balance dataset
    print("Balancing training data...")
    c0 = np.sum(train_labels[:, 0] == 1)
    c1 = np.sum(train_labels[:, 1] == 1)
    print(f"Number of data points per class: c0 = {c0}, c1 = {c1}")
    
    min_c = min(c0, c1)
    idx0 = [i for i, label in enumerate(train_labels) if label[0] == 1]
    idx1 = [i for i, label in enumerate(train_labels) if label[1] == 1]
    new_indices = idx0[:min_c] + idx1[:min_c]
    
    train_data = train_data[new_indices]
    train_labels = train_labels[new_indices]
    
    # Manually split the data into training and validation sets
    val_split = 0.2
    num_samples = len(train_data)
    num_val_samples = int(val_split * num_samples)
    
    # Shuffle the indices
    indices = np.random.permutation(num_samples)
    train_idx, val_idx = indices[num_val_samples:], indices[:num_val_samples]

    # Augment the data
    # Think of the ways of doing it
    
    # Split the data
    x_train, x_val = train_data[train_idx], train_data[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]
    
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create and compile model
    model = RoadSegmentationModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Callbacks with fixed checkpoint filepath
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_{epoch:02d}.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model with separate validation dataset
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(plots_dir, 'training_history.png'))
    
    # Get predictions on validation set
    val_predictions = model.predict(val_dataset)
    
    # Get validation labels from dataset
    y_val = np.concatenate([y for x, y in val_dataset], axis=0)
    
    # Plot evaluation metrics
    plot_confusion_matrix(y_val, val_predictions, 
                         save_path=os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_precision_recall_curve(y_val, val_predictions, 
                              save_path=os.path.join(plots_dir, 'pr_curve.png'))
    plot_roc_curve(y_val, val_predictions, 
                   save_path=os.path.join(plots_dir, 'roc_curve.png'))
    
    # Print classification metrics
    print_classification_metrics(y_val, val_predictions)
    
    return model, history

def predict_on_test_set(model, test_dir):
    """Generate predictions for test images."""
    test_data = extract_data(test_dir, TRAINING_SIZE, IMG_PATCH_SIZE)
    predictions = model.predict(test_data)
    predictions = tf.nn.softmax(predictions)
    return predictions

if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    # Make predictions
    prediction_dir = "predictions_training/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
