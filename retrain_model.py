import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense  # Import Dense layer
import joblib
import tensorflow as tf
from PIL import Image

# Load the existing model
print("🔄 Loading the existing model...")
model_path = "trained_cnn_model.h5"
model = load_model(model_path)

# Load the label encoder
label_encoder_path = "label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)

# Directory setup
dataset_dir = "MY_data"  # Directory for your image dataset
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# Ensure directories exist
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(f"Training or testing directories not found at {dataset_dir}")

# Loading the original dataset labels
print("📂 Loading original dataset labels...")
categories = os.listdir(train_dir)
categories.sort()  # Sorting to maintain consistency with the label encoder
print(f"Categories found: {categories}")

# Load the user feedback CSV
print("📝 Loading user feedback...")
feedback_file = "feedback_data.csv"
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
    print(f"Feedback loaded with {len(feedback_df)} entries.")
else:
    print("No feedback found.")

# Combine original dataset labels with feedback data
feedback_labels = feedback_df['correct_label'].unique() if not feedback_df.empty else []

# Combine original dataset labels with feedback labels, ensuring no duplicates
all_labels = list(set(categories) | set(feedback_labels))  
all_labels.sort()  # Sort labels to ensure consistency
print(f"Total unique labels: {len(all_labels)}")

# Encoding the labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)  # Ensure all labels (original + feedback) are covered

# Check the range of labels after encoding
print(f"Labels encoded: {label_encoder.classes_}")

# Preprocess and load images from directories
image_data = []
encoded_labels = []

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"⚠️ Error loading image {image_path}: {e}")
        return None

# Function to load images and labels from directory
def load_images_from_dir(directory, label):
    images = []
    labels = []
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        img_array = preprocess_image(image_path)
        if img_array is not None:
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load training and test images
print("⚙️ Loading training images...")
for idx, category in enumerate(categories):
    category_train_dir = os.path.join(train_dir, category)
    if os.path.isdir(category_train_dir):
        images, labels = load_images_from_dir(category_train_dir, category)
        image_data.extend(images)
        encoded_labels.extend(labels)

print(f"Loaded {len(image_data)} training images.")

# Convert labels to encoded form
encoded_labels = label_encoder.transform(encoded_labels)

# Convert image data to numpy arrays
image_data = np.array(image_data)
encoded_labels = np.array(encoded_labels)

# Update the model's output layer to match the number of classes
num_classes = len(label_encoder.classes_)
print(f"Updating model output layer for {num_classes} classes...")

# Remove the last layer of the model
model.pop()  # Remove the old output layer

# Add a new output layer with the correct number of classes and a unique name
model.add(Dense(num_classes, activation='softmax', name='output_layer'))  # Ensure unique name

# Recompile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation for better generalization
print("🖼️ Applying data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model
print("🚀 Retraining the model...")
model.fit(datagen.flow(image_data, encoded_labels, batch_size=32), epochs=5)

# Save the retrained model
model.save(model_path)
print(f"✅ Model retrained and saved as {model_path}.")

# Save the updated label encoder
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ Label encoder saved as {label_encoder_path}.")
