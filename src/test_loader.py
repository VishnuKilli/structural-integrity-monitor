import tensorflow as tf
import os

print("DEBUG: Checking folder structure...")
print(f"Current Directory: {os.getcwd()}")
print(f"Exists 'data/train': {os.path.exists('data/train')}")
if os.path.exists('data/train'):
    print(f"Contents of 'data/train': {os.listdir('data/train')}")

print("\nDEBUG: Attempting to load dataset...")
dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(150, 150),
    batch_size=32
)
print("DEBUG: Loading successful!")