import tensorflow as tf
import os

# 1. Load the saved model
model = tf.keras.models.load_model('model/safety_model.keras')

# 2. Load the TEST data (make sure you have a 'data/test' folder)
test_ds = tf.keras.utils.image_dataset_from_directory(
    'data/test',
    image_size=(150, 150),
    batch_size=32
)

# 3. This one command gives you the exact accuracy on new, unseen data
loss, accuracy = model.evaluate(test_ds)
print(f"Final Model Accuracy on Test Data: {accuracy * 100:.2f}%")