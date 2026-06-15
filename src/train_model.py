import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
train_ds = tf.keras.utils.image_dataset_from_directory('data/train', image_size=(150, 150), batch_size=32)

# Build CNN - This structure identifies features in images
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=5)
model.save('model/safety_model.keras')

# Train the model and save the history (this records the learning process)
history = model.fit(train_ds, epochs=5)

# After training, print the final accuracy
final_acc = history.history['accuracy'][-1]
print(f"Final training accuracy achieved: {final_acc * 100:.2f}%")
