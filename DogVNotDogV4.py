import os
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime

# Set up logging
log_dir = r"C:/Users/avira/PycharmProjects/myFirstCNN/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Script started.")

# Define dataset directory
dataset_dir = 'C:/Users/avira/Downloads/Dog vs Not-Dog/new for modell'

# Create an instance of ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=20,
    # zoom_range=0.2,
    # shear_range=0.2,
    # horizontal_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # brightness_range=(0.8, 1.2),
    # fill_mode='nearest'
)

# Generate batches of normalized data for training
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    classes=['other', 'dog']
)

# Generate batches of normalized data for testing
test_generator = datagen.flow_from_directory(
    "C:/Users/avira/Downloads/Dog vs Not-Dog/test",
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    classes=['other', 'dog']
)

# Try loading the entire model
# try:
#     model = load_model(r'the one x/final_dogvsnot.model.h5')
#     logging.info("Model loaded successfully.")
# except Exception as e:
#     logging.warning(f"Error loading model, defining architecture and loading weights: {e}")

    # Define your model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))

model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
# model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))

model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
# model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D())
model.add(Dropout(rate=0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

    # Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

logging.info("Model architecture defined and compiled.")

# Print a summary of the model to verify
model.summary(print_fn=logging.info)

# Ensure the directory for saving the model exists
checkpoint_dir = r"C:/Users/avira/PycharmProjects/myFirstCNN/the one x"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, "dogvsother.h5")

# Set up the checkpoint callback to save the model with the highest validation accuracy and lowest validation loss
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model
logging.info("Training started.")
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator,
    callbacks=[checkpoint]
)
logging.info("Training completed.")

# Save training history to logs
for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
    history.history['loss'],
    history.history['accuracy'],
    history.history['val_loss'],
    history.history['val_accuracy'],
)):
    logging.info(f"Epoch {epoch + 1}: Loss={loss}, Accuracy={acc}, Val_Loss={val_loss}, Val_Accuracy={val_acc}")

# Evaluate the model on the test set
logging.info("Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_generator)
logging.info(f"Test accuracy: {test_acc}")

# Save the final model as .h5
final_model_path = os.path.join(checkpoint_dir, "99 percent baby dogVSother.h5")
model.save(final_model_path)
logging.info(f"Final model saved to {final_model_path}")

print(f'Test accuracy: {test_acc}')
