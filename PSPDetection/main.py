import numpy as np
from scripts.preprocess import preprocess_pet
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Define the available PET files
pet_files = [
    'pet/sub-976_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
    'pet/sub-976_ses-wave1_trc-18FAV45_run-2_pet.nii.gz',
    'pet/sub-976_ses-wave1_trc-18FAV45_run-3_pet.nii.gz',
    'pet/sub-978_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
    'pet/sub-980_ses-wave1_trc-18FAV45_run-1_pet.nii.gz'
]

preprocessed_dir = 'data'
os.makedirs(preprocessed_dir, exist_ok=True)

# Preprocess each PET file
all_data = []
for i, pet_file in enumerate(pet_files):
    preprocessed_path = f'{preprocessed_dir}/preprocessed_slices_{i}.npy'
    preprocess_pet(pet_file, preprocessed_path)
    
    try:
        data = np.load(preprocessed_path)
        print(f"Loaded preprocessed data from {pet_file} with shape: {data.shape}")
        if data.shape[0] > 0:  # Only append non-empty data
            all_data.append(data)
        else:
            print(f"Warning: No data in {pet_file}")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")

# Concatenate all preprocessed data for model training
if all_data:
    x_data = np.concatenate(all_data, axis=0)
    y_data = np.random.randint(2, size=x_data.shape[0])  # Dummy labels

    print(f"Final concatenated data shape: {x_data.shape}")

    # Build the model
    def build_model(input_shape=(7, 128, 128, 1)):
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            layers.GlobalAveragePooling3D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    model = build_model()

    # Check if x_data is empty before training
    if x_data.shape[0] > 0:
        history = model.fit(x_data, y_data, epochs=10, validation_split=0.2)

        # Plot accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    else:
        print("No valid data for training.")
else:
    print("No preprocessed data available for model training.")
