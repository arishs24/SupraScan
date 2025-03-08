import numpy as np
from scripts.preprocess import preprocess
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title='PSP Detection Prototype', layout='wide')

# Display page title and description
st.title('Early Detection of Progressive Supranuclear Palsy (PSP)')
st.write('This prototype uses PET scan data to detect early signs of PSP using a 3D Convolutional Neural Network.')

# Add a sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a section:', ['Overview', 'About PSP', 'How It Works', 'Model Training', 'Results'])

# Define PET files for preprocessing
pet_files = [
    'pet/sub-976_ses-wave2_trc-18FAV45_run-1_pet.nii.gz',
    'pet/sub-976_ses-wave3_trc-18FAV45_run-1_pet.nii.gz',
    'pet/sub-978_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
    'pet/sub-980_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
]

# Directory to save preprocessed data
os.makedirs('data', exist_ok=True)

if page == 'Overview':
    st.header('Overview')
    st.write('This application aims to provide a prototype for the early detection of PSP using PET scan data. The model leverages deep learning techniques, particularly a 3D CNN, to analyze medical imaging data.')
    st.image('brainimage.png', caption='Sample Brain PET Scans: Comparing Advanced PSP, Early PSP, and Parkinson’s Disease', use_column_width=True)

elif page == 'About PSP':
    st.header('What is Progressive Supranuclear Palsy (PSP)?')
    st.write('''
    PSP is a rare brain disorder that affects movement, balance, and eye movements. Early detection is critical for managing symptoms and improving quality of life.
    Common symptoms include stiffness, difficulty walking, and changes in behavior.
    ''')

elif page == 'How It Works':
    st.header('How It Works')
    st.write('''
    1. **Data Preprocessing**: PET scan files are preprocessed into uniform slices.
    2. **Model Architecture**: A 3D Convolutional Neural Network is used.
    3. **Training**: The model is trained on labeled data to detect PSP.
    4. **Results Visualization**: Training and validation accuracy are plotted.
    5. **Prediction Tool**: Users can input sample data to predict the likelihood of PSP.
    ''')

elif page == 'Model Training':
    st.header('Model Training')

    if st.button('Train Model'):
        # Preprocess PET files
        for i, pet_file in enumerate(pet_files):
            preprocessed_path = f'data/preprocessed_slices_{i}.npy'
            preprocess(pet_file, preprocessed_path)

        # Load preprocessed data
        data_list = []
        target_shape = (128, 128, 7, 1)  # Desired uniform shape

        for i in range(len(pet_files)):
            preprocessed_path = f'data/preprocessed_slices_{i}.npy'
            try:
                data = np.load(preprocessed_path)
                st.write(f"Loaded preprocessed data from {preprocessed_path} with shape: {data.shape}")

                # Check and fix dimensions if needed
                if data.shape != target_shape:
                    data = np.reshape(data, target_shape)
                    st.write(f"Reshaped data to {target_shape}")

                data_list.append(data)
            except Exception as e:
                st.write(f"Error loading preprocessed data from {preprocessed_path}: {e}")

        # Concatenate all preprocessed data
        x_data = np.concatenate(data_list, axis=0)
        st.write(f"Final x_data shape: {x_data.shape}")

        # Generate labels for PSP detection (1 = PSP, 0 = Non-PSP)
        num_samples = x_data.shape[0]
        y_data = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2))
        st.write(f"Generated labels with shape: {y_data.shape}")

        # Build a simple model for PSP detection
        def build_model(input_shape=(7, 128, 128, 1)):
            model = models.Sequential([
                layers.InputLayer(input_shape=input_shape),
                layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
                layers.MaxPooling3D((2, 1, 1)),
                layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
                layers.MaxPooling3D((2, 1, 1)),
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

        # Build and train the model
        model = build_model(input_shape=(7, 128, 128, 1))
        history = model.fit(x_data, y_data, epochs=10, validation_split=0.2)

        st.write('Model training completed.')

        # Save the training history
        np.save('data/training_history.npy', history.history)

elif page == 'Results':
    st.header('Results')
    # Load training history if available
    try:
        history_data = np.load('data/training_history.npy', allow_pickle=True).item()
        fig, ax = plt.subplots()
        ax.plot(history_data['accuracy'], label='Train Accuracy')
        ax.plot(history_data['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.write(f'Error loading training history: {e}')
        st.write('Please go to the Model Training section and train the model first.')
