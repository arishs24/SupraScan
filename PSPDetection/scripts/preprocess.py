import os
import numpy as np
import nibabel as nib

# Function to preprocess PET scans
def preprocess_pet(file_paths, save_path='data/preprocessed_slices.npy'):
    all_slices = []
    for file_path in file_paths:
        try:
            print(f"Starting preprocessing for: {file_path}")
            img = nib.load(file_path)
            data = img.get_fdata()
            print(f"Loaded PET data with shape: {data.shape}")

            # Select the middle slices (e.g., 7 slices around the center)
            num_slices = data.shape[2]
            middle_index = num_slices // 2
            slice_range = range(middle_index - 3, middle_index + 4)

            for i in slice_range:
                slice_data = data[:, :, i]
                all_slices.append(slice_data)

            print(f"Collected {len(slice_range)} slices from {file_path}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    all_slices = np.array(all_slices)
    np.save(save_path, all_slices)
    print(f"Saved {all_slices.shape[0]} slices to {save_path}")

# Example usage
pet_files = [
    '../ds004856/sub-976/ses-wave1/pet/sub-976_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
    '../ds004856/sub-976/ses-wave2/pet/sub-976_ses-wave2_trc-18FAV45_run-1_pet.nii.gz',
    '../ds004856/sub-976/ses-wave3/pet/sub-976_ses-wave3_trc-18FAV45_run-1_pet.nii.gz',
    '../ds004856/sub-978/ses-wave1/pet/sub-978_ses-wave1_trc-18FAV45_run-1_pet.nii.gz',
    '../ds004856/sub-980/ses-wave1/pet/sub-980_ses-wave1_trc-18FAV45_run-1_pet.nii.gz'
]

preprocess_pet(pet_files)
