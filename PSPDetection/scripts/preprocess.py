import nibabel as nib
import numpy as np
import os

def preprocess(file_path, output_path, crop_coords=None):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"Loaded PET data with shape: {data.shape}")
        
        # Optional cropping to focus on midbrain and related regions
        if crop_coords:
            x_start, x_end, y_start, y_end, z_start, z_end = crop_coords
            data = data[x_start:x_end, y_start:y_end, z_start:z_end]
            print(f"Cropped data to shape: {data.shape}")
        
        # Select slices (middle 7 for now)
        slice_indices = np.linspace(0, data.shape[-1] - 1, 7, dtype=int)
        slices = data[:, :, slice_indices]

        # Reshape for model input
        slices = slices[..., np.newaxis]
        print(f"Final preprocessed data shape: {slices.shape}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, slices)
        print(f"Saved {len(slices)} slices to {output_path}")

    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
