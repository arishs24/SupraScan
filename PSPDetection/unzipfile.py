import nibabel as nib

# Path to the file
file_path = r'../ds004856/sub-978/ses-wave1/pet/sub-978_ses-wave1_trc-18FAV45_run-1_pet.nii'

try:
    img = nib.load(file_path)
    data = img.get_fdata()
    print(f"Successfully loaded file with shape: {data.shape}")
except Exception as e:
    print(f"Failed to load file: {e}")
