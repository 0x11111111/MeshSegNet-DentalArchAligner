import shutil
import gzip
from pyunpack import Archive
from pathlib import Path
import os
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def recreate_directory(directory_path):
    # 如果目录存在，删除它及其所有内容
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    # 重新创建目录
    os.makedirs(directory_path, exist_ok=True)


def extract_cbct(cbct_path, temp_path):
    # Convert paths to Path objects for easier manipulation
    cbct_path = Path(cbct_path)
    temp_path = Path(temp_path)

    # Check the file extension
    ext = cbct_path.suffix.lower()

    if ext == ".nii":
        # If the file is already a .nii file, return its absolute path
        return str(cbct_path.resolve())

    elif ext == ".gz" and cbct_path.suffixes[-2:] == ['.nii', '.gz']:
        # Handle .nii.gz files
        if not temp_path.exists():
            temp_path.mkdir(parents=True, exist_ok=True)

        # Extract the .nii file from .gz
        nii_file_path = temp_path / cbct_path.with_suffix('').name  # Remove .gz extension
        with gzip.open(cbct_path, 'rb') as f_in:
            with open(nii_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return str(nii_file_path.resolve())

    elif ext in [".zip", ".rar", ".7z"]:
        # If the file is a compressed format, extract it
        if not temp_path.exists():
            temp_path.mkdir(parents=True, exist_ok=True)

        # Extract using pyunpack
        Archive(str(cbct_path)).extractall(str(temp_path))

        # Look for a .nii file in the extracted contents
        nii_files = list(temp_path.rglob("*.nii"))
        if len(nii_files) == 1:
            return str(nii_files[0].resolve())
        else:
            raise FileNotFoundError("No .nii file found in the extracted contents or multiple .nii files found.")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def convert_nii_to_tiff(nii_file: str, output_tiff_folder: str) -> None:
    # Load the NIfTI file
    nii_img = nib.load(nii_file)
    nii_data = nii_img.get_fdata()

    recreate_directory(output_tiff_folder)

    voxel_spacing = nii_img.header.get_zooms()[0]
    print(f"Original voxel spacing: {voxel_spacing}")

    # Desired voxel spacing
    desired_spacing = 0.3

    # Check if resampling is needed
    if abs(voxel_spacing - desired_spacing) > 1e-4:
        # Scale factors for each dimension
        scale_factors = [voxel_spacing / desired_spacing] * 3
        print(f"Scale factors: {scale_factors}")

        # Resample the image data
        nii_data = zoom(nii_data, scale_factors, order=1)  # Linear interpolation
        print(f"Resampled data shape: {nii_data.shape}")
    else:
        print("Voxel spacing matches the desired spacing. No resampling needed.")

    # Normalize the data to the range 0-255 for 8-bit TIFF
    nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255
    nii_data = nii_data.astype(np.uint8)

    # Iterate over slices and save each slice as a separate TIFF file
    num_slices = nii_data.shape[2]
    for i in range(num_slices):
        slice_data = nii_data[:, :, i]
        rotated_slice_data = np.rot90(slice_data, k=-1)  # Rotate 90 degrees clockwise

        # Convert the numpy array to a PIL image
        converted_image = Image.fromarray(rotated_slice_data)
        tiff_filename = os.path.join(output_tiff_folder, f"slice_{i + 1:04d}.tiff")
        converted_image.save(tiff_filename)

        # print(f"Saved {tiff_filename}.")


if __name__ == '__main__':
    cbct_path = r"D:\Code\MeshSegNet\data\04\ct.nii.gz"
    temp_path = r"D:\Code\MeshSegNet\data\04\temp"
    cbct_result_path = extract_cbct(cbct_path, temp_path)
    print(f"The .nii file is located at: {cbct_result_path}")
    tiff_output_path = os.path.join(temp_path, 'tiff')
    if os.path.exists(tiff_output_path):
        # 删除目录及其内容
        shutil.rmtree(tiff_output_path)
        print(f"目录 '{tiff_output_path}' 已删除")
    else:
        print(f"目录 '{tiff_output_path}' 不存在")
    # 创建输出目录（如果不存在）
    os.makedirs(tiff_output_path, exist_ok=True)
    convert_nii_to_tiff(cbct_result_path, tiff_output_path)
