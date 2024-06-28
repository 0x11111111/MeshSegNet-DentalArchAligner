import os
from teeth_segmentation import extract_teeth_from_jaw
from cbct_preprocessing import extract_cbct, convert_nii_to_tiff
from yolov10_detect_arch import preprocess_tiff_and_detect_arch
import shutil
from meshlib import mrmeshpy as mm
abs_path = os.path.abspath(os.path.dirname(__file__))
dll_dir = os.path.join(abs_path, 'dll')
os.add_dll_directory(dll_dir)
import cbct_jaw_registration


def recreate_directory(directory_path):
    # 如果目录存在，删除它及其所有内容
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    # 重新创建目录
    os.makedirs(directory_path, exist_ok=True)


def registration(upper_jaw_mesh: str, lower_jaw_mesh: str, cbct_path: str, temp_path: str):
    yolo_model_rel_path = 'models/detect_arch.pt'
    yolo_model_abs_path = os.path.abspath(yolo_model_rel_path)

    recreate_directory(temp_path)
    output_upper = os.path.join(temp_path, 'upper_teeth.ply')
    output_lower = os.path.join(temp_path, 'lower_teeth.ply')
    extract_teeth_from_jaw(upper_jaw=upper_jaw_mesh, lower_jaw=lower_jaw_mesh, output_upper_teeth=output_upper, output_lower_teeth=output_lower)

    cbct_extracted_path = extract_cbct(cbct_path, temp_path)
    tiff_folder = os.path.join(temp_path, 'tiff')
    convert_nii_to_tiff(nii_file=cbct_extracted_path, output_tiff_folder=tiff_folder)

    min_x, max_x, min_y, max_y, min_z, max_z = preprocess_tiff_and_detect_arch(tiff_folder=tiff_folder, model_file=yolo_model_abs_path, temp_folder=temp_path)
    print('min_x:', min_x)
    print('max_x:', max_x)
    print('min_y:', min_y)
    print('max_y:', max_y)
    print('min_z:', min_z)
    print('max_z:', max_z)

    cbct_registration = cbct_jaw_registration.CBCTJawRegistration(
        upper_teeth_file=output_upper,
        lower_teeth_file=output_lower,
        tiff_folder=tiff_folder,
        upper_jaw_file=upper_jaw_mesh,
        lower_x=min_x,
        upper_x=max_x,
        lower_y=min_y,
        upper_y=max_y,
        lower_z=min_z,
        upper_z=max_z,
        voxel_space=0.3,
        temp=temp_path
    )
    cbct_registration.Registration()
    upper_a, upper_b = cbct_registration.GetUpperTransformation()
    lower_a, lower_b = cbct_registration.GetLowerTransformation()

    affinex3f_upper = mm.AffineXf3f()
    upper_a_matrix = mm.Matrix3f()
    upper_a_matrix.x = mm.Vector3f(*upper_a[0])
    upper_a_matrix.y = mm.Vector3f(*upper_a[1])
    upper_a_matrix.z = mm.Vector3f(*upper_a[2])

    affinex3f_upper.A = upper_a_matrix
    affinex3f_upper.b = mm.Vector3f(*upper_b)

    affinex3f_lower = mm.AffineXf3f()

    lower_a_matrix = mm.Matrix3f()
    lower_a_matrix.x = mm.Vector3f(*lower_a[0])
    lower_a_matrix.y = mm.Vector3f(*lower_a[1])
    lower_a_matrix.z = mm.Vector3f(*lower_a[2])

    affinex3f_lower.A = lower_a_matrix
    affinex3f_lower.b = mm.Vector3f(*lower_b)

    upper_jaw = mm.loadMesh(upper_jaw_mesh)
    lower_jaw = mm.loadMesh(lower_jaw_mesh)

    upper_jaw.transform(affinex3f_upper)
    lower_jaw.transform(affinex3f_lower)

    upper_jaw_output_path = os.path.join(temp_path, 'upper_jaw_transformed.ply')
    lower_jaw_output_path = os.path.join(temp_path, 'lower_jaw_transformed.ply')

    mm.saveMesh(upper_jaw, upper_jaw_output_path)
    mm.saveMesh(lower_jaw, lower_jaw_output_path)


if __name__ == '__main__':
    data_rel_path = 'data/04/'
    data_abs_path = os.path.abspath(data_rel_path)
    print(data_abs_path)

    temp_path = os.path.join(data_abs_path, 'temp')
    upper_jaw_path = os.path.join(data_abs_path, '口扫/UpperJaw.stl')
    lower_jaw_path = os.path.join(data_abs_path, '口扫/LowerJaw.stl')
    cbct_path = os.path.join(data_abs_path, 'ct.nii.gz')
    print(temp_path)

    registration(upper_jaw_path, lower_jaw_path, cbct_path, temp_path)
