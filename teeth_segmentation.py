import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.svm import SVC # uncomment this line if you don't install thudersvm
# from thundersvm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
import utils
import vtk
from typing import Tuple


def process_path(path: str) -> tuple[str, str, str]:
    """
    将路径中的分隔符修改为当前操作系统的分隔符，并返回最后一级文件名和扩展名。

    参数:
    path (str): 原始路径。

    返回:
    tuple[str, str]: 最后一级文件名和扩展名。
    """
    # 获取当前操作系统的目录分隔符
    current_sep = os.path.sep

    # 将路径中的所有分隔符替换为当前操作系统的目录分隔符
    normalized_path = path.replace('/', current_sep).replace('\\', current_sep)

    path = os.path.dirname(normalized_path)
    # 获取最后一级文件名
    basename = os.path.basename(normalized_path)

    # 分割文件名和扩展名
    filename, extension = os.path.splitext(basename)

    return path, filename, extension


def get_arch_teeth(vtp_path: str) -> vtk.vtkPolyData:
    # 读取输入的VTP文件
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()

    m_teeth_poly_data = reader.GetOutput()
    num_points = m_teeth_poly_data.GetNumberOfPoints()
    num_cells = m_teeth_poly_data.GetNumberOfCells()

    celldata = m_teeth_poly_data.GetCellData()
    labels = celldata.GetArray("Label")

    # 初始化数据结构
    m_labeled_poly_data = [vtk.vtkPolyData() for _ in range(20)]
    m_labeled_points = [vtk.vtkPoints() for _ in range(20)]
    m_labeled_triangles = [vtk.vtkCellArray() for _ in range(20)]
    m_split_poly_data_flag = [0] * 20
    m_split_polydata_visit = [[-1] * num_points for _ in range(20)]

    # 根据标签类型分割polydata
    triangle = vtk.vtkTriangle()

    for i in range(labels.GetSize()):
        label = int(labels.GetTuple(i)[0])

        face_i = m_teeth_poly_data.GetCell(i)

        for j in range(3):
            point_id_j = face_i.GetPointId(j)

            if m_split_polydata_visit[label][point_id_j] == -1:
                new_point_id = m_split_poly_data_flag[label]
                m_labeled_points[label].InsertPoint(new_point_id, m_teeth_poly_data.GetPoint(point_id_j))
                triangle.GetPointIds().SetId(j, new_point_id)
                m_split_polydata_visit[label][point_id_j] = new_point_id
                m_split_poly_data_flag[label] += 1
            else:
                triangle.GetPointIds().SetId(j, m_split_polydata_visit[label][point_id_j])

        m_labeled_triangles[label].InsertNextCell(triangle)

    # 设置每个标签的点和三角形
    for i in range(20):
        m_labeled_poly_data[i].SetPoints(m_labeled_points[i])
        m_labeled_poly_data[i].SetPolys(m_labeled_triangles[i])

    # 合并所有标签的polydata
    append_filter = vtk.vtkAppendPolyData()

    for i in range(1, 20):
        append_filter.AddInputData(m_labeled_poly_data[i])

    append_filter.Update()

    output = vtk.vtkPolyData()
    output.ShallowCopy(append_filter.GetOutput())

    return output

def teeth_segmentation(model_path_name : str, mesh_path_name : str, output_teeth_path_name : str) -> None:
    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id)  # assign which gpu will be used (only linux works)

    # upsampling_method = 'SVM'
    upsampling_method = 'KNN'

    num_classes = 15
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    print('output_teeth_path_name:', output_teeth_path_name)
    output_teeth_path, save_name, _ = process_path(output_teeth_path_name)
    print(f"Processing {mesh_path_name} , save to {output_teeth_path} / {save_name}...")

    # load trained model
    checkpoint = torch.load(model_path_name, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Predicting
    model.eval()
    with torch.no_grad():

        start_time = time.time()
        # create tmp folder
        tmp_path = './.tmp/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        # print('Predicting Sample filename: {}'.format(i_sample))
        # read image and label (annotation)
        mesh = vedo.load(mesh_path_name)

        # pre-processing: downsampling
        # print('\tDownsampling...')
        target_num = 10000
        ratio = target_num / mesh.ncells  # calculate ratio
        mesh_d = mesh.clone()
        mesh_d.decimate(fraction=ratio)
        predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

        # move mesh to origin
        # print('\tPredicting...')
        points = mesh_d.points()
        mean_cell_centers = mesh_d.center_of_mass()
        points[:, 0:3] -= mean_cell_centers[0:3]

        ids = np.array(mesh_d.faces())
        cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        mesh_d.compute_normals()
        normals = mesh_d.celldata['Normals']

        # move mesh to origin
        barycenters = mesh_d.cell_centers  # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))

        # computing A_S and A_L
        A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
        D = distance_matrix(X[:, 9:12], X[:, 9:12])
        A_S[D < 0.1] = 1.0
        A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        A_L[D < 0.2] = 1.0
        A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        # numpy -> torch.tensor
        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])
        X = torch.from_numpy(X).to(device, dtype=torch.float)
        A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
        A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
        A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
        A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

        tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
        patch_prob_output = tensor_prob_output.cpu().numpy()

        for i_label in range(num_classes):
            predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1) == i_label] = i_label

        # output downsampled predicted labels
        mesh2 = mesh_d.clone()
        mesh2.celldata['Label'] = predicted_labels_d
        # vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

        # refinement
        # print('\tRefining by pygco...')
        round_factor = 100
        patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, num_classes)

        # parawise
        pairwise = (1 - np.eye(num_classes, dtype=np.int32))

        # edges
        normals = mesh_d.celldata['Normals'].copy()  # need to copy, they use the same memory address
        barycenters = mesh_d.cell_centers  # don't need to copy
        cell_ids = np.asarray(mesh_d.faces())

        lambda_c = 30
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei == 2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]) / np.linalg.norm(
                        normals[i_node, 0:3]) / np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi / 2.0:
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -np.log10(theta / np.pi) * phi]).reshape(1, 3)),
                            axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -beta * np.log10(theta / np.pi) * phi]).reshape(1, 3)),
                            axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c * round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # output refined result
        mesh3 = mesh_d.clone()
        mesh3.celldata['Label'] = refine_labels
        # vedo.write(mesh3, os.path.join(output_path, '{}_d_predicted_refined.vtp'.format(i_sample[:-4])))

        mesh_centered = mesh_d.clone()
        points_centered = mesh_d.points()
        points_centered[:, 0:3] += mean_cell_centers[0:3]
        mesh_centered.points(points_centered)
        mesh_centered.celldata['Label'] = refine_labels
        vtp_file_path = os.path.join(output_teeth_path, save_name+'.vtp')
        teeth_file_path = os.path.join(output_teeth_path, save_name+'.ply')
        print(teeth_file_path)
        print(vtp_file_path)
        vedo.write(mesh_centered, vtp_file_path)

        output_polydata = get_arch_teeth(vtp_file_path)

        print('\tSaving output to {}'.format(teeth_file_path))
        # 保存输出结果
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(teeth_file_path)
        writer.SetInputData(output_polydata)
        writer.Write()

        # upsampling
        # print('\tUpsampling...')
        if mesh.ncells > 50000:
            target_num = 50000  # set max number of cells
            ratio = target_num / mesh.ncells  # calculate ratio
            mesh.decimate(fraction=ratio)
            # print('Original contains too many cells, simpify to {} cells'.format(mesh.ncells))

        # get fine_cells
        barycenters = mesh3.cell_centers  # don't need to copy
        fine_barycenters = mesh.cell_centers  # don't need to copy

        if upsampling_method == 'SVM':
            clf = SVC(kernel='rbf', gamma='auto')
            # train SVM
            clf.fit(barycenters, np.ravel(refine_labels))
            fine_labels = clf.predict(fine_barycenters)
            fine_labels = fine_labels.reshape(-1, 1)
        elif upsampling_method == 'KNN':
            neigh = KNeighborsClassifier(n_neighbors=3)
            # train KNN
            neigh.fit(barycenters, np.ravel(refine_labels))
            fine_labels = neigh.predict(fine_barycenters)
            fine_labels = fine_labels.reshape(-1, 1)

        mesh.celldata['Label'] = fine_labels
        # vedo.write(mesh, os.path.join(output_path, '{}_predicted_refined.vtp'.format(i_sample[:-4])))

        # remove tmp folder
        shutil.rmtree(tmp_path)

        end_time = time.time()
        # print('Sample filename: {} completed'.format(i_sample))
        # print('\tcomputing time: {0:.2f} sec'.format(end_time - start_time))


def extract_teeth_from_jaw(upper_jaw: str, lower_jaw: str, output_upper_teeth: str, output_lower_teeth: str):
    model_rel_path = './models'
    model_abs_path = os.path.abspath(model_rel_path)

    upper_teeth_model = os.path.join(model_abs_path, 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip')
    lower_teeth_model = os.path.join(model_abs_path, 'MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip')

    teeth_segmentation(upper_teeth_model, upper_jaw, output_upper_teeth)
    teeth_segmentation(lower_teeth_model, lower_jaw, output_lower_teeth)
