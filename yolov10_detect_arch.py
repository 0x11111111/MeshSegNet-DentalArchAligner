import ctypes
import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLOv10
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import glob
from typing import List, Tuple, Dict, Optional


def apply_adaptive_threshold_and_filter(input_dir, output_dir):
    # 检查目录是否存在
    if os.path.exists(output_dir):
        # 删除目录及其内容
        shutil.rmtree(output_dir)
        # print(f"目录 '{output_dir}' 已删除")
    else:
        # print(f"目录 '{output_dir}' 不存在")
        pass
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if os.path.join(input_dir, f).endswith('.tiff')]

    for img_name in image_files:
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name.replace('.tiff', '.jpg'))

        # 读取图像
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
        if image is None:
            print(f"无法读取图像: {input_path}")
            continue

        # 应用自适应阈值处理
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 17, 2
        )

        # 定义形态学操作的内核
        kernel = np.ones((3, 3), np.uint8)

        # 进行形态学开运算去除噪点
        morph_open = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        # 进行形态学闭运算填充线段间的空隙
        morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

        # 连通组件分析去除小的联通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_close, connectivity=8)
        min_area = 200  # 设定最小区域面积阈值

        filtered_image = np.zeros_like(morph_close)

        for i in range(1, num_labels):  # 从1开始，跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_image[labels == i] = 255

        # 保存处理后的图像
        cv2.imwrite(output_path, filtered_image)
        # print(f"已处理并保存图像: {output_path}")


def find_valid_indices(positive_indices: Dict[int, float], threshold: float, length: int = 4) -> Tuple[List[int], Optional[int], Optional[int]]:
    """
    找到包含指定长度的连续递增和递减数值的子序列，并且这些数值对应的值都大于阈值。

    参数:
    - positive_indices: 包含索引和对应值的字典
    - threshold: 阈值，子序列中所有值必须大于该阈值
    - length: 子序列的长度，默认为4。如果长度小于2，则默认为1

    返回:
    - complete_subsequence: 完整的子序列列表
    - start_val: 子序列的最小索引值
    - end_val: 子序列的最大索引值
    """
    if length < 2:
        length = 1

    # 将 map 对象转换为列表，并按照索引排序
    sorted_indices = sorted(positive_indices.items())
    indices = [k for k, v in sorted_indices]
    values = {k: v for k, v in sorted_indices}

    n = len(indices)
    if n < length:
        return [], None, None  # 如果列表长度小于指定长度，无法找到指定长度的子序列

    # 找到第一个包含指定长度的连续递增数值的子序列
    start = None
    for i in range(n - length + 1):
        if all(indices[i + j] == indices[i] + j and values[indices[i + j]] > threshold for j in range(length)):
            start = i
            break

    # 找到第一个包含指定长度的连续递减数值的子序列
    end = None
    for i in range(n - 1, length - 2, -1):
        if all(indices[i - j] == indices[i] - j and values[indices[i - j]] > threshold for j in range(length)):
            end = i
            break

    if start is not None and end is not None and start < end:
        complete_subsequence = indices[start:end + 1]
        return complete_subsequence, min(complete_subsequence), max(complete_subsequence)
    else:
        return [], None, None


def plot_predictions(img, preds, output_path):
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        label = f"{pred['label']} {pred['confidence']:.2f} {x1} {y1} {x2} {y2}"

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 绘制标签
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存绘制好预测结果的图像
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def extract_number_from_filename(filename):
    # 假设文件名格式为 'prefix_0000.extension'
    # 提取文件名中四位前补0的数字部分
    return int(filename.split('_')[-1][:4])


def predict_folder(input_folder: str, model_file: str, batch_size: int = 16) -> Tuple[Dict, Dict, Dict]:
    """
    批量预测函数，从指定文件夹中读取图像文件，进行预测，并返回正样本的索引。

    参数:
    - input_folder: 输入图像文件夹路径
    - model_file: YOLO 模型文件路径
    - batch_size: 批处理大小

    返回:
    - positive_indices: 包含正样本索引的列表
    """
    # 获取输入文件夹中的所有 TIFF 和 JPEG 图像文件路径
    image_paths = glob.glob(os.path.join(input_folder, '*.tiff')) + glob.glob(os.path.join(input_folder, '*.jpg'))

    positive_arch_indices = {}
    positive_vomer_indices = {}
    predict_arch_boxes = {}

    # 加载 YOLO 模型
    model = YOLOv10(model_file)

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [cv2.imread(img_path) for img_path in batch_paths]

        # 将 BGR 图像（OpenCV 读取的格式）转换为 RGB
        imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        # 进行预测
        results = model(imgs_rgb)  # 你可以调整 size 参数

        for img_path, img, result in zip(batch_paths, imgs, results):
            preds = []
            has_box = False
            for box in result.boxes:
                confidence = box.conf.item()  # 获取置信度
                # if confidence <= 0.35:
                #     continue
                has_box = True
                bbox = box.xyxy[0].tolist()  # 获取 bbox 的 [x1, y1, x2, y2] 坐标
                label = model.names[int(box.cls)]  # 获取标签名称
                pred = {
                    'bbox': bbox,
                    'label': label,
                    'confidence': confidence
                }
                preds.append(pred)

            # plot_predictions(img, preds, './temp/'+os.path.basename(img_path))

            if has_box:
                key = extract_number_from_filename(os.path.basename(img_path))
                for pred in preds:
                    if pred['label'] == 'arch':
                        current_arch_confidence = positive_arch_indices.get(key, 0)
                        if pred['confidence'] > current_arch_confidence:
                            positive_arch_indices[key] = pred['confidence']
                            predict_arch_boxes[key] = pred['bbox']
                    elif pred['label'] == 'vomer':
                        current_vomer_confidence = positive_vomer_indices.get(key, 0)
                        if pred['confidence'] > current_vomer_confidence:
                            positive_vomer_indices[key] = pred['confidence']

    return positive_arch_indices, positive_vomer_indices, predict_arch_boxes


def preprocess_tiff_and_detect_arch(tiff_folder: str, model_file: str, temp_folder: str) -> Tuple[int, int, int, int, int, int]:
    # adaptive_threshold_folder = os.path.join(temp_folder, 'adaptive_threshold')
    # apply_adaptive_threshold_and_filter(tiff_folder, adaptive_threshold_folder)
    positive_arch_indices, positive_vomer_indices, predict_arch_boxes = predict_folder(tiff_folder, model_file)
    # for key in positive_arch_indices.keys():
    #     print(f"Arch: {key}, {positive_arch_indices[key]}")
    #
    # for key in positive_vomer_indices.keys():
    #     print(f"Vomer: {key}, {positive_vomer_indices[key]}")

    valid_arch, min_arch_index, max_arch_index = find_valid_indices(positive_arch_indices, 0.6, 4)
    valid_vomer, min_vomer_index, max_vomer_index = find_valid_indices(positive_vomer_indices, 0.5, 3)

    max_bound = max_arch_index if min_vomer_index < min_arch_index else min_vomer_index
    min_z, max_z = min_arch_index, min(max_arch_index, max_bound)

    min_x = 1000000
    max_x = 0
    min_y = 1000000
    max_y = 0
    for slice_id, bbox in predict_arch_boxes.items():
        if min_z < slice_id < max_z:
            if bbox[0] < min_x:
                min_x = bbox[0]
            if bbox[1] < min_y:
                min_y = bbox[1]
            if bbox[2] > max_x:
                max_x = bbox[2]
            if bbox[3] > max_y:
                max_y = bbox[3]

    return int(min_x), int(max_x), int(min_y), int(max_y), min_z, max_z


if __name__ == '__main__':
    tiff_folder = 'data/03/temp/tiff'
    model_file = 'models/detect_arch.pt'
    temp_folder = 'data/03/temp'
    res = preprocess_tiff_and_detect_arch(tiff_folder, model_file, temp_folder)
    print(res)
