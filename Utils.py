import math
import os.path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from skimage import measure

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def is_binary_mask(mask: np.ndarray) -> bool:
    unique_values = np.unique(mask)
    return np.array_equal(unique_values, [0, 1])


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET,
                      dice: float = None,
                      name: str = None,
                      box: np.ndarray = None,
                      returncam: bool = False) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    if box is not None:
        box = np.array(box)

    if is_binary_mask(mask):
        # 创建红色叠加颜色（仅用于 mask 为 1 的区域）
        red_color = np.array([0.8, 0, 0], dtype=np.float32)  # 红色 (1, 0, 0)

        # 创建红色叠加图像，但仅在 mask 为 1 的区域进行叠加
        heatmap = np.zeros_like(img, dtype=np.float32)
        heatmap[mask == 1] = red_color  # 在 mask == 1 的地方叠加红色
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    temp = heatmap + img
    temp = temp / np.max(temp)
    if not returncam:
        plt.imshow(np.uint8(255 * temp))
    str_title = f"{name}:"
    if dice is not None:
        str_title = str_title + f"dice:{dice}"
    plt.title(str_title)
    if box is not None and not returncam:
        if box.ndim == 2:
            for boxcaa in box:
                x0, y0 = boxcaa[0], boxcaa[1]
                w, h = boxcaa[2] - boxcaa[0], boxcaa[3] - boxcaa[1]
                plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        else:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        # 如果有 box，使用 OpenCV 在图像上绘制矩形框
    if box is not None and returncam:
        if box.ndim == 2:
            for boxcaa in box:
                x0, y0 = int(boxcaa[0]), int(boxcaa[1])
                x1, y1 = int(boxcaa[2]), int(boxcaa[3])
                cv2.rectangle(temp, (x0, y0), (x1, y1), (0, 255, 0), 1)  # 绿色框，线宽为 2
        else:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            cv2.rectangle(temp, (x0, y0), (x1, y1), (0, 255, 0), 1)  # 绿色框，线宽为 2
    if not returncam:
        plt.show()
    if returncam:
        return temp


def bounding_box(mask):
    left_up_x = -1
    left_up_y = -1
    right_down_x = -1
    right_down_y = -1
    result = []
    for i in range(mask.shape[0]):
        if mask[i].sum() > 0:
            left_up_x = i
            break
    for i in range(mask.shape[1]):
        if mask[:, i].sum() > 0:
            left_up_y = i
            break
    for i in range(mask.shape[0] - 1):
        if mask[i].sum() > 0 and mask[i + 1].sum() == 0:
            right_down_x = i
            break
    for i in range(mask.shape[1] - 1):
        if mask[:, i].sum() > 0 and mask[:, i + 1].sum() == 0:
            right_down_y = i
            break
    result = [left_up_y, left_up_x, right_down_y, right_down_x]
    return result


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def preprossess(img, mask):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] <= 1e-9):
                mask[i][j] = 0
    return mask


def find_max_eras(mask):
    # 图像读取
    img = mask
    img = np.array(img)
    img[img != 0] = 1  # 图像二值化
    # 图像实例化
    img = measure.label(img, connectivity=2)
    props = measure.regionprops(img)
    # 最大区域获取
    max_area = 0
    max_index = 0
    # props只包含像素值不为零区域的属性，因此index要从1开始
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            # index 代表每个联通区域内的像素值；prop.area代表相应连通区域内的像素个数
            max_index = index

    img[img != max_index] = 0
    img[img == max_index] = 1

    # 结果显示
    # return img
    # plt.imshow(img)
    # plt.show()
    return img


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def npztoimage(orifilepath: str, outputpath, frame_name):
    frame = np.load(orifilepath)['arr_0'] * 255
    # print(f"After scaling: min={frame.min()}, max={frame.max()}")
    frame = frame.astype(np.uint8)
    # print(f"After type conversion: min={frame.min()}, max={frame.max()}")
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # print(f"After color conversion: shape={frame.shape}, dtype={frame.dtype}")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    image = Image.fromarray(frame)
    image.save(f"{outputpath}/{frame_name}.png")


def build_frame_list(video_dir):
    frame = []
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
    for frame_name in frame_names:
        frame_path = os.path.join(video_dir, frame_name)
        frame.append(np.array(Image.open(frame_path)))
    return frame_names, frame


def build_video_list(video_dir):
    video = []
    video_names = [
        p for p in os.listdir(video_dir)
    ]
    video_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)
        video.append(video_path)
    return video


def build_cam_list(cam_dir, frame_names):
    cam = []
    for frame_name in frame_names:
        cam_path = os.path.join(cam_dir, frame_name.split('.')[0] + '.npz')
        cam.append(np.load(cam_path)['caa_cam'])
    return cam

def resize_with_interpolation(arr: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    使用最近邻插值调整数组大小（适合标签数据）
    """
    # 使用PIL进行最近邻插值，保持标签值为整数
    img = Image.fromarray(arr.astype(np.uint8))
    resized_img = img.resize(target_size, Image.NEAREST)
    return np.array(resized_img)
def transform_array(arr: np.ndarray) -> np.ndarray:
    # 将非 0 元素变为 1，0 保持不变
    transformed_arr = np.where(arr != 0, 1, 0)
    return transformed_arr


def build_label_list(label_dir, frame_names):
    label = []
    for frame_name in frame_names:
        cam_path = os.path.join(label_dir, frame_name.split('.')[0] + '.npz')
        transformed = transform_array(np.load(cam_path)['arr_0'])
        # 使用最近邻插值调整大小
        resized_label = resize_with_interpolation(transformed, (256, 256))
        label.append(resized_label)
    return label


def build_sam_mask_list(sam_mask_dir, frame_names):
    sam_mask = []
    for frame_name in frame_names:
        cam_path = os.path.join(sam_mask_dir, frame_name.split('.')[0] + '.npz')
        sam_mask.append(transform_array(np.load(cam_path)['caa_sam1_stack_pred']))
    return sam_mask


def save_images_as_video(images: list, output_path: str, fps: int = 5):
    if not images:
        raise ValueError("Image list is empty")

    # 获取视频的宽高
    height, width = images[0].shape[:2]
    is_color = len(images[0].shape) == 3 and images[0].shape[2] == 3
    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

    for img in images:
        img_uint8 = np.uint8(255 * np.clip(img, 0, 1))

        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(img_uint8)

    # 释放 VideoWriter 对象
    video_writer.release()


def save_images(images: list, output_dir: str, file_prefix: str = "image"):
    if not images:
        raise ValueError("Image list is empty")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(images):
        # 将浮点数图像转为 uint8 格式
        img_uint8 = np.uint8(255 * np.clip(img, 0, 1))

        # 检查是否需要转换为 BGR 格式（OpenCV 默认 BGR）
        if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # 构造文件名
        file_name = f"{file_prefix}_{i:04d}.png"  # 文件名如 image_0001.png
        file_path = os.path.join(output_dir, file_name)

        # 保存图像
        cv2.imwrite(file_path, img_uint8)

    print(f"Images saved to {output_dir}")


def merge_videos(video_paths, titles, output_path, dice):
    # 打开所有视频文件
    captures = [cv2.VideoCapture(vp) for vp in video_paths]

    # 检查视频是否成功打开
    if not all(cap.isOpened() for cap in captures):
        raise ValueError("无法打开所有视频文件")

    # 获取第一个视频的属性
    frame_width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captures[0].get(cv2.CAP_PROP_FPS)

    # 创建标题帧
    title_frames = [add_global_title(title, frame_width, 50) for title in titles]
    height_with_title = frame_height + 80  # 增加高度以容纳标题

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 4, height_with_title))  # 增加宽度以容纳三列视频
    frame_idx = 0  # 初始化帧索引
    while True:
        # 读取每个视频的当前帧
        frames = [cap.read()[1] for cap in captures]

        # 如果所有视频都到达末尾，则停止
        if any(frame is None for frame in frames):
            break

        # 确保所有帧的尺寸一致
        for i, frame in enumerate(frames):
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                raise ValueError(f"视频 {video_paths[i]} 的帧尺寸不一致。")

        # 为每个视频添加标题
        titled_frames = []
        for i in range(len(frames)):
            title_frame = title_frames[i].copy()
            dice_value = round(dice[i][frame_idx], 4)  # 动态计算 dice 值
            # iou_value = round(iou[i][frame_idx],4)
            cv2.putText(title_frame, f"Dice: {dice_value}", (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(title_frame, f"mIou: {iou_value}", (90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            titled_frames.append(np.vstack([title_frame, frames[i]]))
        # 拼接每一帧
        combined_frame = np.hstack(titled_frames)

        # 将合并后的帧写入新的视频文件
        output_writer.write(combined_frame)
        frame_idx += 1

    # 释放所有资源
    for cap in captures:
        cap.release()
    output_writer.release()


def add_global_title(title, width, height):
    """创建一个带有中文标题的帧"""
    if not isinstance(title, str):
        raise ValueError("标题必须是字符串类型")

    # 创建一个全黑的标题帧
    title_height = 80  # 标题的高度
    title_frame = np.zeros((title_height, width, 3), dtype=np.uint8)  # 使用全黑背景

    # 使用 Pillow 在标题帧上绘制文本
    pil_img = Image.fromarray(title_frame)
    draw = ImageDraw.Draw(pil_img)

    # 计算文本大小和位置
    text_bbox = draw.textbbox((0, 0), title)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = (width - text_width) // 2
    text_y = (title_height - text_height) // 2 + 20

    # 绘制白色文本
    draw.text((text_x, text_y), title, fill=(255, 255, 255))

    return np.array(pil_img)


def compute_iou(img1, img2):
    """计算两张二值图像之间的IoU"""
    intersection = np.sum((img1 == 1) & (img2 == 1))
    union = np.sum((img1 == 1) | (img2 == 1))
    return intersection / union if union != 0 else 0


def compute_average_iou_for_stack(images, num_neighbors=2):
    """计算每张图像与其相邻图像的平均IoU"""
    num_images = images.shape[0]
    iou_array = np.zeros(num_images, dtype=np.float32)

    for index in range(num_images):
        # 计算相邻图像的起始和结束索引
        start_idx = max(0, index - num_neighbors // 2)
        end_idx = min(num_images, index + num_neighbors // 2 + 1)

        # 确保相邻图像数量为10
        if end_idx - start_idx < num_neighbors:
            if start_idx == 0:
                end_idx = min(num_images, start_idx + num_neighbors)
            else:
                start_idx = max(0, end_idx - num_neighbors)

        # 计算IoU值
        iou_list = []
        for i in range(start_idx, end_idx):
            if i != index:
                iou = compute_iou(images[index], images[i])
                iou_list.append(iou)

        # 计算当前图像的平均IoU
        iou_array[index] = np.mean(iou_list) if iou_list else 0

    return iou_array


def normalize_sum_of_arrays(list_of_arrays):
    """
    输入一个保存多个二维 numpy 数组的列表，输出累加并归一化后的二维数组。

    参数:
    list_of_arrays (list): 包含多个二维 numpy 数组的列表

    返回:
    numpy.ndarray: 归一化后的二维数组
    """
    if not list_of_arrays:
        raise ValueError("输入的列表为空")

    # 确保所有数组的形状一致
    shape = list_of_arrays[0].shape
    for array in list_of_arrays:
        if array.shape != shape:
            raise ValueError("所有数组的形状必须相同")

    # 初始化累加数组
    sum_array = np.zeros(shape)

    # 累加所有数组
    for array in list_of_arrays:
        sum_array += array

    # 计算最大值和最小值
    max_value = np.max(sum_array)
    min_value = np.min(sum_array)

    # 归一化
    if max_value != min_value:
        normalized_array = (sum_array - min_value) / (max_value - min_value)
    else:
        normalized_array = np.zeros_like(sum_array)  # 如果最大值和最小值相同

    return normalized_array


def get_intersection(box1, box2):
    # 计算交集的左上角坐标
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])

    # 计算交集的右下角坐标
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # 判断交集是否存在
    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
        return np.array([intersect_x1, intersect_y1, intersect_x2, intersect_y2])
    else:
        return None


# 填充box数组
def fill_none_with_nearest(arr):
    # 获取数组长度
    n = len(arr)

    # 用于记录最近的非 None 元素的索引
    nearest_left = [None] * n  # 记录每个位置往左最近的非 None 索引
    nearest_right = [None] * n  # 记录每个位置往右最近的非 None 索引

    # 从左向右扫描，记录最近的非 None 元素索引
    last_non_none = None
    for i in range(n):
        if arr[i] is not None:
            last_non_none = arr[i]
        nearest_left[i] = last_non_none

    # 从右向左扫描，记录最近的非 None 元素索引
    last_non_none = None
    for i in range(n - 1, -1, -1):
        if arr[i] is not None:
            last_non_none = arr[i]
        nearest_right[i] = last_non_none

    # 遍历原数组，替换 None 元素
    for i in range(n):
        if arr[i] is None:
            # 比较左右两边最近的非 None 元素，选择距离最近的
            if nearest_left[i] is not None and nearest_right[i] is not None:
                # 如果左右两边都有非 None 元素，取最近的那个
                left_distance = i - nearest_left.index(nearest_left[i])
                right_distance = nearest_right.index(nearest_right[i]) - i
                if left_distance <= right_distance:
                    arr[i] = nearest_left[i]
                else:
                    arr[i] = nearest_right[i]
            elif nearest_left[i] is not None:
                arr[i] = nearest_left[i]
            elif nearest_right[i] is not None:
                arr[i] = nearest_right[i]

    return arr


def getboxcenter(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2.0
    center_y = (y1 + y2) // 2.0
    return (center_x, center_y)


def centerdistance(center1, center2):
    x1, y1 = center1  # 第一个中心点
    x2, y2 = center2  # 第二个中心点
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def get_distances_to_vertices(box):
    x1, y1, x2, y2 = box
    center = getboxcenter(box)
    top_left = (x1, y1)
    return distance_from_center_to_vertex(center, top_left)


def distance_from_center_to_vertex(center, vertex):
    center_x, center_y = center
    x, y = vertex
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)


def get_box_dimensions(box):
    """
    计算边界框的宽度和高度。

    参数：
    box (tuple): 边界框的坐标，格式为 (x1, y1, x2, y2)。

    返回：
    tuple: 边界框的宽度和高度。
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return width, height


def get_largest_box(boxes):
    # 定义初始最大面积为0，初始最大box为None
    max_area = 0
    largest_box = None

    # 遍历所有box，计算每个box的面积并更新最大box
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)  # 计算面积

        if area > max_area:
            max_area = area
            largest_box = box

    return largest_box


def is_point_in_box(box, point):
    """
    判断一个点是否在边界框内。

    参数：
    box (tuple): 边界框的坐标，格式为 (x1, y1, x2, y2)。
    point (tuple): 点的坐标，格式为 (x, y)。

    返回：
    bool: 如果点在边界框内返回 True，否则返回 False。
    """
    x1, y1, x2, y2 = box
    x, y = point

    # 判断点是否在边界框内
    return x1 <= x <= x2 and y1 <= y <= y2


def process_image(image, seg=None, box=None):
    # 假设 image 是原始MRI，seg 是初步分割结果

    if box is not None:
        # 创建一个与 seg 相同大小的空数组
        zero_array = np.zeros_like(image)

        # 遍历每个框并在 zero_array 上进行标记
        for j_ in range(0, len(box)):
            x1, y1, x2, y2 = box[j_]
            zero_array[y1:y2 + 1, x1:x2 + 1] = 1
        # 计算灰度特征
        mean_intensity = np.mean(image[zero_array == 1])
    else:
        # 计算灰度特征
        mean_intensity = np.mean(image[seg == 1])

    # 返回特征结果
    return mean_intensity


def find_all_eras(mask):
    # 图像读取
    img = np.array(mask, dtype=np.uint8)  # 确保输入类型为 uint8
    img[img != 0] = 1  # 图像二值化

    # 图像实例化
    labeled_img = measure.label(img, connectivity=2)
    props = measure.regionprops(labeled_img)

    # 根据区域的面积对 props 进行排序，按面积从大到小排序
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)

    # 保存每个区域的二值图
    all_regions = []

    # 遍历每个区域，将其提取为二值图
    for prop in props_sorted:
        region_mask = np.zeros_like(labeled_img, dtype=np.uint8)  # 明确指定 dtype 为 uint8
        region_mask[labeled_img == prop.label] = 1
        all_regions.append(region_mask)

    return all_regions


# 使用整张图片的平均灰度去做联通区间的处理
def connection_interval_processing(img_data_list, SAM_LIST):
    for cnt in np.arange(0, len(img_data_list)):
        img_data_mask = img_data_list[cnt].copy()
        img_data_mask[img_data_mask > 0.2] = 1
        SAM_mask_caa = SAM_LIST[cnt].copy()
        tmp_SAM_mask_caa = np.zeros_like(SAM_mask_caa)
        SAM_mask_caa = find_all_eras(SAM_mask_caa)
        if len(SAM_mask_caa) == 0:
            continue
        mean_huidu = process_image(img_data_list[cnt], img_data_mask)
        tmp_SAM_mask_caa += SAM_mask_caa[0]
        for mask_SAM in SAM_mask_caa:
            huidu = process_image(img_data_list[cnt], mask_SAM)
            if huidu > mean_huidu:
                tmp_SAM_mask_caa += mask_SAM
        tmp_SAM_mask_caa[tmp_SAM_mask_caa > 0] = 1
        if process_image(img_data_list[cnt], tmp_SAM_mask_caa) < mean_huidu and cnt != 0:
            if process_image(img_data_list[cnt], tmp_SAM_mask_caa) < process_image(img_data_list[cnt],
                                                                                   SAM_LIST[cnt - 1]):
                tmp_SAM_mask_caa = SAM_LIST[cnt - 1]
        tmp_SAM_mask_caa[tmp_SAM_mask_caa > 0] = 1
        SAM_LIST[cnt] = tmp_SAM_mask_caa

    for cnt in np.arange(len(img_data_list) - 1, -1, -1):
        img_data_mask = img_data_list[cnt].copy()
        img_data_mask[img_data_mask > 0] = 1
        SAM_mask_caa = SAM_LIST[cnt].copy()
        tmp_SAM_mask_caa = np.zeros_like(SAM_mask_caa)
        SAM_mask_caa = find_all_eras(SAM_mask_caa)
        if len(SAM_mask_caa) == 0:
            continue
        mean_huidu = process_image(img_data_list[cnt], img_data_mask)
        tmp_SAM_mask_caa += SAM_mask_caa[0]
        for mask_SAM in SAM_mask_caa:
            huidu = process_image(img_data_list[cnt], mask_SAM)
            if huidu > mean_huidu:
                tmp_SAM_mask_caa += mask_SAM
        tmp_SAM_mask_caa[tmp_SAM_mask_caa > 0] = 1
        if process_image(img_data_list[cnt], tmp_SAM_mask_caa) < mean_huidu and cnt != len(img_data_list) - 1:
            if process_image(img_data_list[cnt], tmp_SAM_mask_caa) < process_image(img_data_list[cnt],
                                                                                   SAM_LIST[cnt + 1]):
                tmp_SAM_mask_caa = SAM_LIST[cnt + 1]
        tmp_SAM_mask_caa[tmp_SAM_mask_caa > 0] = 1
        SAM_LIST[cnt] = tmp_SAM_mask_caa

    return SAM_LIST


def scale_box_to_target(box, target_size, original_size=(256, 256)):
    # 计算从原始尺寸到目标尺寸的缩放比例
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    scale = np.array([scale_x, scale_y, scale_x, scale_y])

    # 调整 box 坐标到新的尺寸
    scaled_box = box * scale
    return scaled_box


def concatenate_images_and_save_shuiping(img_list=None,ori_cam=None, cam_img=None, sam_img=None, label_img=None,
                                output_path="output.png", mode='horizontal'):
    """
    拼接多张图像并保存为文件，如果只有一张图像输入，则直接保存该图像。

    参数:
    - img_list, cam_img, label_img, sam_img, ori_img: numpy数组，形状为 (H, W, 3) 的图像数据
    - output_path: str，保存的文件路径
    - mode: str，拼接方式，'horizontal' 表示横向拼接，'vertical' 表示纵向拼接

    返回:
    - 保存的文件路径
    """

    def prepare_image(image):
        # 确保图像为 (H, W, 3) 格式且在 [0, 255] 范围内
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = image[20:236, 20:236]  # 裁剪
        if image.shape[-1] != 3:
            image = np.stack([image] * 3, axis=-1)
        return image

    # 筛选非空图像并预处理
    images = [img for img in [img_list, ori_cam, cam_img, sam_img ,label_img] if img is not None]
    if not images:
        raise ValueError("至少需要提供一个图像")

    # 仅一张图像时直接保存
    if len(images) == 1:
        single_image = prepare_image(images[0])
        Image.fromarray(single_image).save(output_path)
        print(f"单张图像已保存到 {output_path}")
        return output_path

    # 多张图像时进行拼接
    prepared_images = [prepare_image(img) for img in images]

    # 选择拼接方式
    if mode == 'horizontal':
        # 水平拼接所有图像[2,6,8](@ref)
        concatenated_image = np.concatenate(prepared_images, axis=1)
    elif mode == 'vertical':
        # 垂直拼接所有图像[2,6,8](@ref)
        concatenated_image = np.concatenate(prepared_images, axis=0)
    else:
        raise ValueError("mode 参数只能为 'horizontal' 或 'vertical'")

    # 将拼接后的图像保存
    Image.fromarray(concatenated_image).save(output_path)
    print(f"拼接图像已保存到 {output_path}")
    return output_path
def concatenate_images_and_save(img_list=None, cam_img=None, label_img=None, sam_img=None, ori_img=None,
                                output_path="output.png", mode='horizontal'):
    """
    拼接多张图像并保存为文件，如果只有一张图像输入，则直接保存该图像。

    参数:
    - cam_img, label_img, sam_img, ori_img: numpy数组，形状为 (H, W, 3) 的图像数据
    - output_path: str，保存的文件路径
    - mode: str，拼接方式，'horizontal' 表示横向拼接，'vertical' 表示纵向拼接

    返回:
    - 保存的文件路径
    """

    def prepare_image(image):
        # 确保图像为 (H, W, 3) 格式且在 [0, 255] 范围内
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = image[20:236, 20:236]
        if image.shape[-1] != 3:
            image = np.stack([image] * 3, axis=-1)
        return image

    # # 筛选非空图像并预处理
    # images = [img for img in [img_list,cam_img, label_img, sam_img, ori_img] if img is not None]
    # if not images:
    #     raise ValueError("至少需要提供一个图像")
    #
    # # 仅一张图像时直接保存
    # if len(images) == 1:
    #     single_image = prepare_image(images[0])
    #     Image.fromarray(single_image).save(output_path)
    #     print(f"单张图像已保存到 {output_path}")
    #     # return output_path
    #
    # # 多张图像时进行拼接
    # prepared_images = [prepare_image(img) for img in images]
    #
    # # 选择拼接方式
    # if mode == 'horizontal':
    #     concatenated_image = np.concatenate(prepared_images, axis=1)
    # elif mode == 'vertical':
    #     concatenated_image = np.concatenate(prepared_images, axis=0)
    # else:
    #     raise ValueError("mode 参数只能为 'horizontal' 或 'vertical'")
    #
    # # 将拼接后的图像保存
    # Image.fromarray(concatenated_image).save(output_path)
    # # print(f"拼接图像已保存到 {output_path}")
    # # return output_path
    # 步骤1：水平拼接 img_list 和 cam_img
    if img_list is not None and cam_img is not None:
        img_list_prep = prepare_image(img_list)
        cam_img_prep = prepare_image(cam_img)
        top_row = np.concatenate([img_list_prep, cam_img_prep], axis=1)  # 水平拼接[6,7](@ref)
    else:
        raise ValueError("img_list和cam_img必须提供")

    # 步骤2：水平拼接 label_img 和 sam_img
    if label_img is not None and sam_img is not None:
        label_img_prep = prepare_image(label_img)
        sam_img_prep = prepare_image(sam_img)
        bottom_row = np.concatenate([label_img_prep, sam_img_prep], axis=1)  # 水平拼接[6,7](@ref)
    else:
        raise ValueError("label_img和sam_img必须提供")

    # 步骤3：垂直拼接两个结果
    final_image = np.concatenate([top_row, bottom_row], axis=0)  # 垂直拼接[6,7](@ref)
    Image.fromarray(final_image).save(output_path)


def vertical_stitch_with_dashed_line(img1_path, img2_path, output_path,
                                     gap_size=16, line_color=(65,105,225),
                                     dash_pattern=(8, 4), line_width=2):
    """
    水平拼接图片并在间隔中央绘制竖直虚线
    :param gap_size: 间隔区域宽度（建议≥20像素）
    :param line_color: 虚线颜色（RGB元组）
    :param dash_pattern: 虚线模式（实线长度, 空白长度）
    :param line_width: 线条粗细
    """
    # 打开并统一图片高度
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    target_height = max(img1.height, img2.height)

    # 等比例缩放（保持宽高比）
    def resize_image(img):
        scale = target_height / img.height
        new_width = int(img.width * scale)
        return img.resize((new_width, target_height), Image.LANCZOS)

    img1 = resize_image(img1)
    img2 = resize_image(img2)

    # 创建白色间隔画布
    gap_img = Image.new('RGB', (gap_size, target_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(gap_img)

    # 绘制居中竖直虚线（网页3方法改进）
    x_center = gap_size // 2 -1 # 垂直中线坐标
    dash_len, space_len = dash_pattern
    y_start = 0

    while y_start < target_height:
        # 计算当前虚线段的结束位置
        segment_end = min(y_start + dash_len, target_height)
        # 绘制实线部分（网页6方法）
        draw.line([(x_center, y_start), (x_center, segment_end)],
                  fill=line_color, width=line_width)
        # 更新起始位置（加上空白段）
        y_start += dash_len + space_len

    # 拼接三部分
    total_width = img1.width + gap_size + img2.width
    result = Image.new('RGB', (total_width, target_height))
    result.paste(img1, (0, 0))
    result.paste(gap_img, (img1.width, 0))
    result.paste(img2, (img1.width + gap_size, 0))

    result.save(output_path)
    return output_path

def concatenate_images_with_gap(img_list=None, cam_img=None, label_img=None, sam_img=None,
                                output_path="output.png", gap_size=5):
    """
    支持水平/垂直拼接时添加白色间距的增强版
    """

    def prepare_image(image):
        # 保持原有裁剪逻辑（216x216）
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = image[20:236, 20:236]
        if image.shape[-1] != 3:
            image = np.stack([image] * 3, axis=-1)
        return image

    # 创建白色间隔条（水平或垂直）
    def create_gap(shape, direction='horizontal'):
        if direction == 'horizontal':
            return np.ones((shape[0], gap_size, 3), dtype=np.uint8) * 255
        else:
            return np.ones((gap_size, shape[1], 3), dtype=np.uint8) * 255

    # 分步拼接逻辑
    # 步骤1：水平拼接 img_list 和 cam_img（带间距）
    img_list_prep = prepare_image(img_list)
    cam_img_prep = prepare_image(cam_img)
    label_img_prep = prepare_image(label_img)
    sam_img_prep = prepare_image(sam_img)
    gap_h = create_gap(img_list_prep.shape, 'horizontal')
    top_row = np.concatenate([img_list_prep, gap_h, cam_img_prep,gap_h,sam_img_prep,gap_h,label_img_prep], axis=1)  # 水平拼接带间距

    # 步骤2：水平拼接 label_img 和 sam_img（带间距）
    # bottom_row = np.concatenate([label_img_prep, gap_h, sam_img_prep], axis=1)
    #
    # # 步骤3：垂直拼接两行（带间距）
    # gap_v = create_gap(top_row.shape, 'vertical')
    # final_image = np.concatenate([top_row, gap_v, bottom_row], axis=0)

    # 保存结果
    Image.fromarray(top_row).save(output_path)
    print(f"最终图像已保存到 {output_path}")
    return output_path

def chunzhipinjie(img1_path, img2_path,img3_path, output_path="output.png",gap_size=5):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    img3 = np.array(Image.open(img3_path))
    def create_gap(shape, direction='horizontal'):
        if direction == 'horizontal':
            return np.ones((shape[0], gap_size, 3), dtype=np.uint8) * 255
        else:
            return np.ones((gap_size, shape[1], 3), dtype=np.uint8) * 255
    gap_h = create_gap(img1.shape, 'vertical')
    final_image = np.concatenate([img1, gap_h, img2], axis=0)
    Image.fromarray(final_image).save(output_path)
def concatenate_saved_images(img1_path, img2_path, output_path="output.png", mode='horizontal'):
    # 打开两张图片并转换为numpy数组
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))

    # 选择拼接方式
    if mode == 'horizontal':
        concatenated_image = np.concatenate((img1, img2), axis=1)  # 横向拼接
    elif mode == 'vertical':
        concatenated_image = np.concatenate((img1, img2), axis=0)  # 纵向拼接
    else:
        raise ValueError("mode 参数只能为 'horizontal' 或 'vertical'")

    # 保存拼接后的图像
    Image.fromarray(concatenated_image).save(output_path)
    print(f"拼接图像已保存到 {output_path}")
    return output_path


def horizontal_stitch_with_line(img1_path, img2_path, output_path, gap_size=20, line_color=(0, 0, 255)):
    """
    水平拼接图片并添加蓝色虚线间隔
    :param gap_size: 间隔宽度（像素）
    :param line_color: 虚线颜色（BGR格式元组）
    """
    # 打开并统一图片高度
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    target_height = max(img1.height, img2.height)

    # 等比例缩放
    def resize_image(img):
        scale = target_height / img.height
        new_width = int(img.width * scale)
        return img.resize((new_width, target_height), Image.LANCZOS)

    img1 = resize_image(img1)
    img2 = resize_image(img2)

    # 生成蓝色虚线（纵向条纹）
    def create_dashed_line():
        pattern = np.zeros((target_height, gap_size, 3), dtype=np.uint8)
        dash_length = 8  # 虚线片段长度
        space_length = 4  # 空白长度

        # 生成交替条纹
        for y in range(0, target_height, dash_length + space_length):
            end = min(y + dash_length, target_height)
            pattern[y:end, :, :] = line_color  # 填充蓝色

        return Image.fromarray(pattern)

    # 创建带虚线的间隔
    gap_img = create_dashed_line()

    # 拼接图像
    total_width = img1.width + gap_size + img2.width
    result = Image.new('RGB', (total_width, target_height))
    result.paste(img1, (0, 0))
    # result.paste(gap_img, (img1.width, 0))
    result.paste(img2, (img1.width + gap_size, 0))

    result.save(output_path)
    return output_path

def horizontal_stitch_with_line(img1_path, img2_path,img3_path, output_path, gap_size=20, line_color=(0, 0, 255)):
    """
    水平拼接图片并添加蓝色虚线间隔
    :param gap_size: 间隔宽度（像素）
    :param line_color: 虚线颜色（BGR格式元组）
    """
    # 打开并统一图片高度
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img2_path)
    target_width = img1.width


    # 生成蓝色虚线（纵向条纹）
    def create_dashed_line():
        pattern = np.zeros((gap_size,target_width, 3), dtype=np.uint8)
        dash_length = 8  # 虚线片段长度
        space_length = 4  # 空白长度

        # 生成交替条纹
        for y in range(0, target_width, dash_length + space_length):
            end = min(y + dash_length, target_width)
            pattern[y:end, :, :] = line_color  # 填充蓝色

        return Image.fromarray(pattern)

    # 创建带虚线的间隔
    gap_img = create_dashed_line()

    # 拼接图像
    total_height = img1.height + gap_size + img2.height
    result = Image.new('RGB', (total_height, target_width))
    result.paste(img1, (0, 0))
    # result.paste(gap_img, (img1.width, 0))
    result.paste(img2, (img1.width + gap_size, 0))

    result.save(output_path)
    return output_path


def vertical_stitch_with_lines(img1_path, img2_path, img3_path, output_path,
                               gap_size=5, line_color=(0, 0, 255),
                               dash_length=8, space_length=4):
    """
    三图垂直拼接工具
    :param gap_size: 间隔高度（像素）
    :param line_color: 虚线颜色（RGB格式元组）
    :param dash_length: 虚线片段长度
    :param space_length: 虚线间隔长度
    """
    # 打开图片并统一宽度[6,8](@ref)
    images = [Image.open(p) for p in (img1_path, img2_path, img3_path)]
    max_width = max(img.width for img in images)

    # 等比例缩放函数[9](@ref)
    def resize_image(img):
        scale = max_width / img.width
        new_height = int(img.height * scale)
        return img.resize((max_width, new_height), Image.LANCZOS)

    resized_images = [resize_image(img) for img in images]

    # 创建间隔图像
    def create_gap_pattern(is_dashed):
        """生成间隔图像（支持纯色/虚线）"""
        pattern = np.full((2, max_width, 3), 255, dtype=np.uint8)  # 白色背景

        if is_dashed:
            # 横向虚线绘制[1,10](@ref)
            for y in range(gap_size):
                x = 0
                while x < max_width:
                    end = min(x + dash_length, max_width)
                    pattern[y, x:end] = line_color  # 绘制虚线片段
                    x += dash_length + space_length

        return Image.fromarray(pattern)

    # 生成两种间隔[6](@ref)
    white_gap = create_gap_pattern(is_dashed=False)  # img1-img2间
    dashed_gap = create_gap_pattern(is_dashed=True)  # img2-img3间

    # 计算总高度[4](@ref)
    total_height = sum(img.height for img in resized_images) + 2 * gap_size

    # 创建画布并拼接[8,10](@ref)
    result = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0

    # 第一张图
    result.paste(resized_images[0], (0, y_offset))
    y_offset += resized_images[0].height

    # 白色间隔
    result.paste(white_gap, (0, y_offset))
    y_offset += gap_size

    # 第二张图
    result.paste(resized_images[1], (0, y_offset))
    y_offset += resized_images[1].height

    # 蓝色虚线间隔
    result.paste(dashed_gap, (0, y_offset))
    y_offset += gap_size

    # 第三张图
    result.paste(resized_images[2], (0, y_offset))

    result.save(output_path)
    return output_path



