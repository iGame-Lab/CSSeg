caa_grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['caa_cam']
grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['original_cam']
caa_grad_cam = preprossess(img_data, caa_grad_cam)

caa_grad_cam[caa_grad_cam < 0.8 * caa_grad_cam.max()] = 0
caa_grad_cam[caa_grad_cam >= 0.8 * caa_grad_cam.max()] = 1
caa_grad_cam[grad_cam < 0.2] = 0
caa_grad_cam = find_max_eras(caa_grad_cam)
dice = dice_coeff(caa_grad_cam, label_data)


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
    if max_index == 0:
        return img
    img[img != max_index] = 0
    img[img == max_index] = 1

    return img