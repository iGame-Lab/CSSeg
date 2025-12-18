import os
import dicom2nifti # pip install dicom2nifti==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np #  pip install numpy==1.20.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
from PIL import Image # pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
from typing import List
import SimpleITK as sitk # pip install SimpleITK==1.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    # refer to https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/utilities/file_and_folder_operations.py
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    # refer to https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/utilities/file_and_folder_operations.py
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def load_png_stack(folder):
    pngs = subfiles(folder, suffix="png")
    pngs.sort()
    loaded = []
    for p in pngs:
        loaded.append(np.array(Image.open(p)))
    loaded = np.stack(loaded, 0)[::-1]
    return loaded

def convert_CT_seg(loaded_png):
    return loaded_png.astype(np.uint16)

def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image

def convert_MR_seg(loaded_png):
    result = np.zeros(loaded_png.shape)
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1 # liver
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2 # right kidney
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3 # left kidney
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4 # spleen
    return result

if __name__ == "__main__":

    pass

    ############################
    ##### for ct train set #####
    ############################
    # path_to_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets\Train_Sets'
    # path_to_save_for_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets_nifti_ct'
    # path_to_CHAOS_Train_Sets_ct = os.path.join(path_to_CHAOS_Train_Sets, "CT")
    # patients = subdirs(path_to_CHAOS_Train_Sets_ct, join=False)
    # for p in patients:
    #     # for image
    #     img_dir =  os.path.join(path_to_CHAOS_Train_Sets_ct, p, "DICOM_anon")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_image.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

    #     # for segmentation
    #     gt_dir = os.path.join(path_to_CHAOS_Train_Sets_ct, p, "Ground")
    #     # seg = convert_CT_seg(load_png_stack(gt_dir)[::-1])
    #     seg = convert_CT_seg(load_png_stack(gt_dir))
    #     img_sitk = sitk.ReadImage(img_outfile)
    #     img_npy = sitk.GetArrayFromImage(img_sitk)
    #     seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
    #     seg_itk = copy_geometry(seg_itk, img_sitk)
    #     seg_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_segmentation.nii.gz")
    #     sitk.WriteImage(seg_itk, seg_outfile)

    ###########################
    ##### for ct test set #####
    ###########################
    # path_to_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets\Test_Sets'
    # path_to_save_for_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets_nifti_ct'
    # path_to_CHAOS_Test_Sets_ct = os.path.join(path_to_CHAOS_Test_Sets, "CT")
    # patients = subdirs(path_to_CHAOS_Test_Sets_ct, join=False)
    # for p in patients:
    #     # for image
    #     img_dir =  os.path.join(path_to_CHAOS_Test_Sets_ct, p, "DICOM_anon")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Test_Sets, p+"_image.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

    #####################################
    ##### for mr train set (T1DUAL) #####
    #####################################
    # path_to_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets\Train_Sets'
    # path_to_save_for_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets_nifti_mr_T1DUAL'
    # path_to_CHAOS_Train_Sets_mr = os.path.join(path_to_CHAOS_Train_Sets, "MR")
    # patients = subdirs(path_to_CHAOS_Train_Sets_mr, join=False)
    # for p in patients:
    #     # for image, T1DUAL
    #     img_dir =  os.path.join(path_to_CHAOS_Train_Sets_mr, p, "T1DUAL", "DICOM_anon", "InPhase")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_image_T1DUAL_InPhase.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)
    #     img_dir =  os.path.join(path_to_CHAOS_Train_Sets_mr, p, "T1DUAL", "DICOM_anon", "OutPhase")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_image_T1DUAL_OutPhase.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

    #     # for segmentation
    #     gt_dir = os.path.join(path_to_CHAOS_Train_Sets_mr, p, "T1DUAL", "Ground")
    #     seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])
    #     img_sitk = sitk.ReadImage(img_outfile)
    #     img_npy = sitk.GetArrayFromImage(img_sitk)
    #     seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
    #     seg_itk = copy_geometry(seg_itk, img_sitk)
    #     seg_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_segmentation_T1DUAL.nii.gz")
    #     sitk.WriteImage(seg_itk, seg_outfile)

    ####################################
    ##### for mr test set (T1DUAL) #####
    ####################################
    # path_to_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets\Test_Sets'
    # path_to_save_for_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets_nifti_mr_T1DUAL'
    # path_to_CHAOS_Test_Sets_mr = os.path.join(path_to_CHAOS_Test_Sets, "MR")
    # patients = subdirs(path_to_CHAOS_Test_Sets_mr, join=False)
    # for p in patients:
    #     # for image, T1DUAL
    #     img_dir =  os.path.join(path_to_CHAOS_Test_Sets_mr, p, "T1DUAL", "DICOM_anon", "InPhase")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Test_Sets, p+"_image_T1DUAL_InPhase.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)
    #     img_dir =  os.path.join(path_to_CHAOS_Test_Sets_mr, p, "T1DUAL", "DICOM_anon", "OutPhase")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Test_Sets, p+"_image_T1DUAL_OutPhase.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

    ####################################
    #### for mr train set (T2SPIR) #####
    ####################################
    # path_to_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets\Train_Sets'
    path_to_CHAOS_Train_Sets ="F:/brats/Train_Sets"
    # path_to_save_for_CHAOS_Train_Sets = 'E:\chaos_dataset_conversion\CHAOS_Train_Sets_nifti_mr_T2SPIR'
    path_to_save_for_CHAOS_Train_Sets = 'F:/brats/Train_Sets/CHAOS_Train_Sets_nifti_mr_T2SPIR'
    path_to_CHAOS_Train_Sets_mr = os.path.join(path_to_CHAOS_Train_Sets, "MR")

    patients = subdirs(path_to_CHAOS_Train_Sets_mr, join=False)
    for p in patients:
        # for image, T1DUAL
        img_dir =  os.path.join(path_to_CHAOS_Train_Sets_mr, p, "T2SPIR", "DICOM_anon")
        img_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_image_T2SPIR.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        # for segmentation
        gt_dir = os.path.join(path_to_CHAOS_Train_Sets_mr, p, "T2SPIR", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])
        img_sitk = sitk.ReadImage(img_outfile)
        img_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        seg_outfile = os.path.join(path_to_save_for_CHAOS_Train_Sets, p+"_segmentation_T2SPIR.nii.gz")
        sitk.WriteImage(seg_itk, seg_outfile)

    ####################################
    ##### for mr test set (T2SPIR) #####
    ####################################
    # path_to_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets\Test_Sets'
    # path_to_save_for_CHAOS_Test_Sets = 'E:\chaos_dataset_conversion\CHAOS_Test_Sets_nifti_mr_T2SPIR'
    # path_to_CHAOS_Test_Sets_mr = os.path.join(path_to_CHAOS_Test_Sets, "MR")
    # patients = subdirs(path_to_CHAOS_Test_Sets_mr, join=False)
    # for p in patients:
    #     # for image, T1DUAL
    #     img_dir =  os.path.join(path_to_CHAOS_Test_Sets_mr, p, "T2SPIR", "DICOM_anon")
    #     img_outfile = os.path.join(path_to_save_for_CHAOS_Test_Sets, p+"_image_T2SPIR.nii.gz")
    #     _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)