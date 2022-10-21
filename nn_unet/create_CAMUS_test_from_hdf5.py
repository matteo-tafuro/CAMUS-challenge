# Requirements
import glob
import os

import SimpleITK as sitk
import cv2
import numpy as np
# Path to the testing folder
from tqdm import tqdm


def mhd_to_array(path):
    """
    Read a *.mhd file stored in path and return it as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))


def convert_array_to_nifti(img: np.array, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!

    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net

    If Transform is not None it will be applied to the image after loading.

    Segmentations will be converted to np.uint32!

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """

    print("shape", img.shape)
    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'


    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


# test_path = "../data/testing/"
# test_all_frames_list = sorted(glob.glob(test_path + "*"))
#
# output_path = "testset_nifti/"
#
# os.makedirs(output_path, exist_ok=True)
#
#
# # Iterate over images and save them into the corresponding dataset
# for folder in tqdm(test_all_frames_list):
#     for i in os.listdir(folder):
#         if "mhd" in i and "sequence" not in i:
#             array = mhd_to_array(os.path.join(folder, i))
#             new_array = cv2.resize(array[0, :, :], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
#             new_array = np.reshape(new_array, (384, 384))
#             new_array = new_array / 255
#             convert_array_to_nifti(new_array, output_path+ i[:-4])


test_all_frames_list = sorted(glob.glob("../data/training/2ch/frames/*.mhd")) + sorted(glob.glob("../data/training/4ch/frames/*.mhd"))

output_path = "trainset_nifti/"

os.makedirs(output_path, exist_ok=True)


# Iterate over images and save them into the corresponding dataset
for file in tqdm(test_all_frames_list):
    file_base = os.path.basename(file)
    array = mhd_to_array(file)
    new_array = cv2.resize(array[0, :, :], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
    new_array = np.reshape(new_array, (384, 384))
    new_array = new_array / 255
    convert_array_to_nifti(new_array, output_path+ file_base[:-4])


