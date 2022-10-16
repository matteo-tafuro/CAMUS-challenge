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





mhd_path = "../data/testing/"
mhd_patients = sorted(glob.glob(mhd_path + "*"))

masks_base_path = "testset_masks/"



output_path = "testset_submission/"
os.makedirs(output_path, exist_ok=True)




# Iterate over images and save them into the corresponding dataset
for folder in tqdm(mhd_patients):
    for i in os.listdir(folder):
        if "mhd" in i and "sequence" not in i:
            original_mhd = sitk.ReadImage(os.path.join(folder, i))
            # print("original")
            # print( original_mhd.GetSize())
            # print(sitk.GetArrayFromImage(original_mhd).shape)
            # print()

            name = os.path.basename(i)[:-4]

            mask_path = masks_base_path + name + ".nii.gz"

            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkUInt8))

            # inverse of to nifti
            original_shape = original_mhd.GetSize()[:2]
            mask = cv2.resize(mask[0, :, :], dsize=original_shape, interpolation=cv2.INTER_CUBIC)[np.newaxis, ...]

            # print(mask.shape)

            # os.makedirs(output_path + name[:-7], exist_ok=True)
            # out_path = output_path + name[:-7] + "/" + name + ".mhd"
            out_path = output_path  + "/" + name + ".mhd"
            print(out_path)

            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.SetSpacing(original_mhd.GetSpacing())

            # print(mask_sitk)

            sitk.WriteImage(mask_sitk, out_path)










