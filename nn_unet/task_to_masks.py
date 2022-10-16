import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from PIL import Image
from tqdm import tqdm


def mhd_to_array(path):
    """
    Read an *.mhd file stored in path and return it as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))


output_folder = "actual_testset_prediction_masks/"

os.makedirs(output_folder, exist_ok=True)

files = glob.glob("actual_testset_prediction/*.nii.gz")


for file in tqdm(files):

    image = sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkFloat32)).astype(np.uint8)
    im = Image.fromarray(image.squeeze(0))
    im.save(output_folder + os.path.basename(file)+".png")

