# Requirements
import glob
import os
import numpy as np
import SimpleITK as sitk
import h5py
import cv2

# Path to the testing folder
from tqdm import tqdm

test_path = "../data/testing/"

def mhd_to_array(path):
    """
    Read a *.mhd file stored in path and return it as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))


test_all_frames_list = sorted(glob.glob(test_path+"*") )

# Create hierarchical h5py file, ready to be filled with 4 datasets
f = h5py.File("../data/image_test_dataset.hdf5", "w")


f.create_dataset("test all frames", (1800, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "float32")



# Iterate over images and save them into the corresponding dataset
j = 0
for folder in tqdm(test_all_frames_list):
    for i in os.listdir(folder):
        if "mhd" in i and "sequence" not in i:
            array = mhd_to_array(os.path.join(folder, i))
            new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
            new_array = np.reshape(new_array,(384,384,1))
            new_array = new_array/255
            f["test all frames"][j,...] = new_array[...]
            j = j + 1

        
f.close()

