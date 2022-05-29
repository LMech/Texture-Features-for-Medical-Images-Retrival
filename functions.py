import os

import cv2
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

import consts


def partition_image(img: cv2.Mat, horizontal_patch: int, vertical_patch: int) -> list:
    """Partition a list of images symmetrically 
    according to the provided patch dimensions

    Args:
        img_list (cv2.Mat): Image
        horizontal_patch (int): Width of each patch
        vertical_patch (int): Height of each patch

    Returns:
        List: The partitioned patches
    """
    patches = []
    for x in range(0,img.shape[0]+1 - horizontal_patch, horizontal_patch):
        for y in range(0,img.shape[1]+1 - vertical_patch, vertical_patch):
            patch = img[x:x+horizontal_patch,y:y+vertical_patch]
            vector = np.sum(patch, axis = 1)
            patches.append(vector)
            
    return patches

def generate_schmid_kernel(kernel_size: int ,tau: int,sigma: int) -> np.array:
    """Generate a kernel according to Cordelia Schmid's
    paper (inria-00548274) for a Gabor-like filter
    
    Args:
        kernel_size (int): Applied kernel size
        tau (int): Tau value
        sigma (int): Sigma value

    Returns:
        np.array: Normalized kernel
    """
    end_point=int((kernel_size-1)/2)
    x = np.linspace(-end_point, end_point,kernel_size) 
    y = np.linspace(-end_point, end_point,kernel_size)
    
    [xv,yv] = np.meshgrid(x,y)
    
    # x^2 + y^2
    mul=((xv ** 2) + (yv ** 2))    
    
    sr= np.sqrt(mul)
    
    f = (np.cos(sr * np.pi*tau) /sigma)\
        * (np.exp(-(mul) / (2 * sigma** 2)))
    
    # Normalization
    f -= np.mean(f)
    f /= np.sum(np.abs(f))
    
    return f

def apply_schmid_filter(img: cv2.Mat) -> cv2.Mat:
    for parameter_value in consts.schmid_filter_parameter_values:
        kernel = generate_schmid_kernel(consts.schmid_kernel_size, *parameter_value)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return filtered_img

def apply_gabor_filter(img: cv2.Mat) -> cv2.Mat:
    for parameter_value in consts.gabor_filter_parameter_values:
        kernel = cv2.getGaborKernel(consts.gabor_kernel_size, *parameter_value, 0)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return filtered_img

def load_preprocessed_data() -> np.array:
    details_file = pd.read_csv('details.csv', header=0)
    img_folders_df = pd.DataFrame(details_file, columns = ['preprocessed_path'])
    vectors_list = []
    for i in tqdm(range(len(img_folders_df)), desc='Loading the data'):
        for file_path in os.listdir(img_folders_df['preprocessed_path'][i]):
            file_path = os.path.join(img_folders_df['preprocessed_path'][i], file_path)
            np_array = np.load(file_path)
            vectors_list.append(np_array)            
    return np.array(vectors_list)

def hist_match(query_hist,image_hist):
    result=0
    for i in range(query_hist.size):
        if query_hist[i]<=image_hist[i]:
            result=result+query_hist[i]
        else:
            result=result+image_hist[i] 
    return result

def write_csv(new_data: dict): 
    with open('details.csv', mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=consts.fieldnames)
        writer.writerow(new_data)
