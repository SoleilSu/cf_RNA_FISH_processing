# =============================================================================
# imports
# =============================================================================
import glob
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import aicspylibczi as aplc
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import argparse
import yaml

# =============================================================================
# This is a executable python script that takes as arguments:
# configuration file (containing all of the parameters used in the functions)
# input filename
# output segmentation filename
# output segmentation table filename
# The script should load the image, reshape it, segment it, extract the properties table from the segmentation, save the segmentation, and save the properties table
# =============================================================================

# =============================================================================
# Functions
# =============================================================================
def segmented_array(img_sum,thresh_intensity,min_distance=2,footprint=np.ones((3,3,2))):
    """
    This function takes a 3D numpy array of raw intensity values as input and outputs a segmented 3d numpy array with defined threshold value
    """
    #binary mask
    isz_mask = img_sum > thresh_intensity
    #detect peaks
    peaks_indices = peak_local_max(img_sum * isz_mask,min_distance=min_distance, footprint=np.ones(footprint))
    peaks_mask = np.zeros_like(img_sum, dtype=bool)
    peaks_mask[peaks_indices[:,0],peaks_indices[:,1],peaks_indices[:,2]] = True
    #peaks_mask = peak_local_max(isz_mask*img_sum, min_distance, footprint,indices=False)
    #segmentation
    isz_markers = label(peaks_mask)
    isz_seg = watershed(-img_sum, isz_markers, mask=isz_mask)
    return isz_seg
   
def read_image(raw_fn, mosaic_num):
    """
    Read the imaging data from czi file into 4D numpy array with the shape (Channels, Z, Y, X)
    """
    czi = aplc.CziFile(raw_fn)
    img,shp = czi.read_image(M=int(mosaic_num))
    img = np.squeeze(img)
    return img
    
def flatten_channels(img):
    """
    flattens the channel dimension for each tile in the 4D image array.
    """
    img_sum = []
    for z in range(img.shape[1]):
        tile = img[:,z,:,:] # extracts a 3D slice from the 4D array
        tile_sum = np.sum(tile, axis=0) #sum the pixels for all channels
        img_sum.append(tile_sum)
    img_sum = np.dstack(img_sum)
    return img_sum

def measure_regionprops_3d(seg, raw=None):
    if isinstance(raw, type(None)):
        raw = np.zeros(seg.shape)
    sp_ = regionprops(seg, intensity_image = raw)
    properties=['label','centroid','area','max_intensity','mean_intensity',
                'min_intensity', 'bbox', 'major_axis_length','minor_axis_length']
    df = pd.DataFrame([])
    for p in properties:
        if p == 'minor_axis_length':
            minor_lengths = []
            for s in sp_:
                try:
                    ev = s.inertia_tensor_eigvals
                    minor_lengths.append(np.sqrt(10 * (-ev[0] + ev[1] + ev[2])))
                except Exception:
                    minor_lengths.append(np.nan)
            df[p] = minor_lengths
        else:
            df[p] = [s[p] for s in sp_]
    for j in range(2):
        df['centroid-' + str(j)] = [r['centroid'][j] for i, r in df.iterrows()]
    for j in range(4):
        df['bbox-' + str(j)] = [r['bbox'][j] for i, r in df.iterrows()]
    return df


# =============================================================================
# Setup global variables
# =============================================================================


parser = argparse.ArgumentParser(description='The script loads the image, reshapes it, segments it, \
    extracts the properties table from the segmentation, save the segmentation, and save the properties table')

# Add arguments to the parser

parser.add_argument('-cfn', '--config_fn', dest ='config_fn', 
                    type = str, help = 'Path to configureaiton file') 
parser.add_argument('-segd', '--seg_fn', dest ='seg_fn', 
                    type = str, help = 'Filename to save segmentation array.') 
parser.add_argument('-propd', '--prop_fn', dest ='prop_fn', 
                    type = str, help = 'Filename to save properties table.')
parser.add_argument('-M', '--mosaic_num', dest ='m',
                    type = int, help = 'mosaic number')
parser.add_argument('-rw_fn', '--raw_fn', dest ='raw_fn',
                    type = str, help = 'Filname of the input sample')

# Parse the command-line arguments
args = parser.parse_args()

# Load config file
with open(args.config_fn, 'r') as f:
    config = yaml.safe_load(f)

# =============================================================================
# Main function
# =============================================================================


def main():
    """
    Script to segment the hiprfish images using watershed algorithm.

    Usage:
        python segmentation.py input_file output_file [--option1 OPTION1] [--option2 OPTION2]

    Example:
        python script_name.py data.csv processed_data.csv --option1 10 --option2 "value"
    """

    # Get the raw data paths 
       
    # Read image
    img = read_image(args.raw_fn, args.m)
    
    # Reshape image into 3D
    img_sum = flatten_channels(img)
    
    # Image segmentation
    mask_thresh = config['mask_thresh']
    min_dis = config['mini_distance']
    footprint = tuple(config['footprint'])
    isz_seg=segmented_array(img_sum,mask_thresh,min_dis,footprint)
    np.save(args.seg_fn,isz_seg)
    
    # Get and save the properties table 
    isz_props = measure_regionprops_3d(isz_seg, img_sum)
    isz_props.to_csv(args.prop_fn)

if __name__ == "__main__":
    main()
