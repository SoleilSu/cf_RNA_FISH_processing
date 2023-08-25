# Cell free RNA FISH image processing

This snakemake pipeline is used to segment spots from 3d images taken on a confocal i880 microscope in spectral (lambda) mode on a single laser channel.

The pipeline loads one mosaic "tile" at a time consisting of a full z-stack and outputs each mosaic as a separate npy array with features as a csv file. 

## Package requirements:

Install using conda


```
conda env create -f cf_RNA_FISH.yaml
```

## Usage 

Set up your config.yaml file. Make sure to include the sample names you wish to run. For output naming, Wildcards must be 'sample_name' and 'M'.

```
snakemake -j NUM_CORES --configfile CONFIG_FILENAME -p 
```
