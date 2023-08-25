
# Snakefile
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_07_29
# Last edited : 
# =============================================================================
"""
This pipeline was written to segment images using the watershed algorithm and computing region properties
"""
# =============================================================================
import pandas as pd
import os
import sys
import glob
import re
import aicspylibczi as aplc

# =============================================================================
# Functions
# =============================================================================

def get_mosaic_dict():
    dict_sn_m = {}
    raw_fmt = config['data_dir'] + '/' + config['raw_fmt']
    for sn in SAMPLE_NAMES:
        raw_fn = raw_fmt.format(sample_name=sn)
        czi = aplc.CziFile(raw_fn)
        m = czi.get_dims_shape()[0]['M'][1]
        dict_sn_m[sn] = list(range(m))
    return dict_sn_m

def expand_sn_m(fmt, dict_sn_m):
    fns = []
    for sn in SAMPLE_NAMES:
        ms = dict_sn_m[sn]
        for m in ms:
            fn = fmt.format(sample_name=sn, M=m)
            fns.append(fn)
    return fns


# =============================================================================
# Parameters
# =============================================================================

args = sys.argv
config_fn = args[args.index("--configfile") + 1]

SAMPLE_NAMES = config['sample_name']
dict_sn_m = get_mosaic_dict()
seg_fmt = config['output_dir'] + '/' + config['seg_fmt']
seg_fns = expand_sn_m(seg_fmt, dict_sn_m)

# =============================================================================
# Snake rules
# =============================================================================

rule all:
    input:
        seg_fns


rule segmentation:
    input:
        raw_filename = config['data_dir'] + '/' + config['raw_fmt']
    output:
        segmentation_filename = config['output_dir'] + '/' + config['seg_fmt'],
        properties_table = config['output_dir'] + '/' + config['props_fmt']
    params:
        config_fn = config_fn,
        script = config['scripts_dir'] + '/' + config['segmentation_script']
    shell:
        "python {params.script} "
        "-cfn {params.config_fn} "
        "-rw_fn {input.raw_filename} "
        "-M {wildcards.M} "
        "-segd {output.segmentation_filename} "
        "-propd {output.properties_table} "

        

