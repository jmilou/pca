#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:24:34 2019

@author: jmilli
"""

from pca_imagecube import pca_imagecube
import adiUtilities as adi

def pca_julien(cube,angle_list,cube_ref=None,ncomp=1,radii=None,method='cor',\
               trim=0.25,verbose=False):
    """
    Input:
    cube : array_like
        The input cube, 2d (ADI data) or 3d array (IFS data), without fake
        companions.
    angle_list : array_like
        Vector with the parallactic angles.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    """
    
    if cube_ref is None:
        pca= pca_imagecube(cube,method=method,verbose=verbose,radii=radii,\
                        path='.',name='tmp',header=None) 
        residuals_cube = pca.compute_residuals(truncation=ncomp,save=False)
    else:
        pca= pca_imagecube(cube_ref,method=method,verbose=verbose,radii=radii,\
                        path='.',name='tmp',header=None) 
        residuals_cube = pca.compute_residuals(datacube=cube,truncation=ncomp,save=False)
    pca_image = adi.derotateCollapse(residuals_cube,angle_list,trim=trim)
    return pca_image


#vip.pca.pca(
#    ['cube', 'angle_list', 'cube_ref=None', 'scale_list=None', 'ncomp=1', \
#     'ncomp2=1', "svd_mode='lapack'", 'scaling=None', "adimsdi='double'", \
#     'mask_center_px=None', 'source_xy=None', 'delta_rot=1', 'fwhm=4', \
#     "imlib='opencv'", "interpolation='lanczos4'", "collapse='median'", \
#     'check_mem=True', 'crop_ifs=True', 'nproc=1', 'full_output=False', \
#     'verbose=True', 'debug=False'])
