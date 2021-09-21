#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:24:34 2019
@author: jmilli
"""

from pca_imagecube import pca_imagecube
# import adiUtilities as adi
import scipy.ndimage.interpolation
import numpy as np
from scipy.stats import trim_mean


def pca_julien(cube,angle_list,cube_ref=None,ncomp=1,radii=None,method='cor',\
               trim=0.25,verbose=False):
    """
    Input:
    cube : array_like
        The input cube, 2d (ADI data) or 3d array (IFS data), without fake
        companions.
    angle_list : array_like
        Vector with the parallactic angles.
    """
    
    if cube_ref is None:
        pca= pca_imagecube(cube,method=method,verbose=verbose,radii=radii,\
                        path='.',name='tmp',header=None) 
        residuals_cube = pca.compute_residuals(truncation=ncomp,save=False)
    else:
        pca= pca_imagecube(cube_ref,method=method,verbose=verbose,radii=radii,\
                        path='.',name='tmp',header=None) 
        residuals_cube = pca.compute_residuals(datacube=cube,truncation=ncomp,save=False)
    pca_image = derotateCollapse(residuals_cube,angle_list,trim=trim)
    return pca_image


def collapseCube(cube,trim=0.5,mask=None):
    """
    Collapse a cube using the median or a trimmed mean.
    Input:
        - cube: a cube of images
        - trim: The collaspe along the cube 3rd dimension is a trimmed mean. It expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
        - mask: a 3d binary array with True for values to be discarded when collapsing
                the cube
    Output:
        - collapsed image
    """
    if not mask is None:
        cube = np.copy(cube)
        print('Applying a binary mask: True values are discarded')
        cube[mask] = np.nan
    if trim==0.5:
        image = np.nanmedian(cube,axis=0)
    elif trim==0.:
        image =np.nanmean(cube,axis=0)
    elif trim>0. and trim < 0.5:
        image = trim_mean(cube, trim, axis=0)
    else:
        raise ValueError('The clean mean expects value between 0. (mean) \
        and 0.5 (median), received {0:4.2f}'.format(trim))
    return image

def subtractMedian(cube,trim=0.5,mask=None):
    """
    Subtracts the median or clean mean of a cube (first step of ADI)
    Input:
        - cube: a cube of images
        - trim: if 0.5 (default value), it subtracts the median. Otherwise expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
        - mask: a 3d binary array with True for bad values to be replaced by nan
                before computing the star signal
    Output:
        - residual cube after subtraction
    """
    return cube - collapseCube(cube,trim=trim,mask=mask)

def derotate(cube,parang,rotoff=0.,inplace=False):
    """
    Derotates a cube of images according ot the parallactic angle (parang).
    It uses scipy.ndimage.interpolation.rotate for derotation and therefore
    expects a cube with an odd number of rows and columns, centered on the central 
    pixel.
    Input:
        - cube: a cube of images
    Output:
        - derotated cube
    """
    nbFrames=cube.shape[0]
    if len(parang) != nbFrames:
        raise IndexError('The cube has {0:5d} frames and the parallactic angle array only {1:5d} elements'.format(nbFrames,len(parang)))
    if inplace:
        for i in range(nbFrames):
            cube[i,:,:] = scipy.ndimage.interpolation.rotate(\
                cube[i,:,:],-(parang[i]-rotoff),reshape=False,order=1,prefilter=False)
        return cube
    else:            
        cubeDerotated = np.ndarray(cube.shape)
        for i in range(nbFrames):
    #        cubeDerotated[i,:,:] = rot.frame_rotate(cube[i,:,:],-parang[i]+rotoff)
            cubeDerotated[i,:,:] = scipy.ndimage.interpolation.rotate(\
                cube[i,:,:],-(parang[i]-rotoff),reshape=False,order=1,prefilter=False)
        return cubeDerotated

def derotateCollapse(cube,parang,rotoff=0.,trim=0.5,inplace=False):
    """
    Derotates a cube of images according ot the parallactic angle (parang) 
    and collapses it using the median or clean mean of a cube (first step of ADI).
    It uses scipy.ndimage.interpolation.rotate for derotation and therefore
    expects a cube with an odd number of rows and columns, centered on the central 
    pixel.
    Input:
        - cube: a cube of images
        - trim: The collaspe along the cube 3rd dimension is a trimmed mean. It expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
    Output:
        - collapsed image after derotation
    """
    cubeDerotated = derotate(cube,parang,rotoff=rotoff,inplace=inplace)
    return collapseCube(cubeDerotated,trim=trim)



if __name__=='__main__':
    # example how to use this class

    from astropy.io import fits
    from pathlib import Path
    path_package = Path(__file__).parent
    path_data = path_package.joinpath('data')
    cube_science = fits.getdata(path_data.joinpath('cube_science_with_fake_planet.fits'))
    angles = np.loadtxt(path_data.joinpath('derotation_angles_science.txt'))

    pca_adi = pca_julien(cube_science,angles,method='cor',verbose=True,\
                            ncomp=30,radii=[8,20,200])

    # now we assume we have a second data cube of a reference star   
    cube_ref = fits.getdata(path_data.joinpath('cube_reference.fits'))
    pca_rdi = pca_julien(cube_science,angles,method='cor',verbose=True,ncomp=30,\
                            cube_ref=cube_ref,radii=[8,20,100,200])
 
