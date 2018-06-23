#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:51:03 2018

@author: jmilli
"""
import os
from image_tools import distance_array,angle_array
import numpy as np
from astropy.io import fits
import pca as pca

class pca_imagecube(object):
    """
    Wrapper for the pca class that can handle cubes of images. 
    It converts images in 1D vector used as attributes and uses the 3rd dimension
    of the cube as the objects.
    """

    def __init__(self,datacube,method='covariance',verbose=True,radii=None,\
                 path=None,prefix='PCA',header=None):
        """
        Constructor of the pca_imagecube class.
        Input:
            - datacube: a 3d numpy array
            - method: 'covariance' (default), 'correlation' or 'sum_squares'
            - verbose: True or False if you want some information printer on the screen
            - radii: an array containing the radii in pixels of the regions in 
                which the pca must be calculated. For instance: radii=[10,100,200] means
                the pca will be computed in 2 annuli defined by 10px-100px and 100px-200px 
            - path: the path where results must be saved. If no path is specified, 
                    then results can't be saved. 
            - prefix: a string, all output files will start with this prefix. By default it is
                    'PCA'
            - header: the header to use for the output files.
        """
        if datacube.ndim != 3:
            raise IndexError('The input datacube must be a 3D  numpy array !')
        self.nframes,self.ny,self.nx = datacube.shape
        self.method=method
        self.set_path(path,verbose=verbose)
        self.set_prefix(prefix)
        self.header=header
        distarr = distance_array((self.ny,self.nx),verbose=False)
        anglarr = angle_array((self.ny,self.nx),verbose=False)
        self.region_map = np.zeros((self.ny,self.nx),dtype=int)
        self.nb_radii = len(radii)-1
        self.Nobj_array = np.ndarray(self.nb_radii)
        self.pca_array = []
        self.x_indices_array = []
        self.y_indices_array = []
        if verbose:
            print('There are {0:d} frames and {1:d} regions.'.format(self.nframes,self.nb_radii))
        for i in range(self.nb_radii):
            y_indices,x_indices = np.where(np.logical_and(distarr>=radii[i],distarr<radii[i+1]))
            self.y_indices_array.append(y_indices)
            self.x_indices_array.append(x_indices)
            self.Nobj_array[i] = len(y_indices)
            self.region_map[y_indices,x_indices] = i+1
            data = datacube[:,y_indices,x_indices].T#.reshape(self.nframes,self.Nobj_array[i])
            self.pca_array.append(pca.pca(data,method=method,verbose=verbose))
            self.pca_array[i].print_explained_inertia(modes=5)

    def set_path(self,path,verbose=False):
        """
        Sets the path where the data is to be saved
        Input: 
            - path: the path where files are to be written
        """
        if os.path.exists(path) == False:
            os.mkdir(path)
            if verbose:
                print("The directory {0:s} did not exist and was created".format(path))
        self.path = path
        return
    
    def set_prefix(self,prefix,verbose=False):
        """
        Sets the prefix (names will start with that prefix)
        Input: 
            - prefix: a string 
        """
        if verbose:
            print("The prefix for the files was set to {0:s}".format(prefix))
        self.prefix = prefix
        return
    
    def write_map(self):
        """
        Writes the map of the regions in which the PCA is done.
        """
        if self.path is not None:
            fits.writeto(os.path.join(self.path,self.prefix+'_regions_map.fits'),\
                     self.region_map,header=self.header,overwrite=True)
        else:
            print('The path was not specified. Set a path before writing results to disk.')
        return
    
    def apply_pca(self,datacube=None,truncation=None):
        """
        Apply the pca to either truncate the data if data is None, or to project
        a different datacube
        Input:
            - datacube: the data to project. If None, assumes we use the cube itself
                    otherwise, expects a cube of data with the same number of pixels.
            - truncation: integer that should be smaller than the number of frames
                            to perform the truncation of the data. If none, use 
                            all the frames.
        Output:
            - reconstructed_datacube:
        """
        if datacube is None:
            reconstructed_datacube = np.zeros((self.nframes,self.ny,self.nx))*np.nan
            for i in range(self.nb_radii):
                reconstructed_data = self.pca_array[i].project_data(data=datacube,\
                                    truncation=truncation,method=self.method)
                reconstructed_datacube[:,self.y_indices_array[i],self.x_indices_array[i]] = \
                    reconstructed_data.T
            return reconstructed_datacube
        
#    def subtract_reconstructed_cube(self,datacube=None,truncation=None):
#        """
#        Input:
#            - datacube: the data to project. If None, assumes we use the cube itself
#                    otherwise, expects a cube of data with the same number of pixels.
#            - truncation: integer that should be smaller than the number of frames
#                            to perform the truncation of the data. If none, use 
#                            all the frames.
#        Output:
#            - subtracted_datacube:
#        """
#        if datacube is None:
#            return self.data
        
            
if __name__=='__main__':
    import pandas as pd
    import adiUtilities as adi
#    import pyds9
#    ds9=pyds9.DS9()
    import vip_hci as vip
    ds9=vip.fits.ds9()

    path_root = '/Users/jmilli/Documents/RDI'
    path_out = os.path.join(path_root,'test_pipeline')
    cubeA = fits.getdata(os.path.join(path_root,'hip21986A_fc.fits'))
    headerA = fits.getheader(os.path.join(path_root,'hip21986A_fc.fits'))
    cubeB = fits.getdata(os.path.join(path_root,'hip21986B_nfc.fits'))
    parangle = np.asarray(pd.read_csv(os.path.join(path_root,'hip21986A_fc_angs.txt'),header=None)[0])
    
    method= 'covariance' # 'correlation' #'sum_squares'    
    pca_cube = pca_imagecube(cubeA,method=method,verbose=True,radii=[20,100],\
                 path=path_out,prefix='PCA',header=headerA)
    pca_cube.write_map()
    reconstructed_datacube = pca_cube.apply_pca(truncation=10)
    fits.writeto(os.path.join(path_out,'hip21986A_fc_reconstructed.fits'),reconstructed_datacube,overwrite=True)

    residuals = cubeA-reconstructed_datacube
    median_cube = np.median(cubeA,axis=0)
    median_residuals = np.nanmedian(residuals,axis=0)
    ds9.display(median_residuals)
    ds9.display(median_cube)

    cubeSubtracted = cubeA-reconstructed_datacube
    fits.writeto(os.path.join(path_out,'hip21986A_fc_subtracted.fits'),cubeSubtracted,overwrite=True)
    pca_image = adi.derotateCollapse(cubeSubtracted,-parangle,rotoff=0.,trim=0.4)
    fits.writeto(os.path.join(path_out,'hip21986A_pca_adi.fits'),pca_image,overwrite=True)
