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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

class pca_imagecube(object):
    """
    Wrapper for the pca class that can handle cubes of images. 
    It flattens the images into 1D vectors referred to objects. The pixels of 
    the images are referred to as attributes. We therefore end up with a 2D 
    maxtrix of Nobj objects (rows) by Katt attributes (columns) that is the starting
    point for the PCA, implemented in another class called pca.
    """

    def __init__(self,datacube,method='cor',verbose=True,radii=None,\
                 path=None,name='PCA',header=None):
        """
        Constructor of the pca_imagecube class.
        Input:
            - datacube: a 3d numpy array
            - method: 'cov' for covariance (default option), 'cor' for correlation 
                or 'ssq' for sum of squares
            - verbose: True or False if you want some information printed on the terminal
            - radii: an array containing the radii in pixels of the annuli in 
                which the PCA must be calculated. For instance: radii=[10,100,200] means
                the PCA will be computed in 2 annuli defined by 10px-100px and 100px-200px.
                By default, assumes te whole image is used.
            - path: the path where results must be saved. If no path is specified, 
                    then results can't be saved. 
            - name: a string, all output files will start with this name. 
                A good practice here is to use the name of the target and or date 
                of observation. By default it is 'PCA'
            - header: the header to use for the output files.
        """
        if datacube.ndim != 3:
            raise IndexError('The input datacube must be a 3D  numpy array !')
        self.nframes,self.ny,self.nx = datacube.shape
        if radii is None:
            radii = [0,int(np.round(np.sqrt((self.ny//2)**2+(self.nx//2)**2)))]
        self.method=method
        self.set_path(path,verbose=verbose)
        self.set_prefix(name+'_'+method+'_'+'-'.join(['{0:d}'.format(i) for i in radii]))
        self.header=header
        distarr = distance_array((self.ny,self.nx),verbose=False)
        anglarr = angle_array((self.ny,self.nx),verbose=False)
        self.region_map = np.zeros((self.ny,self.nx),dtype=int)
        self.nb_annuli = len(radii)-1
        self.Nobj_array = np.ndarray(self.nb_annuli)
        self.pca_array = []
        self.x_indices_array = []
        self.y_indices_array = []
        if verbose:
            print('There are {0:d} frames and {1:d} regions.'.format(self.nframes,self.nb_annuli))
        for i in range(self.nb_annuli):
            y_indices,x_indices = np.where(np.logical_and(distarr>=radii[i],distarr<radii[i+1]))
            self.y_indices_array.append(y_indices)
            self.x_indices_array.append(x_indices)
            self.Nobj_array[i] = len(y_indices)
            self.region_map[y_indices,x_indices] = i+1
            data = datacube[:,y_indices,x_indices].T # Transpose is used to get a shape (Nobj x Katt) where Katt is the number of frames of the datacube
            self.pca_array.append(pca.pca(data,method=method,verbose=verbose))
            if verbose:
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
    
    def reconstruct_data(self,datacube=None,truncation=None):
        """
        Reconstructs the datacube, by applying a (truncated) pca. If datacube is
        not specified, it assumes the we want to reconstruct the data used to 
        build the eigenmodes. If datacube is specified, it will project this 
        datacube on the eigenmodes. 
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
            for i in range(self.nb_annuli):
                reconstructed_data = self.pca_array[i].project_data(data=datacube,\
                                    truncation=truncation,method=self.method)
                reconstructed_datacube[:,self.y_indices_array[i],self.x_indices_array[i]] = \
                    reconstructed_data.T
        else:
            if datacube.ndim==2:
                datacube = datacube.reshape((1,datacube.shape[0].datacube.shape[1]))
            elif datacube.ndim !=3:
                raise IndexError('The input datacube must be a 3D or 2D numpy array !')
            if datacube.shape[1] != self.ny or datacube.shape[2] != self.nx:
                raise IndexError('The input frames must be {0:d} x {1:d} px !'.format(self.nx,self.ny))
            reconstructed_datacube = np.zeros_like(datacube)*np.nan
            for i in range(self.nb_annuli):
                data_flattened = datacube[:,self.y_indices_array[i],self.x_indices_array[i]].T
                reconstructed_data = self.pca_array[i].project_data(data=data_flattened,\
                                    truncation=truncation,method=self.method)
                reconstructed_datacube[:,self.y_indices_array[i],self.x_indices_array[i]] = \
                    reconstructed_data.T
                if reconstructed_datacube.shape[0]==1: # to return a 2D frame instead of a cube
                    reconstructed_datacube.reshape((self.ny,self.nx))
        return reconstructed_datacube
            
    
                    
    def get_principal_components(self,truncation=None,save=False):
        """
        Returns the principal components in the form of a data cube of K frames, 
        K being the truncation number if specified, or the number of inital 
        frames in the cube otherwise. Optionnaly we can save those frames
        as a cube called prefix_pc.fits
        Input:
            - truncation: integer that should be smaller than the number of frames
                            to perform the truncation of the data. If none, use 
                            all the frames.          
            - save: boolean False by default to save the cube of principal 
                components.
        Output:
            - the cube of principal components
        """
        if truncation==None:
            pc_cube = np.zeros((self.nframes,self.ny,self.nx))*np.nan
        else:
            pc_cube = np.zeros((truncation,self.ny,self.nx))*np.nan
        for i in range(self.nb_annuli):
                pc_tmp = self.pca_array[i].get_principal_components(truncation=truncation)
                pc_cube[:,self.y_indices_array[i],self.x_indices_array[i]] = pc_tmp.T
        if save:
            fits.writeto(os.path.join(self.path,self.prefix+'_pc.fits'),\
                pc_cube,header=self.header,overwrite=True)
        return pc_cube

    def get_quality_of_representation(self,save=False):
        """
        Returns the quality of representation of each pixel by 
        each eigenvector. The quality of representation has 
        the form of a data cube of K frames, K being the number of inital 
        frames in the cube. Optionnaly we can save those frames
        as a cube called prefix_co2.fits
        Input:
            - save: boolean (False by default) to save the cube of quality of representation
        Output:
            - the cube of of quality of representations
        """
        co2_cube = np.zeros((self.nframes,self.ny,self.nx))*np.nan
        for i in range(self.nb_annuli):
                co2_tmp = self.pca_array[i].get_quality_of_representation()
                co2_cube[:,self.y_indices_array[i],self.x_indices_array[i]] = co2_tmp.T
        if save:
            fits.writeto(os.path.join(self.path,self.prefix+'_co2.fits'),\
                co2_cube,header=self.header,overwrite=True)
        return co2_cube

    def get_contribution(self,save=False):
        """
        Returns the relative contribution of each pixel to the formation
        of each eigenvector. The relatvei contribution maxtris has 
        the form of a data cube of K frames, K being the number of inital 
        frames in the cube. Optionnaly we can save those frames
        as a cube called prefix_ctr.fits
        Input:
            - save: boolean False by default to save the cube of relative contributions
        Output:
            - the cube of of relative contributions
        """
        ctr_cube = np.zeros((self.nframes,self.ny,self.nx))*np.nan
        for i in range(self.nb_annuli):
                ctr_tmp = self.pca_array[i].get_contribution()
                ctr_cube[:,self.y_indices_array[i],self.x_indices_array[i]] = ctr_tmp.T
        if save:
            fits.writeto(os.path.join(self.path,self.prefix+'_ctr.fits'),\
                ctr_cube,header=self.header,overwrite=True)
        return ctr_cube

    def get_total_contribution(self,save=False):
        """
        Returns the total contribution of each pixel to the total inertia of the
        data, in the form of a single image. 
        Optionnaly we can save this frame as prefix_total_ctr.fits
        Input:
            - save: boolean False by default to save the cube of relative contributions
        Output:
            - the image of of total contributions
        """
        ctr = np.zeros((self.ny,self.nx))*np.nan
        for i in range(self.nb_annuli):
                ctr_tmp = self.pca_array[i].get_total_contribution()
                ctr[self.y_indices_array[i],self.x_indices_array[i]] = ctr_tmp.T
        if save:
            fits.writeto(os.path.join(self.path,self.prefix+'_total_ctr.fits'),\
                ctr,header=self.header,overwrite=True)
        return ctr

    def save_statistics(self,truncation=None,save_map=True,save_ctr=True,\
                        save_co2=True,save_pc=True):
        """
        Plots the percentage of inertia explained by each mode, for each 
        working region.
        It also optionnally saves the regions map by calling the write_map() function, 
        the principal components (cube), the quality of representation (cube),
        the contribution (cube), the total contribution (image).
        Input:
            - truncation: integer that should be smaller than the number of frames
                            to perform the truncation of the data. If none, use 
                            all the frames.          
            - save_map: boolean True by default to save the region map (image)
            - save_ctr: boolean True by default to save the cube of relative 
                contribution and the total contribution (image).
            - save_co2: boolean True by default to save the quality of 
                representation
            - save_pc: boolean True by default to save the cube of principal 
                components.
        """
        if save_map:
            self.write_map()        
        if save_ctr:
            dummy = self.get_total_contribution(save=True)
            dummy = self.get_contribution(save=True)
        if save_co2:
            dummy = self.get_quality_of_representation(save=True)  
        if save_pc:
            dummy = self.get_principal_components(truncation=truncation,save=True)
        plt.close(1)
        fig = plt.figure(1, figsize=(12,12))
        plt.rcParams.update({'font.size':14})        
        gs = gridspec.GridSpec(2,1, height_ratios=[1,1],)
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.98, wspace=0.2, hspace=0.3)        
        ax1 = plt.subplot(gs[0,0]) # Area for the explained inertia in linear x
        ax2 = plt.subplot(gs[1,0]) # Area for the explained inertia in log x
        for i in range(self.nb_annuli):
            if truncation is None:
                xarray = np.arange(1,self.pca_array[i].Katt+1)
            else:
                xarray = np.arange(1,truncation+1)
            yarray = self.pca_array[i].get_eigenval(truncation=truncation)/self.pca_array[i].total_inertia
            ax1.plot(xarray,yarray,label='Region {0:d} ({1:d}px, total inertia {2:.0f})'.format(i+1,self.pca_array[i].Nobj,self.pca_array[i].get_total_inertia()))
            ax2.plot(xarray,yarray,label='Region {0:d}'.format(i+1))
        for ax in [ax1,ax2]:
            ax.grid()
            ax.set_ylabel('Part of inertia explained by the mode')
            ax.set_xlabel('Eigenmode number')
            ax.set_yscale("log")
            ax.legend(frameon=False,loc='upper right')
            ax.set_ylim(top=1)
            ax.set_xlim(left=1)
        ax2.set_xscale("log")
        fig.savefig(os.path.join(self.path,self.prefix+'_statistics.pdf'))
        return

if __name__=='__main__':

    import adiUtilities as adi
    import vip_hci as vip
    ds9=vip.fits.ds9()

    path_root = '/Users/jmilli/Documents/RDI'
    path_out = os.path.join(path_root,'test_pipeline')
    cubeA = fits.getdata(os.path.join(path_root,'hip21986A_fc.fits'))
    headerA = fits.getheader(os.path.join(path_root,'hip21986A_fc.fits'))
    
    method= 'cor' #'cov' # 'cor' #'ssq'    

    pca_cube = pca_imagecube(cubeA,method=method,verbose=True,radii=[0,20,200],\
                 path=path_out,name='hip21986A',header=headerA)

    pca_cube.save_statistics(truncation=10)

    reconstructed_datacube = pca_cube.reconstruct_data(truncation=None)
    reconstructed_datacube2 = pca_cube.reconstruct_data(datacube=cubeA,truncation=None)
    ds9.display(reconstructed_datacube-reconstructed_datacube2)
    
#    ds9.display(cubeA,reconstructed_datacube,cubeA-reconstructed_datacube)

#    pca_cube.write_map()
#    pc_cube = pca_cube.get_principal_components(save=True)
#    ctr_cube = pca_cube.get_contribution(save=True)
#    ctr = pca_cube.get_total_contribution(save=True)
#    co2 = pca_cube.get_quality_of_representation(save=True)

#    
##    fits.writeto(os.path.join(path_out,'hip21986A_fc_reconstructed.fits'),reconstructed_datacube,overwrite=True)
#
#    ds9.display(cubeA-reconstructed_datacube)
#    residuals = cubeA-reconstructed_datacube
#    median_cube = np.median(cubeA,axis=0)
#    median_residuals = np.nanmedian(residuals,axis=0)
#    ds9.display(median_cube,median_residuals)
# 
#    cubeSubtracted = cubeA-reconstructed_datacube
#    
#    fits.writeto(os.path.join(path_out,'hip21986A_fc_subtracted.fits'),cubeSubtracted,overwrite=True)
    parangle = np.asarray(pd.read_csv(os.path.join(path_root,'hip21986A_fc_angs.txt'),header=None)[0])
#    pca_image = adi.derotateCollapse(cubeSubtracted,-parangle,rotoff=0.,trim=0.4)
#    fits.writeto(os.path.join(path_out,'hip21986A_pca_adi.fits'),pca_image,overwrite=True)
##
    cubeB = fits.getdata(os.path.join(path_root,'hip21986B_nfc.fits'))
    reconstructed_datacubeB_fromA = pca_cube.reconstruct_data(datacube=cubeB,truncation=None)
    ds9.display(cubeB,cubeB-reconstructed_datacubeB_fromA)


    pca_cube = pca_imagecube(cubeB,method='cov',verbose=True,radii=[0,20,200],\
             path=path_out,name='hip21986B',header=headerA)#,radii=[6,20,116]
    reconstructed_datacubeA_fromB = pca_cube.reconstruct_data(datacube=cubeA,truncation=None)
    pca_image = adi.derotateCollapse(cubeA-reconstructed_datacubeA_fromB,parangle,rotoff=0.,trim=0.4)
    fits.writeto(os.path.join(path_out,'hip21986A_pca_rdi.fits'),pca_image,overwrite=True)
#    ds9.display(
            