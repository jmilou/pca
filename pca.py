# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 08:14:10 2016

@author: jmilli
"""
import os
import numpy as np
from astropy.io import ascii
from numpy import linalg
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'data')

class pca(object):
    """
    Object pca. 
    The attributes of the object are:
        - matrix: the data matrix (after centring and normalisation if desired)
                Its shape is (Nobj,Katt) i.e. Nobj lines and Katt attributes
        - inertia_matrix: the inertia matrix, also called covariance matrix.
                        Its shape is (Katt,Katt)
        - total_inertia: the trace of the inertia matrix (scalar value)
        - eigenval: eigenvalues of the inertia matrix, sorted by descending order.
                    This is 1D array of size Katt
        - eigenvect: eigenvectors of the inertia matrix. The shape of this matrix
                    is (Katt,Katt)
        - pc: the principal components. Its shape is (Nobj,Katt)
    pca_core,path_out,I_psi,science_I_psi,covariance=covariance,nomeansub=nomeansub,
    visu=visu,ctr=ctr,qlt=qlt,eigenvect=eigenvect,I_psi_reconstructed=I_psi_reconstructed,
    extra_obj=extra_obj,extra_name=extra_name,science_I_psi_reconstructed=science_I_psi_reconstructed,
    tot_ctr=tot_ctr,ploteps=ploteps,tail=tail,verbose=verbose,K_klip=K_klip,
    automatic_truncation=automatic_truncation,adi=adi,all=all
    """
    
    def __init__(self,data,method='covariance',verbose=True):
        """
        Constructor of the pca class. It checks the format of the input data matrix 
        and builds the covariance matrix. Then it computes the principal components.
        Input:
            - data: a 2d numpy array of shape (Nobj,Katt) i.e with Katt columns and
                    Nobj rows. Katt is the number of attributes and Nobj is the number
                    of objects.
            - method: 'covariance' (default), 'correlation' or 'sum_squares'
            - verbose: True or False if you want some information printer on the screen
        """
        if data.ndim != 2:
            raise Exception('The PCA input matrix is not 2D !')
        if np.nan in np.isfinite(data):
            raise TypeError('The PCA input matrix must not contain NaN values')
        self.Nobj,self.Katt = data.shape
        if method == 'covariance':
            mean_att = np.mean(data,axis=0) #mean of each attribute, size Katt 
            self.matrix = data-mean_att
        elif method == 'correlation':
            mean_att = np.mean(data,axis=0) #mean of each attribute, size Katt 
            std_att = np.std(data,axis=0) #std of each attribute, size Katt
            self.matrix = (data-mean_att)/std_att            
        elif method == 'sum_squares':
            self.matrix = data
        else:
            raise TypeError('method not understood. Must be covariance (default),\
                             correlation or sum_squares. Got {0:s}'.format(method))
        if verbose:
            print('Method chosen: {0:s}'.format(method))
            print('There are {0:d} attributes and {1:d} objects'.format(self.Katt,self.Nobj))
            print('Computing the {0:d}x{0:d} inertia matrix'.format(self.Katt))
        self.inertia_matrix = np.dot(self.matrix.T,self.matrix)/self.Nobj      
#        self.inertia_matrix = np.cov(self.data,rowvar=False,bias=True) # We could add weights here if needed.
            # bias = True means we divide by Nobj instead of Nobj-1
        self.total_inertia = np.trace(self.inertia_matrix)
        if verbose:
            print('Total inertia: {0:e}'.format(self.total_inertia))
        eigenval, eigenvect = linalg.eigh(self.inertia_matrix)    # eigenvalues and eigenvectors
        self.eigenval = eigenval[::-1] 
        self.eigenvect = eigenvect[::-1]
        self.pc = self.compute_principal_components()
        
    def get_inertia_matrix(self):
        """
        Getter to access the inertia matrix
        """
        return self.inertia_matrix

    def get_eigenval(self,truncation=None):
        """
        Returns the eigenvalues, truncated if desired
        """
        if truncation is None:
            return self.eigenval
        else:
            if truncation<=self.Katt:
                return self.eigenval[0:truncation]
            else:
                print("Can't truncate by more than {0:d}".format(self.Katt))

    def get_eigenvect(self,truncation=None):
        """
        Returns the eigenvectors, truncated if desired
        Input:
            - trunaction: None (by default) to return all eigenvectors (matrix of 
            shape (Katt,Katt)) or integer smaller or equal that Katt to return a 
            truncated matrix of shape (trnucation,Katt).
        """
        if truncation is None:
            return self.eigenvect
        else:
            if truncation<=self.Katt:
                return self.eigenvect[0:truncation,:]
            else:
                print("Can't truncate by more than {0:d}".format(self.Katt))
                return

    def get_total_inertia(self):
        """
        Getter to access the total inertia (scalar value)
        """
        return self.total_inertia

    def compute_principal_components(self,truncation=None):
        """
        Compute and return the principal components, optionnally after truncating
        a certain number of modes. The principal components is a matrix of shape 
        (Nobj,Katt) if truncation is None, or of shape (Nobj,truncation) if truncation
        is an integer.
        Input:
            truncation: None (by default) to use all vectors or an integer smaller
            than Katt to truncate the number of modes.
        """
        if truncation is None:
            pc = np.dot(self.matrix,self.eigenvect.T)
        else:
            if truncation<=self.Katt:
                pc = np.dot(self.matrix,self.get_eigenvect(truncation=truncation).T)
            else:
                print("Can't truncate by more than {0:d}".format(self.Katt))
                return            
        return pc

    def reconstruct_data(self):
        """
        Reconstruct the data by a fast matrix multiplication. This function should
        now be updated to include truncation as an argument. Returns a matrix of
        shape (Nobj,Katt) that should be equal to self.matix
        """
        data_projected = np.dot(self.pc,self.eigenvect)
        return data_projected

    def get_contribution(self): 
        """
        Computes the contribution of each eigenvector to the inertia
        Returns a matrix of shape (Nobj,Katt).
        """
        return self.pc**2/self.eigenval
        
#    def quality(self):
#        """
#        Computes the quality of representation of each eigenvector 
#        Returns a matrix of shape (Nobj,Katt).
#        """
        
    
if __name__=='__main__':
    cn2_table = ascii.read(os.path.join(path,'interpolated_Cn2.txt'))
    cn2 = np.asarray(cn2_table.to_pandas())
    wind_speed = np.asarray(ascii.read(os.path.join(path,'interpolated_wind_speed.txt')))
    wind_dir = np.asarray(ascii.read(os.path.join(path,'interpolated_wind_direction.txt')))
    pca_obj = pca(cn2)
    c = pca_obj.get_inertia_matrix()
#    print(c.shape)
    pc = pca_obj.compute_principal_components()
#    print(pc.shape)
#    reconstructed_data = pca_obj.reconstruct_data()
#    print(reconstructed_data.shape)
#    print(np.median(reconstructed_data-pca_obj.matrix))
#    ctr = pca_obj.get_contribution()
#    print(ctr.shape)
    
    plt.semilogy(pca_obj.get_eigenval()/pca_obj.total_inertia)
    plt.axis([0,20,1e-3,1])
    plt.grid()
    plt.ylabel('Inertia')
    plt.xlabel('Mode number')
    