# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 08:14:10 2016
Modified on Mar 31 2018

@author: jmilli
"""
import os
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'data')

class pca(object):
    """
    Object pca. 
    The attributes of the object are:
        - matrix: the data matrix (after centring and normalisation if desired)
                Its shape is (Nobj,Katt) i.e. Nobj objects (lines) and Katt attributes
                (columns)
        - Nobj: the number of objects of the PCA
        - Katt: the number of attributes of the PCA
        - inertia_matrix: the inertia matrix (e.g. covariance matrix, correlation
                        matrix or sum of squares matrix, depending of the PCA
                        method chosen). It is a square matrix of shape is (Katt,Katt)
        - total_inertia: the trace of the inertia matrix (scalar value)
        - eigenval: eigenvalues of the inertia matrix, sorted by descending order.
                    This is a vector of size Katt.
        - eigenvect: eigenvectors of the inertia matrix. The shape of this matrix
                    is (Katt,Katt)
        - pc: the principal components. Its shape is (Nobj,Katt)
    """
    
    def __init__(self,data,method='covariance',verbose=True):
        """
        Constructor of the pca class. It checks the format of the input data matrix 
        and builds the covariance/correlation/sum of squares matrix. 
        Then it computes the principal components.
        Input:
            - data: a 2d numpy array of shape (Nobj,Katt) i.e with Katt columns and
                    Nobj rows. Katt is the number of attributes and Nobj is the number
                    of objects.
            - method: 'covariance' (default), 'correlation' or 'sum_squares'
            - verbose: True or False if you want some information printer on the screen
        """
        if data.ndim != 2:
            raise IndexError('The PCA input matrix is not 2D !')
        if np.nan in np.isfinite(data):
            raise TypeError('The PCA input matrix must not contain NaN values')
        self.Nobj,self.Katt = data.shape
        self.method=method
        if self.method == 'covariance':
            self.mean_att = np.nanmean(data,axis=0) #mean of each attribute, size Katt 
            self.matrix = data-self.mean_att
        elif self.method == 'correlation':
            self.mean_att = np.nanmean(data,axis=0) #mean of each attribute, size Katt 
            self.std_att = np.nanstd(data,axis=0) #std of each attribute, size Katt
            self.matrix = (data-self.mean_att)/self.std_att            
        elif self.method == 'sum_squares':
            self.matrix = data
        else:
            raise TypeError('method not understood. Must be covariance (default),\
                             correlation or sum_squares. Got {0:s}'.format(self.method))
        if verbose:
            print('Method chosen: {0:s}'.format(self.method))
            print('There are {0:d} attributes and {1:d} objects'.format(self.Katt,self.Nobj))
            print('Computing the {0:d}x{0:d} inertia matrix...'.format(self.Katt))
        self.inertia_matrix = np.dot(self.matrix.T,self.matrix)/self.Nobj      
        if verbose:
            print('Done')            
#        self.inertia_matrix = np.cov(self.data,rowvar=False,bias=True) # We could add weights here if needed.
            # bias = True means we divide by Nobj instead of Nobj-1
        self.total_inertia = np.trace(self.inertia_matrix)
        eigenval, eigenvect = linalg.eigh(self.inertia_matrix)    # eigenvalues and eigenvectors
        self.eigenval = eigenval[::-1] 
        self.eigenvect = eigenvect[::-1]
        if verbose:
            print('Computing the principal components')
        self.pc = self._compute_principal_components()
        if verbose:
            print('Done')
        
    def get_inertia_matrix(self):
        """
        Getter to access the inertia matrix
        """
        return self.inertia_matrix

    def get_eigenval(self,truncation=None):
        """
        Returns the array of eigenvalues, truncated if desired.
        Input:
            - trunaction: None (by default) to return all eigenvalues (vector of 
            shape Katt) or integer smaller or equal that Katt to return a 
            truncated vector of shape truncation.
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

#    def _compute_principal_components(self,truncation=None):
#        """
#        Compute and return the principal components, optionnally after truncating
#        a certain number of modes. The principal components is a matrix of shape 
#        (Nobj,Katt) if truncation is None, or of shape (Nobj,truncation) if truncation
#        is an integer.
#        Input:
#            - truncation: None (by default) to use all vectors or an integer smaller
#                        than Katt to truncate the number of modes.
#        """
#        if truncation is None:
#            pc = np.dot(self.matrix,self.eigenvect.T)
#        else:
#            if truncation<=self.Katt:
#                pc = np.dot(self.matrix,self.get_eigenvect(truncation=truncation).T)
#            else:
#                print("Can't truncate by more than {0:d}".format(self.Katt))
#                return            
#        return pc
    
    def _compute_principal_components(self):
        """
        Compute and return the principal components. The principal components 
        is a matrix of shape (Nobj,Katt)
        """
        return np.dot(self.matrix,self.eigenvect.T)

    def get_principal_components(self,truncation=None):
        """
        Returns the principal components, optionnally after truncating
        a certain number of modes. The principal components is a matrix of shape 
        (Nobj,Katt) if truncation is None, or of shape (Nobj,truncation) if truncation
        is an integer.
        Input:
            - truncation: None (by default) to use all vectors or an integer smaller
                        than Katt to truncate the number of modes.
        """
        if truncation is None:
            return self.pc
        else:
            if truncation>self.Katt:
                print("Can't truncate by more than {0:d}".format(self.Katt))
                return            
            else:
                return self.pc[:,0:truncation]

    def reconstruct_matrix(self):
        """
        Reconstruct the matrix by a fast matrix multiplication. This function should
        now be updated to include truncation as an argument. Returns a matrix of
        shape (Nobj,Katt) that should be equal to self.matrix
        """
        matrix_projected = np.dot(self.pc,self.eigenvect)
        return matrix_projected

    def reconstruct_data(self):
        """
        Reconstruct the data by calling self.reconstruct_matrix and in case the pca 
        method was covariance or correlation, adds again the average and multiply
        by the std. 
        Returns a matrix of shape (Nobj,Katt) that should be equal to data
        """
        matrix_projected = self.reconstruct_matrix()
        if self.method == 'covariance':
            return self.matrix+self.mean_att
        elif self.method == 'correlation':
            return (self.matrix+self.mean_att)*self.std_att
        elif self.method == 'sum_squares':
            return matrix_projected

    def project_matrix(self,matrix=None,truncation=None,method='covariance'):
        """
        Projects a matrix or the data themselves on the (truncated) eigenvectors.
        Input:
            - matrix: the maxtrix to project on the eigenvectors. Its shape must 
                    be (Ndata,Katt). By default, assumes you want to project the
                    data itself and in this case uses the variable matrix(Nobj,Katt).
                    of shape 
            - truncation: None (by default) to use all vectors or an integer smaller
                        than Katt to truncate the number of modes.
        Output:
            - the projected matrix of shape (Ndata,Katt) 
        """
        if matrix is None:
            matrix_projected  = np.dot(\
                        self.get_principal_components(truncation=truncation),\
#                        self._compute_principal_components(truncation=truncation),\
                        self.get_eigenvect(truncation=truncation))
        else:
            if matrix.ndim != 2:
                raise IndexError('The data matrix to project is not 2D')
            if matrix.shape[1] != self.Katt:            
                raise IndexError('The data matrix to project has a shape {0:d}x{1:d} and should be {0:d}x{2:d}'.format(matrix.shape[0],matrix.shape[1],self.Katt ))
            matrix_projected = np.dot(\
                        np.dot(matrix,self.get_eigenvect(truncation=truncation).T),\
                        self.get_eigenvect(truncation=truncation))
        return matrix_projected                

    def project_data(self,data=None,truncation=None,method='covariance'):
        """
        Function that calls the project_matrix function but preliminarily subtracts
        the mean or divide by the standard deviation to normalize the data.

        Not sure whether we want to leave the method as an argument. Maybe
        we want to force to use self.method instead. 
        Input:
            - data: the data to project on the eigenvectors. Its shape must 
                    be (Ndata,Katt). By default, assumes you want to project the
                    data itself and in this case uses the reduced matrix
                    of shape (Nobj,Katt)
            - truncation: None (by default) to use all vectors or an integer smaller
                        than Katt to truncate the number of modes.
        Output:
            - the projected data of shape (Ndata,Katt) 
        """
        if data is None:
            reconstructed_matrix = self.project_matrix(matrix=None,\
                                truncation=truncation,method=method)
#            return reconstructed_matrix
            if self.method == 'covariance':
                return reconstructed_matrix+self.mean_att
            elif self.method == 'correlation':
                return (reconstructed_matrix+self.mean_att)*self.std_att
            elif self.method == 'sum_squares':
                return reconstructed_matrix
        else:            
            if method == 'covariance':
                mean_att = np.nanmean(data,axis=0) #mean of each attribute, size Katt 
                matrix = data-mean_att
            elif method == 'correlation':
                mean_att = np.nanmean(data,axis=0) #mean of each attribute, size Katt 
                std_att = np.nanstd(data,axis=0) #std of each attribute, size Katt
                matrix = (data-mean_att)/std_att            
            elif method == 'sum_squares':
                matrix = data
            reconstructed_matrix = self.project_matrix(matrix=matrix,\
                                truncation=truncation,method=method)
            if method == 'covariance':
                return reconstructed_matrix+mean_att
            elif method == 'correlation':
                return (reconstructed_matrix+mean_att)*std_att
            elif method == 'sum_squares':
                return reconstructed_matrix                    

    def get_contribution(self): 
        """
        Computes the contribution of each eigenvector to the inertia
        Returns a matrix of shape (Nobj,Katt).
        """
        return self.pc**2/self.eigenval
        
    def plot_explained_inertia(self,modes=10,filename=None):
        """
        Plots the explained interia as a function of the mode number. This graph can be 
        used to decide how many modes to keep. 
        Input:
            - modes: the numer of modes to plot. By default 10
            - filename: None (default value) if you don't want to save the file.
                Otherwise the full path name to save the plot (e.g. "~/toto.pdf")
        """
        if modes>self.Katt:
            modes=self.Katt
        plt.close(0)
        plt.figure(0)
        plt.semilogy(self.get_eigenval(truncation=modes)/self.total_inertia)
        plt.grid()
        plt.ylabel('Part of inertia explained by the mode')
        plt.xlabel('Eigenmode number')
        if filename!=None:
            plt.savefig(filename)
        self.print_explained_inertia(modes=modes)
        return

    def print_explained_inertia(self,modes=10):
        """
        Prints the part of inertia explained by each mode, as well as the cumulative
        part of inertia. By default uses 10 modes
        Input:
            - modes: the numer of modes to describe
        """
        if modes>self.Katt:
            modes=self.Katt
        cumulative_explained_inertia = 0.
        print('Total inertia: {0:4.2e}'.format(self.total_inertia))
        print('Mode | Eigenvalue | Inertia explained (%) | Cumulative (%)')
        print('----------------------------------------------------------')
        for k in range(modes):
            eigenval = self.eigenval[k]
            explained_inertia = eigenval/self.total_inertia
            cumulative_explained_inertia += explained_inertia
            print('{0:3d}  |  {1:4.2e}  |        {2:6.3f}         |   {3:6.3f}'.format(\
                  k+1,eigenval,explained_inertia*100,cumulative_explained_inertia*100))
        return

        
#    def get_quality_of_representation(self):
#        """
#        Computes the quality of representation of each eigenvector 
#        Returns a matrix of shape (Nobj,Katt).
#        """
   
if __name__=='__main__':
    from astropy.io import ascii

    cn2_table = ascii.read(os.path.join(path,'interpolated_Cn2.txt'))
    cn2 = np.asarray(cn2_table.to_pandas())
    cn2=cn2[:,0:115]

    wind_speed_table = ascii.read(os.path.join(path,'interpolated_wind_speed.txt'))
    wind_speed = np.asarray(wind_speed_table.to_pandas())
    
    wind_dir_table = ascii.read(os.path.join(path,'interpolated_wind_direction.txt'))
    wind_dir = np.asarray(wind_dir_table.to_pandas())

    pca_obj = pca(cn2,method='correlation')

#    ctr = pca_obj.get_contribution()

    pca_obj.plot_explained_inertia()

    reconstructed_matrix = pca_obj.project_matrix(truncation=10)
#    reconstructed_matrix = pca_obj.project_matrix()
    print(np.median(reconstructed_matrix-pca_obj.matrix))

    reconstructed_data = pca_obj.project_data()
    print(np.nanstd(cn2-reconstructed_data))

#    projected_wind_speed = pca_obj.project_matrix(matrix=wind_speed,truncation=10)
#    print(projected_wind_speed.shape)
