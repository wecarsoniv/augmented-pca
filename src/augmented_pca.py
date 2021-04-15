# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  apca.py
# Author:  Billy Carson
# Date written:  04-14-2021
# Last modified:  04-15-2021

"""
Description:  Augmented Principal Component Analysis (APCA) model definitions file. Class definitions for both
adversarial APCA (aAPCA) and supervised APCA (sAPCA).
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT MODULES
# ----------------------------------------------------------------------------------------------------------------------

# Import modules
from abc import ABC, abstractmethod
from warnings import warn
from numpy import mean, real, real_if_close, concatenate, identity
from numpy.linalg import inv, eig
from scipy.sparse import issparse


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Augmented PCA abstract base class
class _APCA(ABC):
    """
    Description
    -----------

    Parameters
    ----------

    Attributes
    ----------
    
    Methods
    -------
    """

    # Instantiation method of augmented PCA base class
    def __init__(self, n_components, mu, form, diag_const):
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        N/A
        """
        
        # Check for proper type/value and assign number of components attribute
        if n_components is not None:
            if n_components < 1:
                raise ValueError('Number of components must be an integer value greater than or equal to 1.')
            elif not isinstance(n_components, int):
                if (n_components - round(n_components)) < 1e-10:
                    warn(message=('Warning: Number of components must be an integer value greater than or equal to 1. ' +
                                  'Rounding to the nearest integer'))
                    n_components = round(n_components)
                else:
                    raise TypeError('Number of components must be an integer value greater than or equal to 1.')
        self.n_components = n_components
        
        # Check for proper type/value and assign adversary strength attribute
        if mu < 0.0:
            raise ValueError('Adversary strength must be a value greater than or equal to 0.')
        else:
            self.mu = mu
        
        # Check for proper type/value and assign APCA formulation attribute
        if not isinstance(form, str):
            raise TypeError('Form must be type string.')
        elif (form != 'local') & (form != 'encoded') & (form != 'joint'):
            raise ValueError('Form not recognized. Acceptable forms include \"local\", \"encoded\", and \"joint\".')
        else:
            self._form = form
        
        # Check for proper type/value and assign diagonal regularization constant attribute
        if (not isinstance(diag_const, float)) & (not isinstance(diag_const, int)):
            raise TypeError('Diagonal regularization constant must be numeric.')
        elif diag_const < 0.0:
            raise ValueError('Diagonal regularization constant must be a positive numeric value.')
        else:
            self.diag_const_ = diag_const
        
        # Assign attributes dependent on APCA form
        if form == 'joint':
            self.mu_star = mu + 1.0
        else:
            self.mu_star = None
        
        # Initialize all other attributes
        self.mean_X_ = None
        self.mean_Y_ = None
        self.mean_Z_ = None
        self.B_ = None
        self.W_ = None
        self.D_ = None
        self.V_ = None
        self.eigvals_ = None
        self.is_fitted_ = False

    # Get eigenvalues
    def get_eigvals(self):
        """
        Description
        -----------
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues

        Parameters
        ----------
        N/A

        Returns
        -------
        self.eigvals_ : numpy.ndarray; 1-dimensional array of sorted eigenvalues
        """

        # Check that encoding matrix attribute has been assigned
        if self.eigvals_ is None:
            raise AttributeError('Eigenvalues attribute not yet assigned.')

        # Return encoding matrix
        return self.eigvals_

    # Get primary data loadings
    def get_W(self):
        """
        Description
        -----------
        Returns primary data loadings

        Parameters
        ----------
        N/A

        Returns
        -------
        self.W_ : numpy.ndarray; 2-dimensional (p x k) primary data loadings matrix
        """

        # Check that primary data loadings attribute has been assigned
        if self.W_ is None:
            raise AttributeError('Primary data loadings attribute not yet assigned.')
        
        # Return primary data loadings
        return self.W_

    # Get encoding matrix
    def get_A(self):
        """
        Description
        -----------
        Returns encoding matrix

        Parameters
        ----------
        N/A

        Returns
        -------
        self.A_ : numpy.ndarray; 2-dimensional (d x p) encoding matrix
        """
        
        # Check that encoding matrix attribute has been assigned
        if self.A_ is None:
            if self._form == 'local':
                raise AttributeError('Local APCA object does not have an encoding matrix attribute.')
            else:
                raise AttributeError('Encoding matrix attribute not yet assigned.')
        
        # Return encoding matrix
        return self.A_

    # Fits augmented PCA model to data
    def fit(self, X, Y):
        """
        Description
        -----------
        Fits local adversary PCA model to data

        Parameters
        ----------
        X : numpy.ndarray; 2-dimensional (n x p) primary data matrix
        Y : numpy.ndarray; 2-dimensional (n x q) concomitant data matrix
        
        Returns
        -------
        self : _APCA instance; reference to abstract base class augmented PCA instance
        """
        
        # Create deep copies of primary and concomitant data matrices
        X_ = X.copy()
        Y_ = Y.copy()
        
        # Raise error for sparse matrices used as input
        if issparse(X_) | issparse(Y_):
            error_msg = '%(name) does not support sparse input.'
            raise TypeError(error_msg % {'name': self.__class__.__name__})
        
        # If number of components is not specified, define as smallest dimension of X
        # Check that specified number of components is not too large
        if self.n_components is None:
            self.n_components = min(X_.shape)
        else:
            if self.n_components > min(X_.shape):
                raise ValueError('Number of components is too large.')
        
        # Calculate means of data matrices
        self.mean_X_ = mean(X_, axis=0)
        self.mean_Y_ = mean(Y_, axis=0)
        
        # Mean-center primary data matrix and concomitant data matrix
        X_ -= self.mean_X_
        Y_ -= self.mean_Y_
        
        # Define Z as concatenation of primary data and concomitant data matrices
        if self._form == 'joint':
            Z_ = concatenate((X_, Y_), axis=1)
            self.mean_Z_ = mean(Z_, axis=0)
            Z_ -= self.mean_Z_
        else:
            Z_ = None
            self.mean_Z_ = None
        
        # Get data dimensions
        N, p = X_.shape
        _, q = Y_.shape
        
        # Get decomposition matrix
        if self._form == 'joint':
            self.B_ = self._get_B(M_=Z_, N_=Y_)
        else:
            self.B_ = self._get_B(M_=X_, N_=Y_)
        
        # Perform eigendecomposition
        eigvals, eigvecs = eig(self.B_)
        eigvals = real(eigvals)
        eigvecs = real(eigvecs)
        
        # Sort eigenvectors according to descending eigenvalues
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]
        
        # Assign eigenvalues to attribute
        self.eigvals_ = eigvals
        
        # Define loadings W, D, and V
        self.W_ = eigvecs[:p, :self.n_components]
        self.D_ = eigvecs[p + q:, :self.n_components]
        if self._form == 'joint':
            self.V_ = eigvecs[:p + q, :self.n_components]
        else:
            self.V_ = None
        
        # Generate encoding matrix
        if self._form == 'encoded':
            diag_reg = self.diag_const_ * identity(n=X_.shape[1])
            inv_XT_X = inv((X_.T @ X_) + diag_reg)
            diag_reg = self.diag_const_ * identity(n=self.W_.shape[1])
            A_1 = inv((self.W_.T @ self.W_) - (self.mu * self.D_.T @ self.D_) + diag_reg)
            A_2 = (self.W_.T @ X_.T) - (self.mu * self.D_.T @ Y_.T)
            self.A_ = A_1 @ A_2 @ X_ @ inv_XT_X
        elif self._form == 'joint':
            diag_reg = self.diag_const_ * identity(n=Z_.shape[1])
            inv_ZT_Z = inv((Z_.T @ Z_) + diag_reg)
            diag_reg = self.diag_const_ * identity(n=self.W_.shape[1])
            A_1 = inv((self.W_.T @ self.W_) - (self.mu * self.D_.T @ self.D_) + diag_reg)
            A_2 = ((self.W_.T @ X_.T) - (self.mu * self.D_.T @ Y_.T)) @ Z_ @ inv_ZT_Z
            self.A_ = A_1 @ A_2
        else:
            self.A_ = None
        
        # Update fitted variable
        self.is_fitted_ = True
        
        # Return reference to object
        return self

    # Transforms data into scores using augmented PCA formulation
    def transform(self, X, Y):
        """
        Description
        -----------
        Transforms data into scores using augmented PCA formulation

        Parameters
        ----------
        X : numpy.ndarray; 2-dimensional (n x p) primary data matrix
        Y : numpy.ndarray; 2-dimensional (n x q) concomitant data matrix

        Returns
        -------
        S : numpy.ndarray; 2-dimensional (n x k) scores matrix
        """

        # Check that model is fitted
        if not self.is_fitted_:
            error_msg = ('This %(name)\'s instance is not fitted yet. Call \'fit\' with '
                         'appropriate arguments before using this method.')
            raise NotFittedError(error_msg % {'name': self.__class__.__name__})

        # Create deep copy of primary data matrix
        X_ = X.copy()

        # Mean-center primary data
        X_ -= self.mean_X_

        # Generate scores
        if self._form == 'local':
            Y_ = Y.copy()
            S_1 = inv((self.W_.T @ self.W_) - (self.mu * self.D_.T @ self.D_))
            S_2 = (self.W_.T @ X_.T) - (self.mu * self.D_.T @ Y_.T)
            S = (S_1 @ S_2).T
        elif self._form == 'encoded':
            S = (self.A_ @ X_.T).T
        else:
            Y_ = Y.copy()
            Z_ = concatenate((X_, Y_), axis=1)
            Z_ -= self.mean_Z_
            S = (self.A_ @ Z_.T).T

        # Return scores
        return S

    # Fit adversarial PCA model to data and transform data into scores
    def fit_transform(self, X, Y):
        """
        Description
        -----------
        Fits local adversary PCA model to data and transforms data into scores

        Parameters
        ----------
        X : numpy.ndarray; 2-dimensional (n x p) primary data matrix
        Y : numpy.ndarray; 2-dimensional (n x q) concomitant data matrix

        Returns
        -------
        S : numpy.ndarray; 2-dimensional (n x k) scores matrix
        """

        # Fit local adversarial PCA instance
        self.fit(X=X, Y=Y)

        # Transforms data using local adversary formulation
        S = self.transform(X=X, Y=Y)

        # Return scores
        return S

    # Reconstruct primary and concomitant data
    def reconstruct(self, X, Y):
        """
        Description
        -----------
        Reconstruct primary or concomitant data.

        Parameters
        ----------
        X : numpy.ndarray; 2-dimensional (n x p) primary data matrix
        Y : numpy.ndarray; 2-dimensional (n x q) concomitant data matrix

        Returns
        -------
        X_recon : numpy.ndarray; 2-dimensional (n x p) reconstruction of primary data
        Y_recon : numpy.ndarray; 2-dimensional (n x q) reconstruction of concomitant data
        """

        # Get scores
        S = self.transform(X=X, Y=Y)

        # Reconstruct both primary and concomitant data
        X_recon = (S @ self.W_.T) + self.mean_X_
        Y_recon = (S @ self.D_.T) + self.mean_Y_

        # Return reconstructed primary and concomitant data
        return X_recon, Y_recon

    # Returns decomposition matrix given primary data matrix and concomitant data matrix
    @abstractmethod
    def _get_B(self):
        pass

    # Get augmenting data loadings
    @abstractmethod
    def get_D(self, aug_data_str):
        """
        Description
        -----------
        Returns augmenting data loadings

        Parameters
        ----------
        aug_data_str : str; augmenting data descriptor

        Returns
        -------
        self.D_ : numpy.ndarray; 2-dimensional (q x k) concomitant data loadings matrix
        """

        # Check that augmenting data loadings attribute has been assigned
        if self.D_ is None:
            raise AttributeError(aug_data_str.capitalize() + ' loadings attribute not yet assigned.')

        # Return augmenting data loadings
        return self.D_

    # Get combined primary and augmenting data loadings
    @abstractmethod
    def get_V(self, aug_data_str):
        """
        Description
        -----------
        Returns combined primary and augmenting data loadings

        Parameters
        ----------
        aug_data_str : str; augmenting data descriptor

        Returns
        -------
        self.V_ : numpy.ndarray; 2-dimensional ((p + q) x k) combined primary and augmenting data loadings matrix
        """

        # Check that combined primary and concomitant data loadings attribute has been assigned
        if self.V_ is None:
            raise AttributeError('Combined primary and ' + aug_data_str + ' data loadings attribute not yet assigned.')

        # Return combined primary and concomitant data loadings
        return self.V_


# Adversarial augmented PCA class
class aAPCA(_APCA):
    """
    Description
    -----------

    Parameters
    ----------

    Attributes
    ----------
    
    Methods
    -------
    """
    
    # Instantiation method of local augmented PCA base class
    def __init__(self, n_components=None, mu=1.0, form='joint', diag_const=1e-8):
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        """
        
        # Inherit from augmented PCA base class
        super().__init__(n_components=n_components, mu=mu, form=form, diag_const=diag_const)

    # Get concomitant data loadings
    def get_D(self, aug_data_str):
        """
        Description
        -----------
        Returns concomitant data loadings

        Parameters
        ----------
        N/A

        Returns
        -------
        self.D_ : numpy.ndarray; 2-dimensional (q x k) concomitant data loadings matrix
        """
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        super(_APCA, self).get_V(aug_data_str='concomitant')

    # Get combined primary and concomitant data loadings
    def get_V(self, aug_data_str):
        """
        Description
        -----------
        Returns combined primary and concomitant data loadings

        Parameters
        ----------
        aug_data_str : str; augmenting data descriptor

        Returns
        -------
        self.V_ : numpy.ndarray; 2-dimensional ((p + q) x k) combined primary and concomitant data loadings matrix
        """
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        super(_APCA, self).get_V(aug_data_str='concomitant')

    # Returns decomposition matrix given primary data matrix and concomitant data matrix
    def _get_B(self, M_, N_):
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        """
        
        # Define components of B
        B_11 = M_.T @ M_
        B_12 = M_.T @ N_
        B_21 = B_12.T
        if (self._form == 'encoded') | (self._form == 'joint'):
            diag_reg = self.diag_const_ * identity(M_.shape[1])
            inv_MT_M = inv((M_.T @ M_) + diag_reg)
            B_22 = B_12.T @ inv_MT_M @ B_12
        else:
            B_22 = N_.T @ N_
        
        # Form decomposition matrix
        B = concatenate((concatenate((B_11, -self.mu * B_12), axis=1),
                         concatenate((B_21, -self.mu * B_22), axis=1)), axis=0)
        
        # Return decomposition matrix
        return B


# Supervised PCA class
class sAPCA(_APCA):
    """
    Description
    -----------

    Parameters
    ----------

    Attributes
    ----------
    
    Methods
    -------
    """
    
    # Instantiation method of local augmented PCA base class
    def __init__(self, n_components=None, mu=1.0, form='joint', diag_const=1e-8):
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        """
    
        # Inherit from augmented PCA base class
        super().__init__(n_components=n_components, mu=mu, form=form, diag_const=diag_const)

    # Get concomitant data loadings
    def get_D(self, aug_data_str):
        """
        Description
        -----------
        Returns concomitant data loadings

        Parameters
        ----------
        N/A

        Returns
        -------
        self.D_ : numpy.ndarray; 2-dimensional (q x k) concomitant data loadings matrix
        """

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        super(_APCA, self).get_V(aug_data_str='concomitant')

    # Get combined primary and concomitant data loadings
    def get_V(self, aug_data_str):
        """
        Description
        -----------
        Returns combined primary and concomitant data loadings

        Parameters
        ----------
        aug_data_str : str; augmenting data descriptor

        Returns
        -------
        self.V_ : numpy.ndarray; 2-dimensional ((p + q) x k) combined primary and concomitant data loadings matrix
        """

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        super(_APCA, self).get_V(aug_data_str='concomitant')

    # Returns decomposition matrix given primary data matrix and concomitant data matrix
    def _get_B(self, M_, N_):
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        """
        
        # Define components of B
        B_11 = M_.T @ M_
        B_12 = M_.T @ N_
        B_21 = B_12.T
        if (self._form == 'encoded') | (self._form == 'joint'):
            diag_reg = self.diag_const_ * identity(M_.shape[1])
            inv_MT_M = inv((M_.T @ M_) + diag_reg)
            B_22 = B_12.T @ inv_MT_M @ B_12
        else:
            B_22 = N_.T @ N_
        
        # Form decomposition matrix
        B = concatenate((concatenate((B_11, self.mu * B_12), axis=1),
                         concatenate((B_21, self.mu * B_22), axis=1)), axis=0)
        
        # Return decomposition matrix
        return B


# Not fitted error class
class NotFittedError(ValueError, AttributeError):
    """
    Description
    -----------
    NotFittedError class inherits from both ValueError and AttributeError.

    Parameters
    ----------
    N/A

    Attributes
    ----------
    N/A
    """

