# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  models.py
# Author:  Billy Carson
# Date written:  04-14-2021
# Last modified:  11-08-2024

"""
Description:  AugmentedPCA model definitions file. Class definitions for adversarial AugmentedPCA (AAPCA), supervised
AugmentedPCA (SAPCA), and combined AugmentedPCA (CAPCA).
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
from abc import ABC, abstractmethod
from warnings import warn
from typing import Union
import numpy
from numpy import mean, real, real_if_close, concatenate, identity
from numpy.random import default_rng
from numpy.linalg import qr
from scipy.linalg import inv, solve, eig
from scipy.sparse import issparse


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# AugmentedPCA model abstract base class
class _APCA(ABC):
    r"""
    AugmentedPCA model abstract base class.
    
    Parameters
    -----------
    n_components : int
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float or tuple or list
        Augmenting objective strength(s).
    inference : str
        Model inference strategy.
    decomp : str
        Decomposition approach.
    pow_iter : int
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int
        Oversampling parameter for randomized approximation.
    diag_const : float
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    
    Attributes
    ----------
    n_components : int
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float or tuple or list
        Augmenting objective strength(s).
    pow_iter : int
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int
        Oversampling parameter for randomized approximation.
    diag_const : float
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    mean_X_ : numpy.ndarray
        1-dimensional (p,) mean array of primary data matrix.
    mean_Y_ : numpy.ndarray
        1-dimensional (q,) mean array of primary data matrix.
    mean_Z_ : numpy.ndarray
        1-dimensional (p + q,) mean array of combined primary and augmenting data matrices.
    B_ : numpy.ndarray
        2-dimensional decomposition matrix.
    W_ : numpy.ndarray
        2-dimensional primary data loadings matrix.
    D_ : numpy.ndarray or tuple / list of numpy.ndarrays
        2-dimensional augmenting data loadings matrix / matrices.
    A_ : numpy.ndarray
        2-dimensional encoding matrix; None if `inference` is set to 'local'.
    eigvals_ : numpy.ndarray
        1-dimensional array of sorted decomposition matrix eigenvalues.
    is_fitted_ : bool
        Indicates whether model has been fitted.
    
    Methods
    -------
    fit(X, Y)
        Fits AugmentedPCA model to data.
    transform(X, Y)
        Transforms data into scores using AugmentedPCA model formulation.
    fit_transform(X, Y)
        Fits AugumentedPCA model to data and transforms data into scores.
    reconstruct(X, Y)
        Reconstructs primary and augmenting data.
    get_components()
        Returns primary data loadings / components. Alias for get_W().
    get_W()
        Returns primary data loadings.
    get_D()
        Abstract method. Returns augmenting data loadings.
    get_A()
        Returns encoding matrix.
    get_eigvals()
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues.
    """
    
    # Instantiation method of AugmentedPCA model base class
    def __init__(
        self,
        n_components: int,
        mu: Union[float, tuple, list],
        inference: str,
        decomp: str,
        pow_iter: int,
        n_oversamp: int,
        diag_const: float,
        random_state: int):
        r"""
        Instantiation method of AugmentedPCA model base class.
        
        Parameters
        ----------
        n_components : int
            Number of components. If None then reduce to minimum dimension of primary data.
        mu : float or tuple or list
            Augmenting objective strength(s).
        pow_iter : int
            Number of power iterations to perform in randomized approximation.
        n_oversamp : int
            Oversampling parameter for randomized approximation.
        inference : str
            Model inference strategy.
        decomp : str
            Decomposition approach.
        diag_const : float
            Constant added to diagonals of matrix prior to inversion.
        random_state : int
            Model random state. Ignored if exact eigenvalue decomposition approach used.
        """
        
        # Check for proper type / value and assign number of components attribute
        if n_components is not None:
            if not isinstance(n_components, float) and not isinstance(n_components, int):
                raise TypeError('Number of components must be an integer value greater than or equal to 1.')
            else:
                if n_components < 1.0:
                    raise ValueError('Number of components must be an integer value greater than or equal to 1.')
                if not isinstance(n_components, int):
                    if (n_components - round(n_components)) < 1e-10:
                        warn(message=('Warning: Number of components must be an integer value greater than or equal ' +
                                      'to 1. Rounding to the nearest integer'))
                        n_components = round(n_components)
                    else:
                        raise TypeError('Number of components must be an integer value greater than or equal to 1.')
        self.n_components = n_components
        
        # Check for proper type / value and assign augmenting strength attribute
        if not isinstance(mu, float) and not isinstance(mu, int) and \
        not isinstance(mu, list) and not isinstance(mu, tuple):
            raise TypeError('Augmenting objective strength must be either an numeric value greater than or equal ' +
                            'to 0.0 or a tuple / list of numeric values.')
        if isinstance(mu, float) or isinstance(mu, int):
            if mu < 0.0:
                raise ValueError('Augmenting objective strength must be an numeric value greater than or equal to ' +
                                 '0.0.')
        elif isinstance(mu, tuple) or isinstance(mu, list):
            if isinstance(mu, list):
                mu = tuple(mu)
            for mu_ in list(mu):
                if mu_ < 0.0:
                    raise ValueError('Both augmenting objective strengths must be an numeric value greater than or ' +
                                     'equal to 0.0. Tuple or list contains negative value(s).')
            if len(mu) != 2:
                raise ValueError('Tuple or list must contain exactly two elements representing the supervised and ' +
                                 'adversarial augmenting strengths, respectively.')
        self.mu = mu
        
        # Check for proper type / value and assign approximate inference strategy attribute
        if not isinstance(inference, str):
            raise TypeError('Approximate inference strategy must be type string. Acceptable strategies include ' +
                            '\"local\", \"encoded\", and \"joint\".')
        elif inference != 'local' and inference != 'encoded' and inference != 'joint':
            raise ValueError(('Approximate inference strategy not recognized. Acceptable strategies include ' +
                              '\"local\", \"encoded\", and \"joint\"'))
        self._inference = inference
        
        # Check for proper type / value and assign decomposition approach attribute
        if not isinstance(decomp, str):
            raise TypeError('Decomposition approach parameter must be type string. Acceptable decomposition ' +
                            'approaches include \"exact\", and \"approx\".')
        elif decomp != 'exact' and decomp != 'approx':
            raise ValueError(('Decomposition approach parameter not recognized. Acceptable decomposition approaches' +
                              'include \"exact\", and \"approx\"'))
        self._decomp = decomp
        
        # Check for proper type / value and assign randomized AugmentedPCA power iterations attribute
        if not isinstance(pow_iter, float) and not isinstance(pow_iter, int):
            raise TypeError('Number of randomized AugmentedPCA power iterations must be an integer value greater ' + 
                            'than or equal to 0.')
        if pow_iter < 0.0:
            raise ValueError('Number of randomized AugmentedPCA power iterations must be a value greater than or ' + 
                             'equal to 0.')
        if not isinstance(pow_iter, int):
            if (pow_iter - round(pow_iter)) < 1e-10:
                warn(message=('Warning: Number of randomized AugmentedPCA power iterations must be an integer ' + 
                              'value greater than or equal to 0. Rounding to the nearest integer'))
                pow_iter = round(pow_iter)
            else:
                raise TypeError('Number of randomized AugmentedPCA power iterations must be an integer value ' + 
                                'greater than or equal to 0.')
        self.pow_iter = pow_iter
        
        # Check for proper type / value and assign randomized AugmentedPCA oversampling attribute
        if not isinstance(n_oversamp, float) and not isinstance(n_oversamp, int):
            raise TypeError('Randomized AugmentedPCA oversampling value must be an integer value greater than or ' + 
                            'equal to 0.')
        if n_oversamp < 1.0:
            raise ValueError('Randomized AugmentedPCA oversampling value must be a value greater than or equal to 0.')
        if not isinstance(n_oversamp, int):
            if (n_oversamp - round(n_oversamp)) < 1e-10:
                warn(message=('Warning: Randomized AugmentedPCA oversampling value must be an integer value ' + 
                              'greater than or equal to 0. Rounding to the nearest integer'))
                n_oversamp = round(n_oversamp)
            else:
                raise TypeError('Randomized AugmentedPCA oversampling value must be an integer value greater than ' + 
                                'or equal to 0.')
        self.n_oversamp = n_oversamp
        
        # Check for proper type / value and assign diagonal regularization constant attribute
        if not isinstance(diag_const, float) and not isinstance(diag_const, int):
            raise TypeError('Diagonal regularization constant must be numeric.')
        elif diag_const < 0.0:
            raise ValueError('Diagonal regularization constant must be a positive numeric value.')
        self.diag_const_ = diag_const
        
        # Check for proper type/value and assign random state value
        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError('Random state value must be an integer value.')
        self.random_state = random_state
        
        # Initialize all other attributes
        self.mean_X_ = None
        self.mean_Y_ = None
        self.mean_Z_ = None
        self.B_ = None
        self.W_ = None
        self.D_ = None
        self.eigvals_ = None
        self.is_fitted_ = False
    
    # Fits AugmentedPCA model to data
    def fit(self, X: numpy.ndarray, Y: Union[numpy.ndarray, tuple, list]):
        r"""
        Fits AugmentedPCA model to data.
        
        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional (n x p) primary data matrix.
        Y : numpy.ndarray or tuple / list of numpy.ndarrays
            2-dimensional (n x q) augmenting data matrix or tuple of two 2-dimensional (n x qs), (n x qa) augmenting
            data matrices.
        """
        
        # Check for proper type and shape of primary data matrix
        if not isinstance(X, numpy.ndarray):
            raise TypeError('Primary data matrix X must be of type numpy.ndarray.')
        if len(X.shape) != 2:
            raise ValueError('Primary data matrix X must be a 2-dimensional numpy.ndarray.')
        
        # Check for proper type and shape of augmenting data matrix / matrices
        if self.__class__.__name__ == 'CAPCA':
            if not isinstance(Y, tuple) and not isinstance(Y, list):
                raise TypeError('For Combined AugmentedPCA, Y must be provided as a tuple or list containing two ' +
                                'numpy.ndarrays.')
                if len(Y.shape) != 2:
                    raise ValueError('Augmenting data matrix Y must be a 2-dimensional numpy.ndarray.')
            else:
                if len(Y) != 2:
                    raise TypeError('For Combined AugmentedPCA, Y must be provided as a tuple or list containing ' +
                                    'two numpy.ndarrays.')
                for Y_ in Y:
                    if not isinstance(Y_, numpy.ndarray):
                        raise TypeError('Y must be of type numpy.ndarray or a tuple / list of numpy.ndarrays.')
                    if len(Y_.shape) != 2:
                        raise ValueError('All augmenting data matrices Y must be a 2-dimensional numpy.ndarrays.')
        else:
            if not isinstance(Y, numpy.ndarray):
                raise TypeError('Y must be of type numpy.ndarray.')
        
        # Create deep copies of primary and augmenting data matrices
        X_ = X.copy()
        if self.__class__.__name__ == 'CAPCA':
            Ys_, Ya_ = Y[0].copy(), Y[1].copy()
        else:
            Y_ = Y.copy()
        
        # Raise error for sparse matrices used as input
        if self.__class__.__name__ == 'CAPCA':
            if issparse(X_) or issparse(Ys_) or issparse(Ya_):
                error_msg = '%(name) does not support sparse input.'
                raise TypeError(error_msg % {'name': self.__class__.__name__})
        else:
            if issparse(X_) or issparse(Y_):
                error_msg = '%(name) does not support sparse input.'
                raise TypeError(error_msg % {'name': self.__class__.__name__})
        
        # Define number of components as smallest dimension of X if not specified
        if self.n_components is None:
            self.n_components = min(X_.shape)
        
        # Check that specified number of components is not too large
        else:
            if self.n_components > min(X_.shape):
                raise ValueError('Number of components is too large.')
        
        # Check oversampling value
        if (self.n_components + self.n_oversamp) > min(X_.shape):
            raise ValueError('Sum of number of components and oversampling number must be less than the smallest ' +
                             'dimension of the primary data matrix.')
        
        # Calculate means of data matrices
        self.mean_X_ = mean(X_, axis=0)
        if self.__class__.__name__ == 'CAPCA':
            self.mean_Y_ = (mean(Ys_, axis=0), mean(Ya_, axis=0))
        else:
            self.mean_Y_ = mean(Y_, axis=0)
        
        # Mean-center primary and augmenting data matrices
        X_ -= self.mean_X_
        if self.__class__.__name__ == 'CAPCA':
            Ys_ -= self.mean_Y_[0]
            Ya_ -= self.mean_Y_[1]
        else:
            Y_ -= self.mean_Y_
        
        # Define Z as concatenation of primary data and augmenting data matrices
        if self._inference == 'local' or self._inference == 'encoded':
            Z_ = None
        elif self._inference == 'joint':
            if self.__class__.__name__ == 'CAPCA':
                Z_ = concatenate((X_, Ys_, Ya_), axis=1)
            else:
                Z_ = concatenate((X_, Y_), axis=1)
            self.mean_Z_ = mean(Z_, axis=0)
            Z_ -= self.mean_Z_
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Get data dimensions
        n, p = X_.shape
        if self.__class__.__name__ == 'CAPCA':
            _, qs = Ys_.shape
            _, qa = Ya_.shape
        else:
            _, q = Y_.shape
        
        # Get decomposition matrix
        if self._inference == 'local' or self._inference == 'encoded':
            if self.__class__.__name__ == 'CAPCA':
                self.B_ = self._get_B(M_=X_, N_=(Ys_, Ya_))
            else:
                self.B_ = self._get_B(M_=X_, N_=Y_)
        elif self._inference == 'joint':
            if self.__class__.__name__ == 'CAPCA':
                self.B_ = self._get_B(M_=Z_, N_=(Ys_, Ya_))
            else:
                self.B_ = self._get_B(M_=Z_, N_=Y_)
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Exact analytic AugmentedPCA formulation via eigendecomposition
        if self._decomp == 'exact':
            # Eigendecomposition
            eigvals, eigvecs = eig(self.B_)
            eigvals = real(eigvals)
            eigvecs = real(eigvecs)
            
            # Sort eigenvectors according to descending eigenvalues
            idx = eigvals.argsort()[::-1]
            self.eigvals_ = eigvals[idx]
            V = eigvecs[:, idx]
        
        # Randomized AugmentedPCA approximation - Algorithm 4.4. in Halko et al. (2009)
        elif self._decomp == 'approx':
            # Random number generator instance
            if self.random_state is None:
                rng = default_rng()
            else:
                rng = default_rng(self.random_state)

            # Draw random matrix from standard normal
            omega = rng.standard_normal((self.B_.shape[1], self.n_components + self.n_oversamp))
            
            # Initial subspace
            Q, _ = qr(self.B_ @ omega)
            
            # Decomposition matrix transpose
            B_T = self.B_.copy().T
            
            # Subspace power iterations
            for i in range(self.pow_iter):
                Q_, _ = qr(B_T @ Q)
                Q, _ = qr(self.B_ @ Q_)
            
            # Approximate decomposition matrix
            B_approx = Q.T @ self.B_ @ Q
            
            # Eigendecomposition of approximate decomposition matrix
            eigvals_approx, eigvecs_approx = eig(B_approx)
            eigvals_approx = real(eigvals_approx)
            eigvecs_approx = real(eigvecs_approx)
            
            # Sort eigenvectors according to descending eigenvalues
            idx = eigvals_approx.argsort()[::-1]
            self.eigvals_ = eigvals_approx[idx]
            V = Q @ eigvecs_approx[:, idx]
        
        # Decomposition approach not recognized
        else:
            raise ValueError('Invalid decomposition approach parameter \'%s\'.' % self._decomp)
        
        # Assign primary loadings W and augmented loadings D
        self.W_ = V[:p, :self.n_components]
        if self._inference == 'local' or self._inference == 'encoded':
            if self.__class__.__name__ == 'CAPCA':
                self.D_ = (V[p:p + qs, :self.n_components], V[p + qs:p + qs + qa, :self.n_components])
            else:
                self.D_ = V[p:p + q, :self.n_components]
        elif self._inference == 'joint':
            if self.__class__.__name__ == 'CAPCA':
                self.D_ = (V[p + qs + qa:p + (2 * qs) + qa, :self.n_components],
                           V[p + (2 * qs) + qa,:p + (2 * qs) + (2 * qa), :self.n_components])
            else:
                self.D_ = V[p + q:, :self.n_components]
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # No encoding matrix for local approximate inference
        if self._inference == 'local':
            self.A_ = None
        
        # Generate encoding matrix for encoded approximate inference
        elif self._inference == 'encoded':
            diag_reg = self.diag_const_ * identity(n=X_.shape[1])
            inv_XT_X = inv((X_.T @ X_) + diag_reg)
            diag_reg = self.diag_const_ * identity(n=self.W_.shape[1])
            A1 = self.W_.T @ self.W_
            A2 = self.W_.T @ X_.T
            if self.__class__.__name__ == 'CAPCA':
                if isinstance(self.mu, tuple) or isinstance(self.mu, list):
                    A1 -= (self.mu[0] * (self.D_[0].T @ self.D_[0])) + \
                    (self.mu[1] * (self.D_[1].T @ self.D_[1])) - diag_reg
                    A2 -= (self.mu[0] * (self.D_[0].T @ Ys_.T)) + (self.mu[1] * (self.D_[1].T @ Ya_.T))
                else:
                    A1 -= (self.mu * (self.D_[0].T @ self.D_[0])) + \
                    (self.mu * (self.D_[1].T @ self.D_[1])) - diag_reg
                    A2 -= (self.mu * (self.D_[0].T @ Ys_.T)) + (self.mu * (self.D_[1].T @ Ya_.T))
            else:
                A1 -= (self.mu * (self.D_.T @ self.D_)) - diag_reg
                A2 -= self.mu * (self.D_.T @ Y_.T)
            self.A_ = solve(A1, A2) @ X_ @ inv_XT_X
        
        # Generate encoding matrix for jointly-encoded approximate inference
        elif self._inference == 'joint':
            diag_reg = self.diag_const_ * identity(n=Z_.shape[1])
            inv_ZT_Z = inv((Z_.T @ Z_) + diag_reg)
            diag_reg = self.diag_const_ * identity(n=self.W_.shape[1])
            A1 = self.W_.T @ self.W_
            A2 = self.W_.T @ X_.T
            if self.__class__.__name__ == 'CAPCA':
                if isinstance(self.mu, tuple) or isinstance(self.mu, list):
                    A1 -= ((self.mu[0] + 1.0) * (self.D_[0].T @ self.D_[0])) + \
                    ((self.mu[1] + 1.0) * (self.D_[1].T @ self.D_[1])) - diag_reg
                    A2 -= ((self.mu[0] + 1.0) * (self.D_[0].T @ Ys_.T)) + \
                    ((self.mu[1] + 1.0) * (self.D_[1].T @ Ya_.T))
                else:
                    A1 -= ((self.mu + 1.0) * (self.D_[0].T @ self.D_[0])) + \
                    ((self.mu + 1.0) * (self.D_[1].T @ self.D_[1])) - diag_reg
                    A2 -= ((self.mu + 1.0) * (self.D_[0].T @ Ys_.T)) + \
                    ((self.mu + 1.0) * (self.D_[1].T @ Ya_.T))
            else:
                A1 -= ((self.mu + 1.0) * (self.D_.T @ self.D_)) - diag_reg
                A2 -= (self.mu + 1.0) * (self.D_.T @ Y_.T)
            self.A_ = solve(A1, A2) @ Z_ @ inv_ZT_Z
        
        # Inference strategy not recognized
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Update fitted variable
        self.is_fitted_ = True
        
        # Return reference to object
        return self
    
    # Transforms data into scores using AugmentedPCA model formulation
    def transform(self, X: numpy.ndarray, Y: Union[numpy.ndarray, tuple, list]) -> numpy.ndarray:
        r"""
        Transforms data into scores using AugmentedPCA model formulation.
        
        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional (n x p) primary data matrix.
        Y : numpy.ndarray or tuple / list of numpy.ndarrays
            2-dimensional (n x q) augmenting data matrix or tuple of two 2-dimensional (n x qs), (n x qa) augmenting
            data matrices. Ignored if `inference` is set to 'encoded'.
        
        Returns
        -------
        S : numpy.ndarray
            2-dimensional (n x k) scores matrix.
        """
        
        # Check that model is fitted
        if not self.is_fitted_:
            error_msg = ('This %(name)\'s instance is not fitted yet. Call \'fit\' with '
                         'appropriate arguments before using this method.')
            raise NotFittedError(error_msg % {'name': self.__class__.__name__})
        
        # Check for proper type and shape of primary data matrix
        if not isinstance(X, numpy.ndarray):
            raise TypeError('Primary data matrix X must be of type numpy.ndarray.')
        if len(X.shape) != 2:
            raise ValueError('Primary data matrix X must be a 2-dimensional numpy.ndarray.')
        
        # Check for proper type and shape of augmenting data matrix / matrices
        if self._inference == 'local' or self._inference == 'joint':
            if self.__class__.__name__ == 'CAPCA':
                if not isinstance(Y, tuple) and not isinstance(Y, list):
                    raise TypeError('For Combined AugmentedPCA, Y must be provided as a tuple or list containing two ' +
                                    'numpy.ndarrays.')
                    if len(Y.shape) != 2:
                        raise ValueError('Augmenting data matrix Y must be a 2-dimensional numpy.ndarray.')
                else:
                    if len(Y) != 2:
                        raise TypeError('For Combined AugmentedPCA, Y must be provided as a tuple or list containing ' +
                                        'two numpy.ndarrays.')
                    for Y_ in Y:
                        if not isinstance(Y_, numpy.ndarray):
                            raise TypeError('Y must be of type numpy.ndarray or a tuple / list of numpy.ndarrays.')
                        if len(Y_.shape) != 2:
                            raise ValueError('All augmenting data matrices Y must be a 2-dimensional numpy.ndarrays.')
            else:
                if not isinstance(Y, numpy.ndarray):
                    raise TypeError('Y must be of type numpy.ndarray.')
        
        # Create deep copy of primary data matrix
        X_ = X.copy()
        
        # Mean-center primary data
        X_ -= self.mean_X_
        
        # Generate scores - local approximate inference
        if self._inference == 'local':
            S1 = self.W_.T @ self.W_
            S2 = self.W_.T @ X_.T
            if self.__class__.__name__ == 'CAPCA':
                Ys_, Ya_ = Y[0].copy(), Y[0].copy()
                Ys_ -= self.mean_Y_[0]
                Ya_ -= self.mean_Y_[1]
                if isinstance(self.mu, tuple) or isinstance(self.mu, list):
                    S1 -= (self.mu[0] * (self.D_[0].T @ self.D_[0])) + (self.mu[1] * (self.D_[1].T @ self.D_[1]))
                    S2 -= (self.mu[0] * (self.D_[0].T @ Ys_.T)) + (self.mu[1] * (self.D_[1].T @ Ya_.T))
                else:
                    S1 -= self.mu * ((self.D_[0].T @ self.D_[0]) + (self.D_[1].T @ self.D_[1]))
                    S2 -= self.mu * ((self.D_[0].T @ Ys_.T) + (self.D_[1].T @ Ya_.T))
            else:
                Y_ = Y.copy()
                Y_ -= self.mean_Y_
                S1 -= self.mu * (self.D_.T @ self.D_)
                S2 -= self.mu * (self.D_.T @ Y_.T)
            S = (solve(S1, S2)).T
        
        # Generate scores - encoded approximate inference
        elif self._inference == 'encoded':
            S = (self.A_ @ X_.T).T
        
        # Generate scores - jointly-encoded approximate inference
        elif self._inference == 'joint':
            if self.__class__.__name__ == 'CAPCA':
                Ys_, Ya_ = Y[0].copy(), Y[0].copy()
                Ys_ -= self.mean_Y_[0]
                Ya_ -= self.mean_Y_[1]
                Z_ = concatenate((X_, Ys_, Ya_), axis=1)
            else:
                Y_ = Y.copy()
                Y_ -= self.mean_Y_
                Z_ = concatenate((X_, Y_), axis=1)
            S = (self.A_ @ Z_.T).T
        
        # Inference strategy not recognized
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Return scores
        return S
    
    # Fit AugmentedPCA model to data and transform data into scores
    def fit_transform(self, X: numpy.ndarray, Y: Union[numpy.ndarray, tuple, list]) -> numpy.ndarray:
        r"""
        Fits AugumentedPCA model to data and transforms data into scores.
        
        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional (n x p) primary data matrix.
        Y : numpy.ndarray or tuple / list of numpy.ndarrays
            2-dimensional (n x q) augmenting data matrix or tuple of two 2-dimensional (n x qs), (n x qa) augmenting
            data matrices.
        
        Returns
        -------
        S : numpy.ndarray
            2-dimensional (n x k) scores matrix.
        """
        
        # Fit AugmentedPCA model instance
        self.fit(X=X, Y=Y)
        
        # Generate factors from data
        S = self.transform(X=X, Y=Y)
        
        # Return scores
        return S
    
    # Reconstruct primary and augmenting data
    def reconstruct(self, X: numpy.ndarray, Y: Union[numpy.ndarray, tuple, list]) -> numpy.ndarray:
        r"""
        Reconstruct primary and augmenting data.
        
        Parameters
        ----------
        X : numpy.ndarray
            2-dimensional (n x p) primary data matrix.
        Y : numpy.ndarray or tuple / list of numpy.ndarrays
            2-dimensional (n x q) augmenting data matrix or tuple of two 2-dimensional (n x qs), (n x qa) augmenting
            data matrices. Ignored if `inference` is set to 'encoded'.
        
        Returns
        -------
        X_recon : numpy.ndarray
            2-dimensional (n x p) reconstruction of primary data.
        Y_recon : numpy.ndarray or tuple / list of numpy.ndarrays
            2-dimensional (n x q) reconstruction of augmenting data or tuple of 2-dimensional (n x qs), (n x qa)
            augmenting data reconstructions.
        """
        
        # Generate factors from data
        S = self.transform(X=X, Y=Y)
        
        # Reconstruct both primary and augmenting data
        X_recon = (S @ self.W_.T) + self.mean_X_
        if self.__class__.__name__ == 'CAPCA':
            Y_recon = ((S @ self.D_[0].T) + self.mean_Y_[0], (S @ self.D_[1].T) + self.mean_Y_[1])
        else:
            Y_recon = (S @ self.D_.T) + self.mean_Y_
        
        # Return reconstructed primary and augmenting data
        return X_recon, Y_recon
    
    # Get eigenvalues
    def get_eigvals(self) -> numpy.ndarray:
        r"""
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.eigvals_.copy() : numpy.ndarray
            1-dimensional array of sorted eigenvalues.
        """
        
        # Check that encoding matrix attribute has been assigned
        if self.eigvals_ is None:
            raise AttributeError('Eigenvalues attribute not yet assigned.')
        
        # Return encoding matrix
        return self.eigvals_.copy()
    
    # Get primary data loadings / components
    def get_components(self) -> numpy.ndarray:
        r"""
        Returns primary data loadings / components.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.W_.copy() : numpy.ndarray
            2-dimensional (p x k) primary data loadings matrix.
        """
        
        # Retrieve primary data loadings
        W = self.get_W()
        
        # Return primary data loadings
        return W
    
    # Get primary data loadings
    def get_W(self) -> numpy.ndarray:
        r"""
        Returns primary data loadings.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.W_.copy() : numpy.ndarray
            2-dimensional (p x k) primary data loadings matrix.
        """
        
        # Check that primary data loadings attribute has been assigned
        if self.W_ is None:
            raise AttributeError('Primary data loadings attribute not yet assigned.')
        
        # Return primary data loadings
        return self.W_.copy()
    
    # Get augmenting data loadings
    @abstractmethod
    def get_D(self):
        r"""
        Abstract method : Returns augmenting data loadings.
        """
    
    # Get encoding matrix
    def get_A(self) -> numpy.ndarray:
        r"""
        Returns encoding matrix.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.A_.copy() : numpy.ndarray
            2-dimensional (d x p) encoding matrix.
        """
        
        # Check that encoding matrix attribute has been assigned
        if self.A_ is None:
            if self._inference == 'local':
                raise AttributeError('AugmentedPCA object with local approximate inference does not have an ' + 
                                     'encoding matrix attribute.')
            else:
                raise AttributeError('Encoding matrix attribute not yet assigned.')
        
        # Return encoding matrix
        return self.A_.copy()
    
    # Returns decomposition matrix given primary data matrix and augmenting data matrix
    @abstractmethod
    def _get_B(self):
        r"""
        Abstract method : Calculates and returns decomposition matrix
        """


# Supervised AugmentedPCA model class
class SAPCA(_APCA):
    r"""
    Supervised AugmentedPCA (sAPCA) model class. The objective of the sAPCA model is to find components that 1)
    represent the maximum variance expressed in the primary data (primary objective) and 2) represent the variance
    expressed in the data labels or outcome data (augmenting objective).
    
    Parameters
    ----------
    n_components : int; optional, default is None
        Number of components. If None reduce to minimum dimension of primary data.
    mu : float; optional, default is 1.0
        Supervision strength.
    inference : str; optional, default is 'encoded'
        Model inference strategy.
    decomp : str; optional, default is 'approx'
        Decomposition approach.
    pow_iter : int; optional, default is 5
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int; optional, default is 5
        Oversampling parameter for randomized approximation.
    diag_const : float; optional, default is 1e-8
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    
    Attributes
    ----------
    n_components : int
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float
        Supervision strength.
    pow_iter : int
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int
        Oversampling parameter for randomized approximation.
    diag_const : float
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    mean_X_ : numpy.ndarray
        1-dimensional (p,) mean array of primary data matrix.
    mean_Y_ : numpy.ndarray
        1-dimensional (q,) mean array of primary data matrix.
    mean_Z_ : numpy.ndarray
        1-dimensional (p + q,) mean array of combined primary and supervised data matrices.
    B_ : numpy.ndarray
        2-dimensional decomposition matrix.
    W_ : numpy.ndarray
        2-dimensional primary data loadings matrix.
    D_ : numpy.ndarray
        2-dimensional supervised data loadings matrix.
    A_ : numpy.ndarray
        2-dimensional encoding matrix. None if `inference` is set to 'local'.
    eigvals_ : numpy.ndarray
        1-dimensional array of sorted decomposition matrix eigenvalues.
    is_fitted_ : bool
        Indicates whether model has been fitted.
    
    Methods
    -------
    fit(X, Y)
        Fits AugmentedPCA model to data.
    transform(X, Y)
        Transforms data into scores using AugmentedPCA model formulation.
    fit_transform(X, Y)
        Fits AugumentedPCA model to data and transforms data into scores.
    reconstruct(X, Y)
        Reconstructs primary and supervised data.
    get_components()
        Returns primary data loadings / components. Alias for get_W().
    get_W()
        Returns primary data loadings.
    get_D()
        Returns supervised data loadings.
    get_A()
        Returns encoding matrix.
    get_eigvals()
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues.
    """
    
    # Instantiation method of supervised AugmentedPCA model base class
    def __init__(
        self,
        n_components=None,
        mu=1.0,
        inference='encoded',
        decomp='approx',
        pow_iter=5,
        n_oversamp=5,
        diag_const=1e-8,
        random_state=None):
        r"""
        Instantiation method of supervised AugmentedPCA model class.
        
        Parameters
        ----------
        n_components : int; optional, default is None
            Number of components. If None reduce to minimum dimension of primary data.
        mu : float; optional, default is 1.0
            Supervision strength.
        inference : str; optional, default is 'encoded'
            Model inference strategy.
        decomp : str; optional, default is 'approx'
            Decomposition approach.
        pow_iter : int; optional, default is 5
            Number of power iterations to perform in randomized approximation.
        n_oversamp : int; optional, default is 5
            Oversampling parameter for randomized approximation.
        diag_const : float; optional, default is 1e-8
            Constant added to diagonals of matrix prior to inversion.
        random_state : int
            Model random state. Ignored if exact eigenvalue decomposition approach used.
        """
        
        # Inherit from AugmentedPCA model base class
        super().__init__(
            n_components=n_components,
            mu=mu,
            inference=inference,
            decomp=decomp,
            pow_iter=pow_iter,
            n_oversamp=n_oversamp,
            diag_const=diag_const,
            random_state=random_state)
    
    # Get supervised data loadings
    def get_D(self) -> numpy.ndarray:
        r"""
        Returns supervised data loadings.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.D_.copy() : numpy.ndarray
            2-dimensional (q x k) supervised data loadings matrix.
        """
        
        # Check that supervised data loadings attribute has been assigned
        if self.D_ is None:
            raise AttributeError('Supervised data loadings attribute not yet assigned.')
        
        # Return supervised data loadings
        return self.D_.copy()
    
    # Returns decomposition matrix given primary data matrix and supervised data matrix
    def _get_B(self, M_: numpy.ndarray, N_: numpy.ndarray) -> numpy.ndarray:
        r"""
        Returns decomposition matrix given primary data matrix and supervised data matrix.
        
        Parameters
        ----------
        M_ : numpy.ndarray
            Deep copy of 2-dimensional (n x p) or (n x (p + q)) matrix.
        N_ : numpy.ndarray
            Deep copy of 2-dimensional (n x q) supervised data matrix.
        
        Returns
        -------
        B : numpy.ndarray
            2-dimensional ((p + q) x (p + q)) or ((p + 2 * q) x (p + 2 * q)) decomposition matrix.
        """
        
        # Define components of decomposition matrix B
        B11 = M_.T @ M_
        B12 = M_.T @ N_
        B21 = B12.T
        if self._inference == 'local':
            B22 = N_.T @ N_
        elif self._inference == 'encoded' or self._inference == 'joint':
            diag_reg = self.diag_const_ * identity(M_.shape[1])
            MT_M = (M_.T @ M_) + diag_reg
            B22 = B12.T @ solve(MT_M, B12)
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Decomposition matrix with positive augmenting objective strength - local or encoded approximate inference
        if self._inference == 'local' or self._inference == 'encoded':
            B = concatenate((concatenate((B11, self.mu * B12), axis=1),
                             concatenate((B21, self.mu * B22), axis=1)), axis=0)
        
        # Decomposition matrix with positive augmenting objective strength - jointly-encoded approximate inference
        elif self._inference == 'joint':
            B = concatenate((concatenate((B11, (self.mu + 1.0) * B12), axis=1),
                             concatenate((B21, (self.mu + 1.0) * B22), axis=1)), axis=0)
        
        # Inference strategy not recognized
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Return decomposition matrix
        return B


# Adversarial AugmentedPCA model class
class AAPCA(_APCA):
    r"""
    Adversarial AugmentedPCA (aAPCA) model class. The objective of the aAPCA model is to find components that 1)
    represent the maximum variance expressed in the primary data (primary objective) while 2) maintaining a degree of
    invariance to a set of concomitant data (augmenting objective).
    
    Parameters
    ----------
    n_components : int; optional, default is None
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float; optional, default is 1.0
        Adversary strength.
    inference : str; optional, default is 'encoded'
        Model inference strategy.
    decomp : str; optional, default is 'approx'
        Decomposition approach.
    pow_iter : int; optional, default is 5
        Number of power iterations to perform in randomized AugmentedPCA approximation.
    n_oversamp : int; optional, default is 5
        Oversampling parameter for randomized AugmentedPCA approximations.
    diag_const : float; optional, default is 1e-8
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    
    Attributes
    -----------
    n_components : int
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float
        Adversary strength.
    pow_iter : int
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int
        Oversampling parameter for randomized approximation.
    diag_const : float
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    mean_X_ : numpy.ndarray
        1-dimensional (p,) mean array of primary data matrix.
    mean_Y_ : numpy.ndarray
        1-dimensional (q,) mean array of primary data matrix.
    mean_Z_ : numpy.ndarray
        1-dimensional (p + q,) mean array of combined primary and concomitant data matrices.
    B_ : numpy.ndarray
        2-dimensional decomposition matrix.
    W_ : numpy.ndarray
        2-dimensional primary data loadings matrix.
    D_ : numpy.ndarray
        2-dimensional concomitant data loadings matrix.
    A_ : numpy.ndarray
        2-dimensional encoding matrix. None if `inference` is set to 'local'.
    eigvals_ : numpy.ndarray
        1-dimensional array of sorted decomposition matrix eigenvalues.
    is_fitted_ : bool
        Indicates whether model has been fitted.
    
    Methods
    -------
    fit(X, Y)
        Fits AugmentedPCA model to data.
    transform(X, Y)
        Transforms data into scores using AugmentedPCA model formulation.
    fit_transform(X, Y)
        Fits AugumentedPCA model to data and transforms data into scores.
    reconstruct(X, Y)
        Reconstructs primary and concomitant data.
    get_components()
        Returns primary data loadings / components. Alias for get_W().
    get_W()
        Returns primary data loadings.
    get_D()
        Returns concomitant data loadings.
    get_A()
        Returns encoding matrix.
    get_eigvals()
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues.
    """
    
    # Instantiation method of adversarial AugmentedPCA model base class
    def __init__(
        self,
        n_components=None,
        mu=1.0,
        inference='encoded',
        decomp='approx',
        pow_iter=5,
        n_oversamp=5,
        diag_const=1e-8,
        random_state=None):
        r"""
        Instantiation method of adversarial AugmentedPCA model class.
        
        Parameters
        ----------
        n_components : int; optional, default is None
            Number of components. If None reduce to minimum dimension of primary data.
        mu : float; optional, default is 1.0
            Adversary strength
        inference : str; optional, default is 'encoded'
            Model inference strategy.
        decomp : str; optional, default is 'approx'
            Decomposition approach.
        pow_iter : int; optional, default is 5
            Number of power iterations to perform in randomized approximation.
        n_oversamp : int; optional, default is 5
            Oversampling parameter for randomized approximation.
        diag_const : float; optional, default is 1e-8
            Constant added to diagonals of matrix prior to inversion.
        random_state : int; optional, default is None
            Model random state. Ignored if exact eigenvalue decomposition approach used.
        """
        
        # Inherit from AugmentedPCA model base class
        super().__init__(
            n_components=n_components,
            mu=mu,
            inference=inference,
            decomp=decomp,
            pow_iter=pow_iter,
            n_oversamp=n_oversamp,
            diag_const=diag_const,
            random_state=random_state)
    
    # Get concomitant data loadings
    def get_D(self) -> numpy.ndarray:
        r"""
        Returns concomitant data loadings.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.D_.copy() : numpy.ndarray
            2-dimensional (q x k) concomitant data loadings matrix.
        """
        
        # Check that concomitant data loadings attribute has been assigned
        if self.D_ is None:
            raise AttributeError('Concomitant data loadings attribute not yet assigned.')
        
        # Return concomitant data loadings
        return self.D_.copy()
    
    # Returns decomposition matrix given primary data matrix and concomitant data matrix
    def _get_B(self, M_: numpy.ndarray, N_: numpy.ndarray) -> numpy.ndarray:
        r"""
        Returns decomposition matrix given primary data matrix and concomitant data matrix.
        
        Parameters
        ----------
        M_ : numpy.ndarray
            Deep copy of 2-dimensional (n x p) or (n x (p + q)) matrix.
        N_ : numpy.ndarray
            Deep copy of 2-dimensional (n x q) concomitant data matrix.
        
        Returns
        -------
        B : numpy.ndarray
            2-dimensional ((p + q) x (p + q)) or ((p + (2 * q)) x (p + (2 * q))) decomposition matrix.
        """
        
        # Define components of decomposition matrix B
        B11 = M_.T @ M_
        B12 = M_.T @ N_
        B21 = B12.T
        if self._inference == 'local':
            B22 = N_.T @ N_
        elif self._inference == 'encoded' or self._inference == 'joint':
            diag_reg = self.diag_const_ * identity(M_.shape[1])
            MT_M = (M_.T @ M_) + diag_reg
            B22 = B12.T @ solve(MT_M, B12)
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Decomposition matrix with negative augmenting objective strength - local or encoded approximate inference
        if self._inference == 'local' or self._inference == 'encoded':
            B = concatenate((concatenate((B11, -self.mu * B12), axis=1),
                             concatenate((B21, -self.mu * B22), axis=1)), axis=0)
        
        # Decomposition matrix with negative augmenting objective strength - jointly-encoded approximate inference
        elif self._inference == 'joint':
            B = concatenate((concatenate((B11, -(self.mu + 1.0) * B12), axis=1),
                             concatenate((B21, -(self.mu + 1.0) * B22), axis=1)), axis=0)
        
        # Inference strategy not recognized
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Return decomposition matrix
        return B


# Combined AugmentedPCA model class
class CAPCA(_APCA):
    r"""
    Combined AugmentedPCA (cAPCA) model class. The objective of the cAPCA model is to find components that 1)
    represent the maximum variance expressed in the primary data (primary objective) and 2) represent the variance
    expressed in the data labels or outcome data and maintain a degree of invariance to a set of concomitant data
    (augmenting objectives).
    
    Parameters
    ----------
    n_components : int; optional, default is None
        Number of components. If None reduce to minimum dimension of primary data.
    mu : float or tuple or list; optional, default is 1.0
        Augmenting objective strength(s).
    inference : str; optional, default is 'encoded'
        Model inference strategy.
    decomp : str; optional, default is 'approx'
        Decomposition approach.
    pow_iter : int; optional, default is 5
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int; optional, default is 5
        Oversampling parameter for randomized approximation.
    diag_const : float; optional, default is 1e-8
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    
    Attributes
    ----------
    n_components : int
        Number of components. If None then reduce to minimum dimension of primary data.
    mu : float or tuple or list
        Augmenting objective strength(s).
    pow_iter : int
        Number of power iterations to perform in randomized approximation.
    n_oversamp : int
        Oversampling parameter for randomized approximation.
    diag_const : float
        Constant added to diagonals of matrix prior to inversion.
    random_state : int
        Model random state. Ignored if exact eigenvalue decomposition approach used.
    mean_X_ : numpy.ndarray
        1-dimensional (p,) mean array of primary data matrix.
    mean_Y_ : numpy.ndarray
        1-dimensional (q,) mean array of primary data matrix.
    mean_Z_ : numpy.ndarray
        1-dimensional (p + q,) mean array of combined primary and supervised data matrices.
    B_ : numpy.ndarray
        2-dimensional decomposition matrix.
    W_ : numpy.ndarray
        2-dimensional primary data loadings matrix.
    D_ : numpy.ndarray
        2-dimensional supervised data loadings matrix.
    A_ : numpy.ndarray
        2-dimensional encoding matrix. None if `inference` is set to 'local'.
    eigvals_ : numpy.ndarray
        1-dimensional array of sorted decomposition matrix eigenvalues.
    is_fitted_ : bool
        Indicates whether model has been fitted.
    
    Methods
    -------
    fit(X, Y)
        Fits AugmentedPCA model to data.
    transform(X, Y)
        Transforms data into scores using AugmentedPCA model formulation.
    fit_transform(X, Y)
        Fits AugumentedPCA model to data and transforms data into scores.
    reconstruct(X, Y)
        Reconstructs primary and supervised data.
    get_components()
        Returns primary data loadings / components. Alias for get_W().
    get_W()
        Returns primary data loadings.
    get_D()
        Returns supervised data loadings.
    get_A()
        Returns encoding matrix.
    get_eigvals()
        Returns 1-dimensional array of sorted decomposition matrix eigenvalues.
    """
    
    # Instantiation method of combined AugmentedPCA model base class
    def __init__(
        self,
        n_components=None,
        mu=1.0,
        inference='encoded',
        decomp='approx',
        pow_iter=5,
        n_oversamp=5,
        diag_const=1e-8,
        random_state=None):
        r"""
        Instantiation method of combined AugmentedPCA model class.
        
        Parameters
        ----------
        n_components : int; optional, default is None
            Number of components. If None reduce to minimum dimension of primary data.
        mu : float or tuple or list; optional, default is 1.0
            Augmenting ojective strength(s).
        inference : str; optional, default is 'encoded'
            Model inference strategy.
        decomp : str; optional, default is 'approx'
            Decomposition approach.
        pow_iter : int; optional, default is 5
            Number of power iterations to perform in randomized approximation.
        n_oversamp : int; optional, default is 5
            Oversampling parameter for randomized approximation.
        diag_const : float; optional, default is 1e-8
            Constant added to diagonals of matrix prior to inversion.
        random_state : int
            Model random state. Ignored if exact eigenvalue decomposition approach used.
        """
        
        # Inherit from AugmentedPCA model base class
        super().__init__(
            n_components=n_components,
            mu=mu,
            inference=inference,
            decomp=decomp,
            pow_iter=pow_iter,
            n_oversamp=n_oversamp,
            diag_const=diag_const,
            random_state=random_state)
    
    # Get supervised data loadings
    def get_D(self) -> numpy.ndarray:
        r"""
        Returns supervised data loadings.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.D_.copy() : numpy.ndarray
            Tuple of two 2-dimensional (qs x k), (qa x k) augmenting data loadings matrices.
        """
        
        # Check that supervised data loadings attribute has been assigned
        if self.D_ is None:
            raise AttributeError('Augmenting data loadings attribute not yet assigned.')
        
        # Return supervised data loadings
        return (self.D_[0].copy(), self.D_[1].copy())
    
    # Returns decomposition matrix given primary data matrix and augmenting data matrices
    def _get_B(self, M_: numpy.ndarray, N_: numpy.ndarray) -> numpy.ndarray:
        r"""
        Returns decomposition matrix given primary data matrix and augmenting data matrices.
        
        Parameters
        ----------
        M_ : numpy.ndarray
            Deep copy of 2-dimensional (n x p) or (n x (p + q)) matrix.
        N_ : numpy.ndarray
            Tuple of two deep copies of 2-dimensional (n x qs), (n x qa) augmenting data.
        
        Returns
        -------
        B : numpy.ndarray
            2-dimensional (p + qs + qa) x (p + qs + qa) decomposition matrix.
        """
        
        # Define components of decomposition matrix B
        B11 = M_.T @ M_
        B12 = M_.T @ N_[0]
        B13 = M_.T @ N_[1]
        B31 = B13.T
        B21 = B12.T
        if self._inference == 'local':
            B22 = N_[0].T @ N_[0]
            B33 = N_[1].T @ N_[1]
            B23 = N_[0].T @ N_[1]
            B32 = B23.T
        elif self._inference == 'encoded' or self._inference == 'joint':
            diag_reg = self.diag_const_ * identity(M_.shape[1])
            MT_M = (M_.T @ M_) + diag_reg
            B22 = B12.T @ solve(MT_M, B12)
            B33 = B13.T @ solve(MT_M, B13)
            B23 = B12.T @ solve(MT_M, B13)
            B32 = B23.T
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Decomposition matrix with positive augmenting objective strength - local or encoded approximate inference
        if self._inference == 'local' or self._inference == 'encoded':
            if isinstance(self.mu, tuple) or isinstance(self.mu, list):
                B = concatenate((concatenate((B11, self.mu[0] * B12, -self.mu[1] * B13), axis=1),
                                 concatenate((B21, self.mu[0] * B22, -self.mu[1] * B23), axis=1),
                                 concatenate((B31, self.mu[0] * B32, -self.mu[1] * B33), axis=1)),
                                axis=0)
            else:
                B = concatenate((concatenate((B11, self.mu * B12, -self.mu * B13), axis=1),
                                 concatenate((B21, self.mu * B22, -self.mu * B23), axis=1),
                                 concatenate((B31, self.mu * B32, -self.mu * B33), axis=1)),
                                axis=0)
        
        # Decomposition matrix with positive augmenting objective strength - jointly-encoded approximate inference
        elif self._inference == 'joint':
            if isinstance(self.mu, tuple) or isinstance(self.mu, list):
                B = concatenate((concatenate((B11, (self.mu[0] + 1.0) * B12, -(self.mu[1] + 1.0) * B13), axis=1),
                                 concatenate((B21, (self.mu[0] + 1.0) * B22, -(self.mu[1] + 1.0) * B23), axis=1),
                                 concatenate((B31, (self.mu[0] + 1.0) * B32, -(self.mu[1] + 1.0) * B33), axis=1)),
                                axis=0)
            else:
                B = concatenate((concatenate((B11, (self.mu + 1.0) * B12, -(self.mu + 1.0) * B13), axis=1),
                                 concatenate((B21, (self.mu + 1.0) * B22, -(self.mu + 1.0) * B23), axis=1),
                                 concatenate((B31, (self.mu + 1.0) * B32, -(self.mu + 1.0) * B33), axis=1)),
                                axis=0)
        
        # Inference strategy not recognized
        else:
            raise ValueError('Invalid inference strategy parameter \'%s\'.' % self._inference)
        
        # Return decomposition matrix
        return B


# Not fitted error class
class NotFittedError(ValueError, AttributeError):
    r"""
    NotFittedError class inherits from both ValueError and AttributeError.
    
    Parameters
    ----------
    none
    
    Attributes
    ----------
    none
    """

