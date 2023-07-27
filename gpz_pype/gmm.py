import logging

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


class GMMbasic(object):
    """Gaussian Mixture Model (GMM) for Photo-z augmentation with sklearn.

    Attributes
    ----------
    ncomp : int
        Number of components for the GMM model.

    niter : int
        Number of iterations for the GMM model.

    tol : float
        Tolerance for the GMM model.

    random_state : int
        Random state for the GMM model.

    scale : bool
        If True, the data is scaled before training the GMM model.

    threshold : float
        Threshold for the GMM model.

    scaler : sklearn.preprocessing.RobustScaler
        Scaler for the GMM model.

    gmm_pop : sklearn.mixture.GaussianMixture
        GMM model for the population.

    gmm_train : sklearn.mixture.GaussianMixture
        GMM model for the training set.

    mixture_samples : dict
        Dictionary with the mixture samples for the population and
        the training set.

    preprocess : function
        Function to preprocess the data before training the GMM model.

    fit : function
        Function to train the GMM model.

    population : function
        Function to train the GMM model for the population.

    train : function
        Function to train the GMM model for the training set.

    divide : function
        Function to divide the training set into mixture samples.

    """

    def __init__(
        self,
        X_pop: np.ndarray = None,
        X_train: np.ndarray = None,
        Y_train: np.ndarray = None,
        ncomp: int = 10,
        threshold: float = 0.5,
        niter: int = 100,
        tol: float = 1e-3,
        random_state: int = 0,
        scale: bool = True,
    ):
        """

        Parameters
        ----------

        X_pop : array-like, shape (n_samples, n_features)
            The data array for the reference population.

        X_train : array-like, shape (n_samples, n_features)
            The data array for the training sample.

        Y_train : array-like, shape (n_samples,)
            The labels for the training sample.

        ncomp : int, default=10
            Number of components for the GMM model.

        threshold : float, default=0.5
            Threshold for the GMM model.

        niter : int, default=100
            Number of iterations for the GMM model.

        tol : float, default=1e-3
            Tolerance for the GMM model.

        random_state : int, default=0
            Random state for the GMM model.

        scale : bool, default=True
            If True, the data is scaled before training the GMM model.

        """
        self.ncomp = ncomp
        self.niter = niter
        self.tol = tol
        self.random_state = random_state
        self.scale = scale
        self.threshold = threshold
        self.Y_train = Y_train

        if X_pop is not None:
            if self.scale:
                self.rescaler = self.rescale(X_pop)
            self.population(X_pop)

        if X_train is not None:
            if X_pop is None:
                logging.warning(
                    "The reference population is not defined.\
The training set will be used as population."
                )
                if self.scale:
                    self.rescaler = self.rescale(X_train)
                self.population(X_train)
            else:
                self.train(X_train)

    def rescale(self, X):
        """Preprocess the data before training the GMM model.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        Returns
        -------

        X : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        """
        scaler = RobustScaler()
        X = scaler.fit(X)
        return X

    def fit(self, X):
        """Train the GMM model.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        Returns
        -------

        gmm : sklearn.mixture.GaussianMixture
            GMM model.

        """
        if self.scale:
            X = self.rescaler.transform(X)

        gmm = GaussianMixture(
            n_components=self.ncomp,
            n_init=self.niter,
            tol=self.tol,
            random_state=self.random_state,
        )
        gmm.fit(X)
        return gmm

    def population(self, X):
        """Train the GMM model for the population.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        """
        self.gmm_pop = self.fit(X)
        return self

    def train(self, X):
        """Train the GMM model for the training set.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        """
        self.gmm_train = self.fit(X)
        return self

    def divide(self, X, weight=False, eta=0.001, max_weight=100):
        """Divide the input array into mixture samples.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The input data to map onto the GMM.

        weight : bool, default=False
            If True, the cost-sensitive learning weights are calculated.

        eta : float, default=0.001
            Softening parameter for the weights.

        max_weight : float, default=100
            Maximum weight for the weights.

        """
        if not isinstance(self.gmm_pop, GaussianMixture):
            self.population(X)

        if weight:
            weights = self.calc_weights(
                X_train=X, X_pop=None, eta=0.001, max_weight=100
            )

        if self.scale:
            X = self.rescaler.transform(X)

        prob_i = self.gmm_pop.predict_proba(X)

        self.mixture_samples = {}

        for mx in range(self.ncomp):
            i_sample = np.where(prob_i[:, mx] > self.threshold)[0]

            if len(i_sample) > 0:
                self.mixture_samples[
                    f"{mx}_X"
                ] = self.rescaler.inverse_transform(X[i_sample])
                if weight:
                    self.mixture_samples[f"{mx}_w"] = weights[i_sample]

            else:
                self.mixture_samples[f"{mx}_X"] = None
                if weight:
                    self.mixture_samples[f"{mx}_w"] = None

        return self

    def calc_weights(self, X_train, X_pop=None, eta=0.001, max_weight=100):
        """Calculate the weights for the training set.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        X_pop : array-like, shape (n_samples, n_features)
            The data to train the GMM model.

        eta : float, default=0.001
            Softening parameter for the weights.

        max_weight : float, default=100
            Maximum weight to apply to the training set.

        Returns
        -------
        weights : array-like, shape (n_samples, )
            The weights for the training set.

        """
        if not isinstance(self.gmm_train, GaussianMixture):
            self.train(X_train)

        if not isinstance(self.gmm_pop, GaussianMixture) and X_pop is not None:
            self.population(X_pop)
        elif not isinstance(self.gmm_pop, GaussianMixture) and X_pop is None:
            raise ValueError("The population (X_pop) is not defined.")

        if self.scale:
            X_train = self.rescaler.transform(X_train)
        lnpi_train = self.gmm_train.score_samples(X_train)
        lnpi_pop = self.gmm_pop.score_samples(X_train)

        # norm = np.min([np.min(lnpi_train), np.min(lnpi_pop)])

        weights = (np.exp(lnpi_pop) + eta) / (np.exp(lnpi_train) + eta)
        weights[weights > max_weight] = max_weight

        return weights


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=10000,
        centers=10,
        n_features=5,
        random_state=0,
        cluster_std=3.0,
    )

    X_train_a, ya = make_blobs(
        n_samples=100,
        centers=10,
        n_features=5,
        random_state=0,
        cluster_std=2.0,
        center_box=(-10, 10),
    )
    X_train_b, yb = make_blobs(
        n_samples=300,
        centers=10,
        n_features=5,
        random_state=0,
        cluster_std=2.0,
        center_box=(-5, 5),
    )

    X_train = np.concatenate((X_train_a, X_train_b))

    gmm = GMMbasic(X_train=X_train, X_pop=X, ncomp=10)
    gmm.divide(X)

    weights = gmm.calc_weights(X_train)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=weights)
    ax[0].set_title("Original data")
    ax[1].scatter(X_train[:, 2], X_train[:, 3], c=weights)
    # ax[1].set_title("GMM")
    plt.show()
