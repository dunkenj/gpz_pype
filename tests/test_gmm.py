from gpz_pype.gmm import GMMsk


# Write a test for the GMMsk class
def test_GMMbasic():
    # Create a GMMsk object with some random data
    gmm = GMMbasic(
        X_pop=np.random.randn(100, 10),
        X_train=np.random.randn(100, 10),
        ncomp=10,
        threshold=0.5,
        niter=100,
        tol=1e-3,
    )

    # Check the attributes
    assert gmm.ncomp == 10
    assert gmm.niter == 100
    assert gmm.tol == 1e-3
    assert gmm.random_state == 0
    assert gmm.scale == True
    assert gmm.threshold == 0.5
    assert hasattr(gmm, "scaler") == True
    assert hasattr(gmm, "gmm_pop") == True
    assert hasattr(gmm, "gmm_train") == True
    assert hasattr(gmm, "mixture_samples") == True
    assert hasattr(gmm, "preprocess") == True
    assert hasattr(gmm, "fit") == True
    assert hasattr(gmm, "population") == True
    assert hasattr(gmm, "train") == True
    assert hasattr(gmm, "divide") == True

    # Check the type of the attributes
    assert isinstance(gmm.ncomp, int) == True
    assert isinstance(gmm.niter, int) == True
    assert isinstance(gmm.tol, float) == True
    assert isinstance(gmm.random_state, int) == True
    assert isinstance(gmm.scale, bool) == True
    assert isinstance(gmm.threshold, float) == True
    assert isinstance(gmm.scaler, RobustScaler) == True
    assert isinstance(gmm.gmm_pop, GaussianMixture) == True
    assert isinstance(gmm.gmm_train, GaussianMixture) == True
    assert isinstance(gmm.mixture_samples, dict) == True
    assert callable(gmm.preprocess) == True
    assert callable(gmm.fit) == True
    assert callable(gmm.population) == True
    assert callable(gmm.train) == True
    assert callable(gmm.divide) == True

    # Check the methods
    assert gmm.ncomp == 10
    assert gmm.niter == 100
    assert gmm.tol == 1e-3
    assert gmm.random_state == 0
    assert gmm.scale == True
    assert gmm.threshold == 0.5


def test_GMMbasic_divide():
    # Create a GMMsk object with some random data
    gmm = GMMsk(
        X_pop=np.random.randn(100, 10),
        X_train=np.random.randn(100, 10),
        ncomp=10,
        threshold=0.5,
        niter=100,
        tol=1e-3,
    )

    # Divide the data
    gmm.divide(np.random.randn(100, 10))

    # Check the attributes
    assert hasattr(gmm, "mixture_samples") == True
    assert isinstance(gmm.mixture_samples, dict) == True
    assert len(gmm.mixture_samples) == 10

    # Check the type of the attributes
    for mx in range(gmm.ncomp):
        assert isinstance(gmm.mixture_samples[mx], np.ndarray) == True
        assert gmm.mixture_samples[mx].shape[1] == 10
        assert gmm.mixture_samples[mx].shape[0] <= 100
