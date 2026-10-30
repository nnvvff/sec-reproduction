"""Small offline end-to-end demo; not a paper benchmark."""
import numpy as np
from sklearn.datasets import make_blobs
from threadpoolctl import threadpool_limits
from sec import SpectralEmbeddedClustering
from sec.metrics import clustering_accuracy


def main():
    X, y = make_blobs(n_samples=300, centers=3, n_features=16, cluster_std=.5, random_state=42)
    order = np.random.default_rng(42).permutation(len(X))
    train, test = order[:180], order[180:]
    with threadpool_limits(limits=2):
        for variant in ("sc", "lr"):
            model = SpectralEmbeddedClustering(3, variant=variant, mu=1,
                                               lr_include_self=True, n_init=10).fit(X[train])
            print(f"SEC/{variant.upper()} train ACC={clustering_accuracy(y[train], model.labels_):.3f}, "
                  f"unseen ACC={clustering_accuracy(y[test], model.predict(X[test])):.3f}, "
                  f"eigen residual={model.eigen_residual_:.2e}")


if __name__ == "__main__":
    main()
