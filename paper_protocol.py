"""Literal dataset sizes, fractions and selected hyperparameters (TNN 2011 Tables I/IV)."""

PROTOCOL = {
    "ar": dict(shape=(840, 768), clusters=120, fraction=.6, mu_sc=1e6, mu_lr=1e6, dkm_gamma=1e-6, clgr_eta=1e15),
    "yaleb": dict(shape=(2414, 1024), clusters=38, fraction=.6, mu_sc=1e9, mu_lr=1e9, dkm_gamma=1, clgr_eta=1e15),
    "pie": dict(shape=(3329, 1024), clusters=68, fraction=.6, mu_sc=1e6, mu_lr=1e9, dkm_gamma=1, clgr_eta=1e12),
    "mpeg7": dict(shape=(1400, 6000), clusters=70, fraction=.6, mu_sc=1e12, mu_lr=1e9, dkm_gamma=1e-9, clgr_eta=1),
    "coil20": dict(shape=(1440, 16384), clusters=20, fraction=.6, mu_sc=1e6, mu_lr=1e-9, dkm_gamma=1e-9, clgr_eta=1e-9),
    "optdigits": dict(shape=(5620, 64), clusters=10, fraction=.6, mu_sc=1e-6, mu_lr=1e-3, dkm_gamma=1e-6, clgr_eta=1),
    "usps": dict(shape=(9298, 256), clusters=10, fraction=.2, mu_sc=1, mu_lr=1e-9, dkm_gamma=1e-3, clgr_eta=1e3),
    "mnist": dict(shape=(70000, 784), clusters=10, fraction=.05, mu_sc=1e-3, mu_lr=1e-9, dkm_gamma=1e-9, clgr_eta=1e3),
}

# Columns: KM, DKM, KM-r, SC, LL, CLGR, SEC/SC, SEC/LR; mean ACC (%).
PAPER_TRAIN = {
    "ar": [36.3,43.6,59.7,42.4,47.9,48.1,60.4,60.3],
    "yaleb": [14.1,39.7,47.3,34.5,32.6,37.7,47.5,48.6],
    "pie": [20.1,55.5,68.0,39.6,41.1,50.2,66.6,67.0],
    "mpeg7": [47.9,48.2,56.8,55.7,54.8,55.9,56.5,56.6],
    "coil20": [68.0,61.3,62.3,77.5,68.9,78.2,79.0,78.0],
    "optdigits": [85.3,85.8,78.5,86.4,88.7,86.4,86.6,90.5],
    "usps": [66.3,71.0,67.4,65.2,63.9,65.4,70.8,74.2],
    "mnist": [53.2,54.2,51.6,61.7,62.8,67.6,61.7,71.0],
}
# Same order; SC/LL/CLGR test columns refer to their SEC ridge extensions.
PAPER_TEST = {
    "ar": [33.6,26.4,66.6,43.0,47.3,47.4,67.1,67.1],
    "yaleb": [13.0,20.8,43.5,35.9,35.0,35.6,42.5,42.8],
    "pie": [19.2,29.8,65.4,45.3,48.6,49.4,64.0,64.4],
    "mpeg7": [47.6,47.6,55.7,55.2,54.7,55.2,55.7,55.7],
    "coil20": [61.6,61.8,76.6,68.9,77.4,77.4,78.4,77.2],
    "optdigits": [85.4,85.8,78.9,85.8,88.2,85.9,86.0,90.0],
    "usps": [66.0,68.4,67.1,61.5,60.2,61.8,68.1,70.3],
    "mnist": [52.8,53.6,51.8,55.1,54.7,59.7,55.1,62.9],
}
METHOD_COLUMNS = ["KMeans", "DKM", "KM-r", "SC+ridge", "LL-linear", "CLGR-linear", "SEC/SC", "SEC/LR"]
SWEEP_VALUES = [1e-9,1e-6,1e-3,1.,1e3,1e6,1e9,1e12,1e15]
