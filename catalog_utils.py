"""
catalog_utils.py
================
Everything needed to build the DESI DR1 absorber catalogue with the
fine-tuned continuum autoencoder (``DESI_abs_finetuned_ewconstrained.h5``),
as ``desi_catalog_finetuned.py`` does, in one self-contained module.

Pipeline, per spectrum (``process_chunk``)
------------------------------------------
1. **Continuum** (``predict_continuum``): the fine-tuned dense autoencoder
   predicts the continuum of ``flux / norm`` (masked pixels set to 0); the
   prediction is multiplied back by ``norm`` and smoothed with a
   ``POST_MEDIAN_WINDOW``-pixel running median.
2. **Line detection** (``detect_lines``): the U-Net labels each pixel of the
   continuum-normalized flux ``F / C`` as continuum (0) or absorption (1).
3. **Continuum error** (``cont_frac_err_vs_snr``): a fractional continuum
   error per spectrum, interpolated in log S/N from the noise-degradation
   test of the fine-tuned autoencoder.
4. **Absorbers** (``classify_absorbers``): each contiguous run of absorption
   pixels is one region; its centroid, EW, EW error and S/N_line are
   measured and the regions are classified into ions and transitions by
   redshift coincidence (velocity tolerance ``V_TOL``), with the
   equivalent-width ordering / ratio constraints of the curve of growth
   (significance margin ``EW_SIGMA_MARGIN``) -- see
   ``enforce_transition_strength_ew_ordering``.

``build_catalog`` runs these steps in chunks over the parent spectra file and
writes the two outputs of ``desi_catalog_finetuned.py`` (spectra file with
continuum and U-Net labels, and the catalogue ``pred_df`` + ``spectra``).

Inputs (``PATHS``, all in this directory)
-----------------------------------------
- models/autoencoder_finetuned_nmf.h5: fine-tuned continuum autoencoder
  (Zenodo: https://doi.org/10.5281/zenodo.24500565)
- models/best_unet_nonoise_snr0_longrun.h5: U-Net line finder
  (Zenodo: https://doi.org/10.5281/zenodo.22711575, file UNet.h5)
- data/rest_wave.h5: QSO rest-frame wavelength grid (16384 pixels, 1230-3095 A)
- data/filtered_ion_lams.pkl: transitions to classify (ion -> list of astropy
  Quantity rest wavelengths, the transitions of the training mocks)
- data/desi_dr1_subsample500.h5: 500 DESI DR1 quasar spectra, an example of
  the input spectra file (default of read_spectra / build_catalog)

Input spectra file (``read_spectra``; build your own to make a catalogue)
--------------------------------------------------------------------------
HDF5 with, for n spectra on the rest_wave grid (QSO rest frame, linear
resampling; flux in any unit):
  ids (n,) int: identifier (TARGETID), the ``ID`` of the catalogue
  zqso (n,): quasar redshift
  norm (n,): median flux in 1425-1475 A (rest frame)
  flux, ivar (n, 16384): flux and inverse variance (the ivar propagated
    through the resampling), any value on masked pixels
  nan_mask (n, 16384) bool: True on pixels outside the observed range or bad
  rest_wave (16384,) optional: the grid (data/rest_wave.h5 otherwise)
The networks were trained for DESI quasars at 2.5 < z < 4 on this grid.

The classifier (``enforce_transition_strength_ew_ordering``) and its helper
``find_predicted_regions`` are verbatim copies of ``DESI/utils_analysis.py``.
"""
import os
import pickle
import time

import h5py
import numpy as np
import pandas as pd
from astropy import constants
from scipy.ndimage import median_filter

# ============================================================
# Configuration
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))

# every input lives in this directory (models/, data/), so the module and the
# notebook work wherever the directory is copied
PATHS = dict(
    wavelength=os.path.join(HERE, "data", "rest_wave.h5"),                  # dataset rest_wave
    autoencoder=os.path.join(HERE, "models", "autoencoder_finetuned_nmf.h5"),
    unet=os.path.join(HERE, "models", "best_unet_nonoise_snr0_longrun.h5"),
    ion_lams=os.path.join(HERE, "data", "filtered_ion_lams.pkl"),
    ion_colors=os.path.join(HERE, "data", "ion_color_map.pkl"),
    # 500 random DESI DR1 spectra (make_subsample.py), in the input format of read_spectra
    subsample=os.path.join(HERE, "data", "desi_dr1_subsample500.h5"),
)

# datasets of the parent file linked (not copied) into the spectra output
LINK_FIELDS = ["ids", "flux", "flux_scaled", "ivar", "nan_mask", "norm", "wave_obs", "zqso"]

CHUNK = 10000          # spectra per chunk of build_catalog
PRED_BATCH = 128       # batch size of the autoencoder / U-Net predictions
POST_MEDIAN_WINDOW = 10   # pixels of the running median on the predicted continuum

# classification settings
V_TOL = 250.0          # km/s, redshift tolerance of the line associations
EW_SIGMA_MARGIN = 3.0  # sigma margin of the EW ordering / ratio tests

# fractional continuum error (1 sigma) vs median S/N per pixel, from the
# noise-degradation test of the fine-tuned autoencoder (degradation_test.py:
# 1.4826 x MAD over spectra of median(C / C_ref) - 1)
CONT_ERR_SNR = np.array([0.48, 0.96, 1.93, 2.91, 4.90, 13.37])
CONT_ERR = np.array([0.0298, 0.0162, 0.0096, 0.0078, 0.0065, 0.0052])

C_KMS = constants.c.to("km/s").value


# ============================================================
# TensorFlow and models
# ============================================================
def configure_tensorflow():
    """
    Import and configure TensorFlow as in the production run and return it.

    - TF32 disabled: on Ampere GPUs TF32 matmuls (~10-bit mantissa) change the
      continuum by up to ~0.5-2.5% per pixel and from run to run; full float32
      keeps the predictions reproducible.
    - GPU memory growth on, so several processes can share a GPU.
    """
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:     # already initialized
            pass
    return tf


def _unet_custom_objects():
    """Loss and metric the U-Net was compiled with (NNs_arch.py), needed by
    tf.keras.models.load_model even when the model is only used to predict."""
    import tensorflow as tf
    from tensorflow import keras
    import keras.backend as K

    class WeightedSCCE(keras.losses.Loss):
        """Sparse categorical cross-entropy with per-class weights."""

        def __init__(self, class_weight, from_logits=False, name="weighted_scce",
                     reduction=keras.losses.Reduction.NONE):
            self.from_logits = from_logits
            if class_weight is None or all(v == 1.0 for v in class_weight):
                self.class_weight = None
            else:
                self.class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
            self.name = name
            self.reduction = reduction
            self.unreduced_scce = keras.losses.SparseCategoricalCrossentropy(
                from_logits=self.from_logits, name=name, reduction=self.reduction)

        def __call__(self, y_true, y_pred, sample_weight=None):
            loss = self.unreduced_scce(y_true, y_pred, sample_weight)
            if self.class_weight is not None:
                loss = loss * tf.cast(tf.gather(self.class_weight, tf.cast(y_true, tf.int32)), tf.float32)
            return loss

        def get_config(self):
            config = super().get_config()
            config.update({
                "class_weight": None if self.class_weight is None else self.class_weight.numpy().tolist(),
                "from_logits": self.from_logits,
                "reduction": self.reduction,
            })
            return config

        @classmethod
        def from_config(cls, config):
            class_weight = config.pop("class_weight", None)
            if class_weight is not None:
                class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
            return cls(class_weight=class_weight, **config)

    def F1_score(y_true, y_pred):
        """F1 of the absorption class (argmax of the U-Net output)."""
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        FP = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
        FN = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
        precision = TP / (TP + FP + K.epsilon())
        recall = TP / (TP + FN + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    mse = keras.losses.MeanSquaredError()
    return {"mse": mse, "loss": mse, "WeightedSCCE": WeightedSCCE, "F1_score": F1_score}


def load_models(autoencoder_path=PATHS["autoencoder"], unet_path=PATHS["unet"]):
    """
    The fine-tuned continuum autoencoder and the line-detection U-Net.

    Returns
    -------
    autoencoder, unet : tf.keras.Model
        autoencoder: (n, 16384) flux / norm -> (n, 16384) continuum / norm
        unet:        (n, 16384) F / C        -> (n, 16384, 2) class probabilities
    """
    tf = configure_tensorflow()
    autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
    unet = tf.keras.models.load_model(unet_path, custom_objects=_unet_custom_objects())
    return autoencoder, unet


# ============================================================
# Inputs
# ============================================================
def load_wavelength(path=PATHS["wavelength"]):
    """QSO rest-frame wavelength grid [A] of the spectra (16384 pixels, 1230-3095 A,
    step ~0.114 A); the same grid as the parent spectra file."""
    with h5py.File(path, "r") as f:
        return f["rest_wave"][:]


def load_ion_lams(path=PATHS["ion_lams"]):
    """Transitions to classify: dict ion -> list of astropy Quantity rest wavelengths
    (e.g. 'CIV': [1548.204 A, 1550.781 A]), the transitions present in the training mocks."""
    with open(path, "rb") as f:
        return pickle.load(f)


def n_parent_spectra(path=PATHS["subsample"]):
    """Number of spectra (rows) of the parent file."""
    with h5py.File(path, "r") as f:
        return f["flux"].shape[0]


def read_spectra(rows, path=PATHS["subsample"]):
    """
    Rows `rows` (sorted indices) of the parent spectra file.

    Returns a dict with
      ids (n,) int64, zqso (n,), norm (n,): TARGETID, redshift and the
        normalization (median flux in 1425-1475 A, desi_dataprocessing.py)
      flux, ivar (n, 16384): rest-frame resampled flux and inverse variance
        (the ivar propagated through the resampling, resample_ivar of
        desi_dataprocessing.py), NaN on masked pixels
      nan_mask (n, 16384) bool: pixels outside the observed range or masked
    """
    rows = np.asarray(rows)
    with h5py.File(path, "r") as f:
        sel = slice(rows[0], rows[-1] + 1) if np.all(np.diff(rows) == 1) else rows
        out = dict(
            ids=f["ids"][sel].astype(np.int64).ravel(),
            zqso=f["zqso"][sel].astype(np.float32).ravel(),
            norm=f["norm"][sel].astype(np.float32).ravel(),
            flux=f["flux"][sel].astype(np.float32),
            ivar=f["ivar"][sel].astype(np.float32),
            nan_mask=f["nan_mask"][sel].astype(bool),
        )
    out["flux"][out["nan_mask"]] = np.nan
    out["ivar"][out["nan_mask"]] = np.nan
    return out


# ============================================================
# Steps
# ============================================================
def calculate_snr(flux, ivar):
    """
    Median S/N per pixel of each spectrum: median of flux * sqrt(ivar) after
    clipping each spectrum to its own 1st-99th percentile range (robust to
    spikes). This is the spectrum S/N that sets the continuum error and of the
    spectrum-S/N cuts of the analysis.
    """
    s = np.sqrt(np.maximum(ivar, 0)) * flux
    lo, hi = np.nanpercentile(s, 1, axis=1), np.nanpercentile(s, 99, axis=1)
    return np.nanmedian(np.clip(s, lo[:, None], hi[:, None]), axis=1)


def cont_frac_err_vs_snr(snr):
    """
    1-sigma fractional continuum error of the fine-tuned autoencoder at
    spectrum S/N `snr`: CONT_ERR interpolated in log S/N, constant beyond the
    measured S/N range (3% at S/N 0.5 down to 0.5% at S/N > 13).
    """
    x = np.log(np.clip(snr, CONT_ERR_SNR[0], CONT_ERR_SNR[-1]))
    return np.interp(x, np.log(CONT_ERR_SNR), CONT_ERR)


def predict_continuum(autoencoder, flux, norm, nan_mask, batch_size=PRED_BATCH):
    """
    Continuum of each spectrum with the fine-tuned autoencoder.

    The network input is flux / norm with 0 on masked (non-finite) pixels; its
    output, continuum / norm, is multiplied back by norm and smoothed with a
    POST_MEDIAN_WINDOW-pixel running median (edges: nearest value).

    Returns
    -------
    cont_scaled : (n, 16384) raw network output (continuum / norm), NaN on masked pixels
    continuum   : (n, 16384) float32 continuum in flux units, NaN on masked pixels
    """
    x = flux / norm[:, None]
    x[~np.isfinite(x)] = 0.0
    cont_scaled = autoencoder.predict(x, batch_size=batch_size, verbose=0)
    continuum = median_filter(cont_scaled * norm[:, None], size=(1, POST_MEDIAN_WINDOW), mode="nearest")
    continuum[nan_mask] = np.nan
    cont_scaled[nan_mask] = np.nan
    return cont_scaled, continuum.astype(np.float32)


def normalize(flux, ivar, continuum, nan_mask):
    """
    Continuum-normalized flux F / C and its 1-sigma error sigma / C
    (sigma = 1 / sqrt(ivar), ivar floored at 1e-10), NaN on masked pixels.
    """
    err = np.sqrt(1.0 / np.clip(ivar, 1e-10, None))
    flux_n, noise_n = flux / continuum, err / continuum
    flux_n[nan_mask] = np.nan
    noise_n[nan_mask] = np.nan
    return flux_n, noise_n


def detect_lines(unet, flux_n, nan_mask, batch_size=PRED_BATCH):
    """
    U-Net pixel labels of the continuum-normalized flux: argmax of the two
    class probabilities, 0 = continuum, 1 = absorption, NaN on masked pixels.
    flux_n is passed as is (NaN on masked pixels), as in the production run.
    """
    import tensorflow as tf
    labels = tf.argmax(unet.predict(flux_n, batch_size=batch_size, verbose=0), axis=-1).numpy()
    labels = labels.astype(np.float32)
    labels[nan_mask] = np.nan
    return labels


def classify_absorbers(ids, flux_n, noise_n, wavelength, zqso, labels, ion_lams, cont_frac_err,
                       v_tol=V_TOL, ew_sigma_margin=EW_SIGMA_MARGIN):
    """
    Absorber catalogue of the labelled spectra: enforce_transition_strength_ew_ordering
    (per-spectrum continuum error, fully correlated between the regions of a
    pair). One row per labelled region; see the classifier's docstring and
    CATALOG_COLUMNS for the columns.

    v_tol: velocity tolerance [km/s] of the line associations: transitions
      whose redshifts agree within v_tol / c are matched into one system
      (production: V_TOL = 250 km/s).
    ew_sigma_margin: significance [sigma] by which a pair must violate the EW
      ordering / ratio constraints to be rejected (production: EW_SIGMA_MARGIN = 3).
    """
    return enforce_transition_strength_ew_ordering(
        ids, flux_n, noise_n, wavelength, zqso, labels, ion_lams,
        tol=v_tol / C_KMS, ew_sigma_margin=ew_sigma_margin, cont_frac_err=cont_frac_err,
    )


CATALOG_COLUMNS = {
    "ID": "DESI TARGETID of the spectrum",
    "zqso": "quasar redshift",
    "lam": "observed-frame centroid of the region [A]",
    "start, end": "first / last pixel of the region on the rest-frame grid (valid pixels only)",
    "lam05, lam95": "observed-frame wavelength of the region edges [A]",
    "ion_pred, wl_pred": "classified ion and floor(rest wavelength) of the transition (None / NaN: unclassified)",
    "zabs": "absorber redshift of the classified transition",
    "score": "redshift-match x completeness score of the classification",
    "SNR_line": "EW / dEW",
    "EW, dEW": "equivalent width and its error, absorber rest frame if classified (else QSO rest frame) [A]",
    "dEW_stat, dEW_cont": "photon-noise and continuum parts of dEW (added in quadrature)",
    "EW_qso, dEW_qso": "EW and dEW in the QSO rest frame [A]",
}


def process_chunk(rows, autoencoder, unet, wavelength, ion_lams, cont_frac_err=None,
                  path=PATHS["subsample"], v_tol=V_TOL, ew_sigma_margin=EW_SIGMA_MARGIN):
    """
    All the steps for the parent rows `rows`.

    cont_frac_err: None for the S/N-dependent continuum error (production), or
    a constant fractional error for every spectrum.
    v_tol, ew_sigma_margin: classifier settings (classify_absorbers).

    Returns continuum, cont_scaled, labels (n, 16384), the catalogue rows
    (DataFrame) and the per-spectrum table (TARGETID, snr, cont_frac_err).
    """
    d = read_spectra(rows, path)
    snr = calculate_snr(d["flux"], d["ivar"])
    cont_scaled, continuum = predict_continuum(autoencoder, d["flux"], d["norm"], d["nan_mask"])
    flux_n, noise_n = normalize(d["flux"], d["ivar"], continuum, d["nan_mask"])
    labels = detect_lines(unet, flux_n, d["nan_mask"])
    cfe = cont_frac_err_vs_snr(snr) if cont_frac_err is None else np.full(len(snr), cont_frac_err)
    df = classify_absorbers(d["ids"], flux_n, noise_n, wavelength, d["zqso"], labels, ion_lams, cfe,
                            v_tol=v_tol, ew_sigma_margin=ew_sigma_margin)
    spectra = pd.DataFrame({"TARGETID": d["ids"], "snr": snr, "cont_frac_err": cfe})
    return continuum, cont_scaled, labels, df, spectra


def build_catalog(spectra_out, catalog_out, n_max=None, chunk=CHUNK, cont_frac_err=None,
                  parent=PATHS["subsample"], v_tol=V_TOL, ew_sigma_margin=EW_SIGMA_MARGIN):
    """
    The full production run of desi_catalog_finetuned.py.

    Writes
      spectra_out : continuum, continuum_scaled, labels_pred (n, 16384) plus external
                    links to the parent datasets LINK_FIELDS (same dataset names
                    as the parent file)
      catalog_out : pred_df (one row per region) and spectra (TARGETID, snr,
                    cont_frac_err), pandas HDF5

    n_max: only the first n_max spectra (tests). v_tol [km/s], ew_sigma_margin:
    classifier settings (classify_absorbers), stored as attributes of both
    outputs. Returns pred_df, spectra.
    """
    with h5py.File(parent, "r") as fp:   # the grid is stored in the subsample, not in the parent file
        wavelength = load_wavelength(parent if "rest_wave" in fp else PATHS["wavelength"])
    autoencoder, unet = load_models()
    ion_lams = load_ion_lams()
    n = n_parent_spectra(parent) if n_max is None else min(n_max, n_parent_spectra(parent))
    npix = wavelength.size
    # row-chunked, lossless (gzip + byte shuffle) 2D datasets, as h5_io.create
    kw = dict(chunks=(min(16, n), npix), compression="gzip", compression_opts=4, shuffle=True)
    t0 = time.time()
    dfs, spectra = [], []
    with h5py.File(spectra_out, "w") as fout:
        with h5py.File(parent, "r") as fp:
            present = set(fp)
        for k in LINK_FIELDS:
            if k in present:          # the subsample has no flux_scaled, wave_obs
                fout[k] = h5py.ExternalLink(parent, k)
        ds = {name: fout.create_dataset(name, shape=(n, npix), dtype=np.float32, **kw)
              for name in ("continuum", "continuum_scaled", "labels_pred")}
        fout.attrs.update(continuum_model=os.path.basename(PATHS["autoencoder"]),
                          unet_model=os.path.basename(PATHS["unet"]), parent_file=os.path.abspath(parent),
                          v_tol=v_tol, ew_sigma_margin=ew_sigma_margin)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            t1 = time.time()
            cont, cont_scaled, labels, df, sp = process_chunk(
                np.arange(i0, i1), autoencoder, unet, wavelength, ion_lams, cont_frac_err, parent,
                v_tol, ew_sigma_margin)
            ds["continuum"][i0:i1] = cont
            ds["continuum_scaled"][i0:i1] = cont_scaled
            ds["labels_pred"][i0:i1] = labels
            dfs.append(df)
            spectra.append(sp)
            print(f"spectra {i0}-{i1} of {n}: {len(df)} regions ({time.time() - t1:.0f} s)", flush=True)
    pred_df = pd.concat(dfs, ignore_index=True)
    spectra = pd.concat(spectra, ignore_index=True)
    summarize(pred_df)
    pred_df.to_hdf(catalog_out, key="pred_df", mode="w")
    spectra.to_hdf(catalog_out, key="spectra", mode="a")
    with pd.HDFStore(catalog_out, "a") as store:    # classifier settings of this catalogue
        store.get_storer("pred_df").attrs.v_tol = v_tol
        store.get_storer("pred_df").attrs.ew_sigma_margin = ew_sigma_margin
    print(f"Catalogue saved to {catalog_out} ({len(pred_df)} regions, {time.time() - t0:.0f} s)")
    return pred_df, spectra


def summarize(pred_df):
    """
    Checks of desi_catalog_finetuned.py on a catalogue: the share of the
    continuum term in the EW error, the absorber-frame EW conversion, and the
    most frequent classified ions. Returns a dict of the numbers printed.
    """
    R = (pred_df["dEW_cont"] / pred_df["dEW_stat"]).replace([np.inf, -np.inf], np.nan)
    p16, p50, p84 = np.nanpercentile(R, [16, 50, 84])
    has_z = pred_df["zabs"].notna()
    frame = (1 + pred_df["zqso"]) / (1 + pred_df["zabs"])
    assert np.allclose(pred_df.loc[has_z, "EW"], pred_df.loc[has_z, "EW_qso"] * frame[has_z], equal_nan=True)
    out = dict(n_regions=len(pred_df), n_spectra=pred_df["ID"].nunique(),
               dEW_cont_over_stat_median=p50, dEW_cont_over_stat_16_84=(p16, p84),
               frac_cont_dominated=float(np.nanmean(R > 1)), frac_classified=float(has_z.mean()))
    print(f"{out['n_regions']} regions in {out['n_spectra']} spectra; classified {out['frac_classified']:.1%}")
    print(f"dEW_cont / dEW_stat: median {p50:.2f} (16-84%: {p16:.2f}-{p84:.2f}), "
          f"> 1 for {out['frac_cont_dominated']:.1%} of the regions")
    print(pred_df["ion_pred"].value_counts().head(10).to_string())
    return out


# ============================================================
# Classifier: verbatim copies of DESI/utils_analysis.py
# (find_predicted_regions, enforce_transition_strength_ew_ordering)
# ============================================================
def find_predicted_regions(mask):
    mask = np.asarray(mask, dtype=bool)  # NaNs become False, which is fine
    diff = np.diff(mask.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask) - 1]

    return list(zip(starts, ends))


def enforce_transition_strength_ew_ordering(
    ids, flux, noise, wave, zqso, labels_pred,
    ion_lams, tol=0.01, ew_sigma_margin=2.0, cont_frac_err=0.0,
    cont_corr_length=np.inf
):
    """
    Independent, self-contained twin of centroids_classification: runs
    the same centroid-finding / SNR-EW / multiplet-classification
    pipeline from the same raw inputs (it does not call
    centroids_classification and does not consume its output -- the
    two functions share no runtime dependency, so a change to one
    cannot silently break the other), and additionally enforces the
    equivalent-width ordering implied by transition strength.

    Unlike centroids_classification, this function is deterministic --
    there is no bootstrap resampling of the flux/continuum (that
    machinery aggregated candidate classifications across draws, which
    is exactly what the running "keep only the current best match"
    update() below already does within a single deterministic pass,
    since with one draw there is only ever one candidate per region;
    the bootstrap loop added resampling overhead without changing the
    result). `score` in the output is therefore the raw redshift-match
    x completeness score, not a bootstrap-confidence-weighted one.

    Physical justification for the added EW constraint: for a curve of
    growth at fixed column density N and Doppler b, the optical depth
    of a resonance transition scales at every velocity as
    tau(v) ~ N f lambda_rest phi(v), where phi(v) is the (shared)
    velocity profile of the absorbing gas. So for two transitions of
    the SAME ion in the SAME physical system (same N, b, phi), the one
    with the larger f * lambda_rest has tau(v) >= the other's at every
    v, and therefore W_v = integral[1 - exp(-tau(v))] dv is also >=
    the other's (in velocity units; see below for the EW-ratio limits
    in wavelength units) -- this holds regardless of saturation, unlike the
    optically-thin-only relation "EW proportional to f". For CIV this
    reduces to the familiar EW(1548) >= EW(1550), since
    f_1548 * 1548 > f_1550 * 1550; the same reasoning generalises to
    any multi-transition ion in `ion_lams` (MgII, SiIV, FeII, SII,
    SiII, NiII, MnII, ...).

    A possible doublet/multiplet match is only accepted if the
    redshift condition (|Δz| <= tol) and this EW ordering are BOTH
    true at the same time -- checked jointly, right where the match is
    found, in the "ALL MULTIPLETS" loop below (this is what DEFINES a
    doublet/multiplet match). The same joint check also gates "ALL
    LINES" (every other transition assigned to a system purely from
    redshift proximity to it, once "ALL MULTIPLETS" has anchored the
    system's redshift) and every post-aggregation correction further
    down, so no stage can ever create or extend a system that violates
    it.
    Systems (grouped by same spectrum, zabs within `tol`) that still
    end up inconsistent are represented as unclassified
    (ion_pred/wl_pred/zabs_pred/score_pred = None/NaN), the same way a
    region that never matched anything is represented. Rows are never
    dropped, consistent with how the rest of this function always
    emits one row per detected region.

    The allowed range of the EW ratio follows from the curve of growth.
    EWs here are in wavelength units, W_lambda = (lambda / c) W_v, so for
    a stronger transition s and a weaker one w of the same ion:
      - optically thick (both saturated, same velocity profile, equal
        W_v): W_s / W_w = lambda_s / lambda_w
      - optically thin (W_lambda ~ N f lambda^2):
        W_s / W_w = f_s lambda_s^2 / (f_w lambda_w^2)
    and a genuine pair must lie between the two (`cog_limits`). The
    thick limit replaces the bare "EW_s >= EW_w" ordering (identical for
    CIV, where lambda_s / lambda_w = 0.998, but not for widely spaced
    transitions, e.g. 0.70 for SiII 1260 / 1808); the thin limit is the
    ceiling that rejects confusable *cross-ion* pairs such as CIV
    1548/1550 vs. OI 1302 + SiII 1304, whose ratio has no such bound.

    Both limits are tested against MEASURED EWs with their errors
    (`ew_limits_ok`): a pair is only rejected if EW_s - k EW_w falls
    outside the limit by more than `ew_sigma_margin` standard
    deviations of that difference, so noise on nearly degenerate
    transitions does not veto genuine matches.

    The check is applied every time a doublet/multiplet is defined or
    extended: in ALL MULTIPLETS (the pair defining the match), in ALL
    LINES (a line joining an anchored system is checked against every
    transition of the same ion already assigned there), in every
    post-aggregation correction, and in a FINAL PASS that re-checks
    every pair of distinct transitions of each ion system (same ion,
    zabs within `tol`), since a system assembled from several matches
    can hold pairs no earlier stage compared. A system with any pair
    outside the limits is unclassified.

    `flux` and `noise` must already be continuum-normalized (F / C and
    sigma / C, with C the fitted continuum): no continuum division is
    done here.

    `cont_frac_err` (default 0.0, i.e. photon noise only) is the 1-sigma
    fractional error of the continuum fit: a scalar, or an array with one
    value per spectrum (e.g. S/N dependent). Over a single absorption
    region the continuum error is a coherent offset shared by all its
    pixels, not independent per-pixel noise, so it is added once per
    region, dEW_cont = cont_frac_err * sum(F/C) * dlambda
    (~ cont_frac_err * (window width - EW)), in quadrature with the
    photon-noise dEW. It therefore propagates to `SNR_line`, `dEW` and
    every `ew_sigma_margin` test. The two components are also returned
    separately as `dEW_stat` and `dEW_cont`.

    In the pairwise `ew_sigma_margin` tests the continuum error of the
    two regions is NOT treated as independent: the same continuum
    misplacement hits both lines of a close doublet, so it partly
    cancels in EW_strong - k * EW_weak. With S = sum(F/C) * dlambda per
    region and correlation rho between the two regions' continuum
    errors, the continuum variance of the difference is
        eps^2 * [(1 - rho) * (S_a^2 + k^2 S_b^2) + rho * (S_a - k S_b)^2],
    added to the independent photon-noise variances. rho =
    exp(-|lam_a - lam_b| / cont_corr_length), with lam the observed
    centroids in Angstrom; the default cont_corr_length=np.inf gives
    rho = 1 (fully coherent, appropriate for doublets a few Angstrom
    apart), and cont_corr_length=0 recovers the independent-errors
    quadrature sum.

    `noise` must be the 1-sigma error of each pixel of `wave` as
    propagated through the resampling onto `wave` (ivar_interp from
    desi_dataprocessing.py:resample_ivar), not the
    native-pixel sigma interpolated onto it: the photon-noise dEW is
    sqrt(sum(n^2)) * dlambda with no oversampling correction.

    Output `EW`/`dEW`/`dEW_stat`/`dEW_cont` are rest-frame of the
    absorber (W_r = W_qso (1+zqso)/(1+zabs)) for regions with a zabs, and
    QSO rest-frame for unclassified regions; the QSO-frame values are
    always available as `EW_qso`/`dEW_qso`.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from linetools.lists.linelist import LineList

    # ---------------------------
    # Helper: centroid
    # ---------------------------
    def centroid_for_region(flux_1d, wave_1d, noise_1d):
        flux_1d = np.maximum(flux_1d, 0.0)
        noise2 = np.maximum(noise_1d, 1e-12) ** 2

        # Suppress pixels where absorption is negligible (flux ≈ 1)
        min_absorption = 0.01  # tunable
        weights = np.where(
            (flux_1d < 1.0) & ((1.0 - flux_1d) > min_absorption),
            (1.0 - flux_1d) / noise2,
            0.0
        )
        weights = np.where(np.isnan(wave_1d), 0.0, weights)

        Wsum = np.sum(weights)
        num = np.nansum(weights * wave_1d)

        centroid = num / Wsum if Wsum > 0 else np.nanmedian(wave_1d)

        # Clip centroid to the wavelength range of the region
        wave_min, wave_max = np.nanmin(wave_1d), np.nanmax(wave_1d)
        return float(np.clip(centroid, wave_min, wave_max))

    # ---------------------------
    # Helper: SNR / EW
    # ---------------------------
    def snr_for_region(flux_1d, noise_1d, dlambda):
        f = np.where(np.isfinite(flux_1d), flux_1d, 1.0)
        n = np.where(np.isfinite(noise_1d), noise_1d, 0.0)

        absorption = np.clip(1.0 - f, 0.0, None)
        ew = 1.1 * np.nansum(absorption) * dlambda
        dew_stat = np.sqrt(np.nansum(n ** 2)) * dlambda

        # Coherent continuum-placement error (see docstring)
        cont_sum = np.nansum(np.clip(f, 0.0, None)) * dlambda
        dew_cont = cont_frac_err * cont_sum
        dew = np.sqrt(dew_stat ** 2 + dew_cont ** 2)

        snr = ew / dew
        if not np.isfinite(snr):
            snr = 0.0

        return snr, ew, dew, dew_stat, cont_sum

    # ---------------------------
    # Helper: system finder
    # ---------------------------
    def cluster_by_redshift(indices, zabs, tol):
        """Return list of lists of indices clustered by |Δz| <= tol."""
        if len(indices) == 0:
            return []

        order = np.argsort(zabs[indices])
        idx_sorted = indices[order]
        z_sorted = zabs[idx_sorted]

        systems = []
        current = [idx_sorted[0]]

        for i in range(1, len(idx_sorted)):
            if abs(z_sorted[i] - z_sorted[i-1]) <= tol:
                current.append(idx_sorted[i])
            else:
                systems.append(current)
                current = [idx_sorted[i]]
        systems.append(current)

        return systems

    # ---------------------------
    # Helper: transition strength (f * lambda_rest), for the added
    # EW-ordering constraint -- see the docstring above for the
    # physical derivation.
    # ---------------------------
    _ll_ism = LineList('ISM')

    def transition_strength(ion, wl_int):
        row = _ll_ism[f'{ion} {wl_int}']
        return float(row['f']) * float(row['wrest'].value)

    # ---------------------------
    # Helper: curve-of-growth limits on EW_strong / EW_weak for two
    # transitions of the same ion (indices into rest_by_ion[ion],
    # `idx_s` the stronger by f * lambda_rest). EWs are in wavelength
    # units: optically thick (saturated, same velocity profile, equal
    # W_v) gives lambda_s / lambda_w; optically thin (W ~ N f lambda^2)
    # gives f_s lambda_s^2 / (f_w lambda_w^2). See the docstring.
    # ---------------------------
    def cog_limits(ion, idx_s, idx_w):
        s = strength_by_ion[ion]
        lam_ratio = rest_by_ion[ion][idx_s] / rest_by_ion[ion][idx_w]
        return lam_ratio, (s[idx_s] / s[idx_w]) * lam_ratio

    # ---------------------------
    # Setup
    # ---------------------------
    rest_by_ion = {
        ion: np.array([float(q.value) for q in lst])
        for ion, lst in ion_lams.items()
    }
    LAM = {
        (ion, int(np.floor(float(q.value)))): float(q.value)
        for ion, lst in ion_lams.items()
        for q in lst
    }

    # f * lambda_rest for every tabulated transition, aligned index-for-
    # index with rest_by_ion[ion] -- precomputed once (rather than
    # inside the per-spectrum loop below) since it never depends on the
    # spectrum, only on `ion_lams`.
    strength_by_ion = {}
    for ion, rest in rest_by_ion.items():
        try:
            strength_by_ion[ion] = np.array([
                transition_strength(ion, int(np.floor(lam0))) for lam0 in rest
            ])
        except Exception as exc:
            print(f'Skipping EW-ordering check for {ion}: {exc}')
            strength_by_ion[ion] = None

    results = []

    # ===========================
    # Loop over spectra
    # ===========================
    cont_frac_err_all = np.broadcast_to(np.asarray(cont_frac_err, dtype=float), (len(flux),))
    for i in tqdm(range(len(flux)), desc="Processing spectra"):

        id = ids[i]
        F, n = flux[i], noise[i]
        z = zqso[i]
        mask = labels_pred[i]
        # per-spectrum continuum error, read by the nested helpers below
        cont_frac_err = float(cont_frac_err_all[i])

        valid = ~np.isnan(F)
        if not np.any(valid):
            continue

        F, n, mask = F[valid], n[valid], mask[valid]
        wave_sub = wave[valid]

        # flux is already continuum-normalized by the caller
        f = F

        # ---------------------------
        # Regions
        # ---------------------------
        regions = np.asarray(find_predicted_regions(mask), dtype=int)
        if len(regions) == 0:
            continue

        starts, ends = regions[:, 0], regions[:, 1]
        R = len(starts)

        f_regions = [f[s:e + 1] for s, e in zip(starts, ends)]
        w_regions = [wave_sub[s:e + 1] for s, e in zip(starts, ends)]
        n_regions = [n[s:e + 1] for s, e in zip(starts, ends)]

        # ---------------------------
        # Centroids
        # ---------------------------
        cmedian = np.array([
            centroid_for_region(fr, wr, nr)
            for fr, wr, nr in zip(f_regions, w_regions, n_regions)
        ])  # (R,)

        # Enforce centroid inside label region in wavelength space
        # (centroid_for_region already clips, so this only ever fires
        # if something upstream is inconsistent -- kept as a sanity
        # check, same as centroids_classification's own version).
        for r, (s, e) in enumerate(zip(starts, ends)):
            wmin, wmax = wave_sub[s], wave_sub[e]
            if not (wmin <= cmedian[r] <= wmax):
                print(f"Region {r} has a bad centroid: {cmedian[r]}")

        cmedian *= (1 + z)

        # ---------------------------
        # zabs
        # ---------------------------
        zabs_by_ion = {
            ion: np.where(
            (0 <= (zabs := cmedian[:, None] / rest[None, :] - 1.0)) &
            (zabs <= zqso[i]),
            zabs,
            np.nan
            )
            for ion, rest in rest_by_ion.items()
        }

        # ---------------------------
        # SNR / EW / dEW
        # ---------------------------
        snr_arr, ew_arr, dew_arr, stat_arr, csum_arr = [], [], [], [], []
        for fr, nr, wr in zip(f_regions, n_regions, w_regions):
            dl = np.nanmedian(np.diff(wr))
            s_, e_, d_, st_, cs_ = snr_for_region(fr, nr, dl)
            snr_arr.append(s_)
            ew_arr.append(e_)
            dew_arr.append(d_)
            stat_arr.append(st_)
            csum_arr.append(cs_)

        snr_arr  = np.array(snr_arr)   # (R,)
        ew_arr   = np.array(ew_arr)    # (R,)
        dew_arr  = np.array(dew_arr)   # (R,)
        stat_arr = np.array(stat_arr)  # (R,) photon-noise dEW
        csum_arr = np.array(csum_arr)  # (R,) sum(F/C) * dlambda

        # ---------------------------
        # Helper: 1-sigma uncertainty of EW[a] - k * EW[b] for regions
        # a, b (index arrays, broadcast like a bare comparison would):
        # photon noise is independent between regions, the continuum
        # error is correlated with rho = exp(-|dlam| / cont_corr_length)
        # (see the function docstring).
        # ---------------------------
        def pair_sigma(a, b, k):
            if cont_corr_length == 0:
                rho = 0.0
            else:
                rho = np.exp(-np.abs(cmedian[a] - cmedian[b]) / cont_corr_length)
            Sa, Sb = csum_arr[a], csum_arr[b]
            var_cont = cont_frac_err ** 2 * (
                (1.0 - rho) * (Sa ** 2 + (k * Sb) ** 2) + rho * (Sa - k * Sb) ** 2
            )
            return np.sqrt(stat_arr[a] ** 2 + (k * stat_arr[b]) ** 2 + var_cont)

        # ---------------------------
        # Helper: significance-aware curve-of-growth check between
        # region `a` (assigned to the stronger transition) and region
        # `b` (the weaker one): the ratio may only fall outside
        # [k_thick, k_thin] by more than `ew_sigma_margin` standard
        # deviations of EW[a] - k EW[b] (pair_sigma). Broadcasts over
        # index arrays like a bare comparison would.
        # ---------------------------
        def ew_limits_ok(a, b, k_thick, k_thin):
            thick_ok = (ew_arr[a] - k_thick * ew_arr[b]) >= \
                -ew_sigma_margin * pair_sigma(a, b, k_thick)
            thin_ok = (ew_arr[a] - k_thin * ew_arr[b]) <= \
                ew_sigma_margin * pair_sigma(a, b, k_thin)
            return thick_ok & thin_ok

        # ===========================
        # MATCHING
        # ===========================

        ion_pred   = np.full(R, None, dtype=object)
        tran_pred  = np.full(R, np.nan)
        wl_pred    = np.full(R, np.nan)
        score_pred = np.full(R, np.nan)
        zabs_pred  = np.full(R, np.nan)

        def update(mask, score, zvals, ion, lam0):
            better = mask & (np.isnan(score_pred) | (score > score_pred))
            score_pred[better] = score[better]
            zabs_pred[better]  = zvals[better]
            ion_pred[better]   = ion
            tran_pred[better]  = lam0
            wl_pred[better]    = np.floor(lam0)

        # ---------------------------
        # Shared helper for every stage below that proposes a NEW
        # classification (ALL MULTIPLETS, ALL LINES, and every
        # POST-AGGREGATION FIX): before a candidate (ion, wl_new) is
        # accepted for a region, check it against every OTHER region
        # already classified as this ion within `tol` of the same
        # redshift -- i.e. each pair's EW ratio must lie between the
        # optically thick and thin limits (cog_limits), within
        # `ew_sigma_margin` standard deviations (pair_ok). Companions are read off `ion_pred`/`wl_pred`
        # AT CALL TIME: `wl_pred` is updated by `update()` from the
        # very first ALL MULTIPLETS match onward (not just materialized
        # after ALL LINES finishes), and every POST-AGGREGATION FIX
        # below mutates it directly in lockstep with
        # `ion_pred`/`zabs_pred` -- so this always sees the live,
        # incrementally-updated classification state, regardless of
        # which stage is calling it.
        # ---------------------------
        def transition_idx(ion, wl_int):
            """Index of transition `wl_int` (floor-int convention) in
            rest_by_ion[ion], or None if its strength is unknown."""
            if strength_by_ion.get(ion) is None:
                return None
            for idx, lam0 in enumerate(rest_by_ion[ion]):
                if int(np.floor(lam0)) == int(wl_int):
                    return idx
            return None

        def pair_ok(ion, idx_a, ra, idx_b, rb):
            """True if regions `ra` and `rb`, assigned to transitions
            `idx_a` and `idx_b` of `ion`, have EWs within the optically
            thin/thick limits."""
            s = strength_by_ion[ion]
            if s[idx_a] < s[idx_b]:
                idx_a, ra, idx_b, rb = idx_b, rb, idx_a, ra
            k_thick, k_thin = cog_limits(ion, idx_a, idx_b)
            return bool(ew_limits_ok(ra, rb, k_thick, k_thin))

        def ew_ordering_ok(ion, wl_new, r_new, z_new):
            """True if assigning `wl_new` to region `r_new` (at
            redshift `z_new`) as `ion` is within the optically
            thin/thick limits with every OTHER region currently
            classified as this ion within `tol` of `z_new`."""
            idx_new = transition_idx(ion, wl_new)
            if idx_new is None:
                return True
            companions = np.where(
                (ion_pred == ion) & np.isfinite(wl_pred) &
                (np.abs(zabs_pred - z_new) <= tol)
            )[0]
            for rc in companions:
                if rc == r_new or int(wl_pred[rc]) == int(wl_new):
                    continue
                idx_c = transition_idx(ion, wl_pred[rc])
                if idx_c is None:
                    continue
                if not pair_ok(ion, idx_new, r_new, idx_c, rc):
                    return False
            return True

        # ---------- ALL MULTIPLETS ----------
        detected_z = []

        for ion, rest in rest_by_ion.items():
            n_lines = len(rest)
            if n_lines < 2:
                continue

            # Get redshifts for all lines in multiplet
            # zabs_by_ion[ion] has shape (R, n_lines)
            z_multiplet = zabs_by_ion[ion]  # (R, n_lines)
            strength = strength_by_ion.get(ion)

            # For each pair of regions (i, j), compute pairwise redshift differences
            # across all line combinations
            for line_idx in range(n_lines):
                z_line = z_multiplet[:, line_idx]  # (R,)

                # Compute differences with all other lines
                for other_idx in range(n_lines):
                    if line_idx == other_idx:
                        continue

                    z_other = z_multiplet[:, other_idx]  # (R,)

                    # dz shape: (R_i, R_j)
                    dz = np.abs(z_line[:, None] - z_other[None, :])
                    match = dz <= tol

                    # A possible doublet/multiplet match is only kept
                    # if the EW ratio of the two regions lies between
                    # the optically thick and thin limits (cog_limits,
                    # see the function docstring), within
                    # `ew_sigma_margin` standard deviations
                    # (ew_limits_ok).
                    # line_idx vs other_idx strength is fixed for this
                    # ion, so this is a single EW comparison broadcast
                    # over every (i, j).
                    if strength is not None:
                        r_i = np.arange(R)[:, None]  # (R_i, 1)
                        r_j = np.arange(R)[None, :]  # (1, R_j)
                        if strength[line_idx] >= strength[other_idx]:
                            k_thick, k_thin = cog_limits(ion, line_idx, other_idx)
                            match = match & ew_limits_ok(r_i, r_j, k_thick, k_thin)
                        else:
                            k_thick, k_thin = cog_limits(ion, other_idx, line_idx)
                            match = match & ew_limits_ok(r_j, r_i, k_thick, k_thin)

                    # Base score from redshift matching
                    base_score = np.where(match, 1.0 - dz / tol, np.nan)

                    # For each region pair, count how many lines match --
                    # this gives us the completeness of the multiplet
                    completeness = np.zeros((R, R))
                    for check_idx in range(n_lines):
                        z_check = z_multiplet[:, check_idx]
                        for ref_r in range(R):
                            dz_check = np.abs(z_check - z_line[ref_r])
                            completeness[ref_r, :] += (dz_check <= tol).astype(float)

                    # Normalize completeness: 1.0 when all lines match, 1/n_lines for single match
                    completeness_score = completeness / n_lines

                    # Combined score: base matching score * completeness bonus
                    score = base_score * completeness_score

                    # Update for primary line
                    best_score_i = np.nanmax(score, axis=1)
                    update(np.isfinite(best_score_i), best_score_i, z_line, ion, rest[line_idx])

                    # Update for matching line
                    best_score_j = np.nanmax(score, axis=0)
                    update(np.isfinite(best_score_j), best_score_j, z_other, ion, rest[other_idx])

            # Store detected redshifts (median across all matched line pairs)
            z_matches = []
            for line_idx in range(n_lines):
                for other_idx in range(line_idx + 1, n_lines):
                    z1 = z_multiplet[:, line_idx]
                    z2 = z_multiplet[:, other_idx]
                    dz = np.abs(z1[:, None] - z2[None, :])
                    match = dz <= tol

                    # Same thin/thick-limit gate as above, so a pair
                    # that isn't a physically valid doublet/multiplet
                    # also doesn't seed a "detected" system redshift.
                    if strength is not None:
                        r_1 = np.arange(R)[:, None]  # (R_i, 1)
                        r_2 = np.arange(R)[None, :]  # (1, R_j)
                        if strength[line_idx] >= strength[other_idx]:
                            k_thick, k_thin = cog_limits(ion, line_idx, other_idx)
                            match = match & ew_limits_ok(r_1, r_2, k_thick, k_thin)
                        else:
                            k_thick, k_thin = cog_limits(ion, other_idx, line_idx)
                            match = match & ew_limits_ok(r_2, r_1, k_thick, k_thin)

                    zmatch = np.where(match, 0.5 * (z1[:, None] + z2[None, :]), np.nan)
                    if np.any(np.isfinite(zmatch)):
                        z_matches.append(np.nanmedian(zmatch))

            if z_matches:
                detected_z.extend(z_matches)

        detected_z = np.array(detected_z) if detected_z else None

        # ---------- ALL LINES ----------
        if detected_z is not None:
            for ion, rest in rest_by_ion.items():
                zarr = zabs_by_ion[ion]
                for ell, lam0 in enumerate(rest):
                    zvals = zarr[:, ell]
                    dz = np.abs(zvals[:, None] - detected_z[None, :])
                    min_dz = np.nanmin(dz, axis=1)
                    mask = min_dz <= tol
                    score = 1.0 - min_dz / tol

                    # A line joining a detected system from redshift
                    # proximity alone extends that system, so it must
                    # be within the thin/thick limits with every
                    # transition of the same ion already assigned there
                    # (ew_ordering_ok).
                    wl_int = int(np.floor(lam0))
                    for r in np.where(mask)[0]:
                        if not ew_ordering_ok(ion, wl_int, r, zvals[r]):
                            mask[r] = False

                    update(mask, score, zvals, ion, lam0)

        # `ion_pred`/`wl_pred`/`zabs_pred`/`score_pred` already hold,
        # per region, the single best-scoring match found across every
        # ion/line combination above (`wl_pred` maintained live by
        # `update()` in the floor-int convention used everywhere else
        # in this file) -- with no bootstrap draws to reconcile, that
        # running best already IS the final classification.

        # =====================================================
        # POST-AGGREGATION FIX: CIV ↔ SiIV system cross-identification
        #   If CIV doublet exists, reclassify OTHER lines as SiIV if consistent
        #   If SiIV doublet exists, reclassify OTHER lines as CIV if consistent
        #   (skip lines already correctly classified)
        # =====================================================

        civ1548_idx = np.where((ion_pred == "CIV") & (wl_pred == 1548))[0]
        civ1550_idx = np.where((ion_pred == "CIV") & (wl_pred == 1550))[0]
        siiv1393_idx = np.where((ion_pred == "SiIV") & (wl_pred == 1393))[0]
        siiv1402_idx = np.where((ion_pred == "SiIV") & (wl_pred == 1402))[0]

        civ1548_sys = cluster_by_redshift(civ1548_idx, zabs_pred, tol)
        civ1550_sys = cluster_by_redshift(civ1550_idx, zabs_pred, tol)
        siiv1393_sys = cluster_by_redshift(siiv1393_idx, zabs_pred, tol)
        siiv1402_sys = cluster_by_redshift(siiv1402_idx, zabs_pred, tol)

        # --- build CIV systems ---
        civ_systems = []
        for s1548 in civ1548_sys:
            z1548 = np.nanmedian(zabs_pred[s1548])
            for s1550 in civ1550_sys:
                z1550 = np.nanmedian(zabs_pred[s1550])
                if abs(z1548 - z1550) <= tol:
                    civ_systems.append((s1548, s1550, 0.5 * (z1548 + z1550)))

        # --- build SiIV systems ---
        siiv_systems = []
        for s1393 in siiv1393_sys:
            z1393 = np.nanmedian(zabs_pred[s1393])
            for s1402 in siiv1402_sys:
                z1402 = np.nanmedian(zabs_pred[s1402])
                if abs(z1393 - z1402) <= tol:
                    siiv_systems.append((s1393, s1402, 0.5 * (z1393 + z1402)))

        # -----------------------------------------------------
        # Case 1: CIV system exists → look for OTHER lines = SiIV
        # -----------------------------------------------------
        for s1548, s1550, z_civ in civ_systems:

            civ_indices = set(s1548) | set(s1550)

            for r in range(R):
                if r in civ_indices:
                    continue  # do not touch the CIV doublet itself

                # skip if already correctly classified as SiIV
                if ion_pred[r] == "SiIV" and wl_pred[r] in (1393, 1402):
                    continue

                lam_obs = cmedian[r]

                # test as SiIV 1393
                z_test_1393 = lam_obs / LAM[("SiIV", 1393)] - 1.0
                if abs(z_test_1393 - z_civ) <= tol and \
                   ew_ordering_ok("SiIV", 1393, r, z_test_1393):
                    ion_pred[r]   = "SiIV"
                    wl_pred[r]    = 1393
                    zabs_pred[r]  = z_test_1393
                    score_pred[r] = -1.0
                    continue

                # test as SiIV 1402
                z_test_1402 = lam_obs / LAM[("SiIV", 1402)] - 1.0
                if abs(z_test_1402 - z_civ) <= tol and \
                   ew_ordering_ok("SiIV", 1402, r, z_test_1402):
                    ion_pred[r]   = "SiIV"
                    wl_pred[r]    = 1402
                    zabs_pred[r]  = z_test_1402
                    score_pred[r] = -1.0

        # -----------------------------------------------------
        # Case 2: SiIV system exists → look for OTHER lines = CIV
        # -----------------------------------------------------
        for s1393, s1402, z_siiv in siiv_systems:

            siiv_indices = set(s1393) | set(s1402)

            for r in range(R):
                if r in siiv_indices:
                    continue  # do not touch the SiIV doublet itself

                # skip if already correctly classified as CIV
                if ion_pred[r] == "CIV" and wl_pred[r] in (1548, 1550):
                    continue

                lam_obs = cmedian[r]

                # test as CIV 1548
                z_test_1548 = lam_obs / LAM[("CIV", 1548)] - 1.0
                if abs(z_test_1548 - z_siiv) <= tol and \
                   ew_ordering_ok("CIV", 1548, r, z_test_1548):
                    ion_pred[r]   = "CIV"
                    wl_pred[r]    = 1548
                    zabs_pred[r]  = z_test_1548
                    score_pred[r] = -1.0
                    continue

                # test as CIV 1550
                z_test_1550 = lam_obs / LAM[("CIV", 1550)] - 1.0
                if abs(z_test_1550 - z_siiv) <= tol and \
                   ew_ordering_ok("CIV", 1550, r, z_test_1550):
                    ion_pred[r]   = "CIV"
                    wl_pred[r]    = 1550
                    zabs_pred[r]  = z_test_1550
                    score_pred[r] = -1.0

        # =====================================================
        # POST-AGGREGATION FIX: CIV fake doublet ↔ SiII + OI system
        #   (CII as anchor)
        # =====================================================

        # --- select SiII but exclude 1304 as system anchor ---
        cii_mask = (ion_pred == "CII") & np.isin(wl_pred, [1334])
        cii_idx_all = np.where(cii_mask)[0]

        cii_systems = cluster_by_redshift(cii_idx_all, zabs_pred, tol)

        civ1548_idx = np.where((ion_pred == "CIV") & (wl_pred == 1548))[0]
        civ1550_idx = np.where((ion_pred == "CIV") & (wl_pred == 1550))[0]

        # cluster each, then pair by median z
        civ1548_sys = cluster_by_redshift(civ1548_idx, zabs_pred, tol)
        civ1550_sys = cluster_by_redshift(civ1550_idx, zabs_pred, tol)

        civ_systems = []

        for s1548 in civ1548_sys:
            z1548 = np.nanmedian(zabs_pred[s1548])
            for s1550 in civ1550_sys:
                z1550 = np.nanmedian(zabs_pred[s1550])
                if abs(z1548 - z1550) <= tol:
                    civ_systems.append((s1548, s1550, 0.5*(z1548+z1550)))

        if len(cii_idx_all) > 0 and len(civ1548_idx) > 0 and len(civ1550_idx) > 0:

            for cii_sys in cii_systems:

                z_cii = np.nanmedian(zabs_pred[cii_sys])

                for s1548, s1550, z_civ in civ_systems:

                    # use medians of each component
                    i1548 = s1548[np.argmin(abs(zabs_pred[s1548] - z_civ))]
                    i1550 = s1550[np.argmin(abs(zabs_pred[s1550] - z_civ))]

                    lam_obs_1548 = cmedian[i1548]
                    lam_obs_1550 = cmedian[i1550]

                    z_test_1304 = lam_obs_1550 / LAM[("SiII", 1304)] - 1.0
                    z_test_1302 = lam_obs_1548 / LAM[("OI", 1302)]  - 1.0

                    # require both to match the SiII system redshift,
                    # and the resulting SiII 1304 assignment to be
                    # EW-consistent with any companion SiII line
                    # already present at that system's redshift.
                    if abs(z_test_1304 - z_cii) > tol or \
                       not ew_ordering_ok("SiII", 1304, i1550, z_test_1304):
                        continue

                    # ===== APPLY FIX =====
                    # CIV 1550 → SiII 1304
                    ion_pred[i1550]   = "SiII"
                    wl_pred[i1550]    = 1304
                    zabs_pred[i1550]  = z_test_1304
                    score_pred[i1550] = -1.0

                    # CIV 1548 → OI 1302
                    ion_pred[i1548]   = "OI"
                    wl_pred[i1548]    = 1302
                    zabs_pred[i1548]  = z_test_1302
                    score_pred[i1548] = -1.0

        # =====================================================
        # POST-AGGREGATION FIX: CIV fake doublet ↔ SiII + OI system
        #   (SiII systems built from ALL lines EXCEPT 1304)
        # =====================================================

        # --- select SiII but exclude 1304 as system anchor ---
        si_mask = (ion_pred == "SiII") & np.isin(wl_pred, [1260, 1526, 1808])
        si_idx_all = np.where(si_mask)[0]

        si_systems = cluster_by_redshift(si_idx_all, zabs_pred, tol)

        civ1548_idx = np.where((ion_pred == "CIV") & (wl_pred == 1548))[0]
        civ1550_idx = np.where((ion_pred == "CIV") & (wl_pred == 1550))[0]

        # cluster each, then pair by median z
        civ1548_sys = cluster_by_redshift(civ1548_idx, zabs_pred, tol)
        civ1550_sys = cluster_by_redshift(civ1550_idx, zabs_pred, tol)

        civ_systems = []

        for s1548 in civ1548_sys:
            z1548 = np.nanmedian(zabs_pred[s1548])
            for s1550 in civ1550_sys:
                z1550 = np.nanmedian(zabs_pred[s1550])
                if abs(z1548 - z1550) <= tol:
                    civ_systems.append((s1548, s1550, 0.5*(z1548+z1550)))

        if len(si_idx_all) > 0 and len(civ1548_idx) > 0 and len(civ1550_idx) > 0:

            for si_sys in si_systems:

                z_si = np.nanmedian(zabs_pred[si_sys])

                for s1548, s1550, z_civ in civ_systems:

                    # use medians of each component
                    i1548 = s1548[np.argmin(abs(zabs_pred[s1548] - z_civ))]
                    i1550 = s1550[np.argmin(abs(zabs_pred[s1550] - z_civ))]

                    lam_obs_1548 = cmedian[i1548]
                    lam_obs_1550 = cmedian[i1550]

                    z_test_1304 = lam_obs_1550 / LAM[("SiII", 1304)] - 1.0
                    z_test_1302 = lam_obs_1548 / LAM[("OI", 1302)]  - 1.0

                    # require both to match the SiII system redshift,
                    # and the resulting SiII 1304 assignment to be
                    # EW-consistent with its companion SiII line(s)
                    # (1260/1526/1808) anchoring this very system.
                    if abs(z_test_1304 - z_si) > tol or \
                       not ew_ordering_ok("SiII", 1304, i1550, z_test_1304):
                        continue

                    # ===== APPLY FIX =====
                    # CIV 1550 → SiII 1304
                    ion_pred[i1550]   = "SiII"
                    wl_pred[i1550]    = 1304
                    zabs_pred[i1550]  = z_test_1304
                    score_pred[i1550] = -1.0

                    # CIV 1548 → OI 1302
                    ion_pred[i1548]   = "OI"
                    wl_pred[i1548]    = 1302
                    zabs_pred[i1548]  = z_test_1302
                    score_pred[i1548] = -1.0

        # =====================================================
        # POST-AGGREGATION FIX: MgII → SII using FeII anchor
        # =====================================================

        # --- FeII anchor systems ---
        feii_mask = (ion_pred == "FeII") & np.isin(wl_pred, [1608, 2382, 2600])
        feii_idx_all = np.where(feii_mask)[0]
        feii_systems = cluster_by_redshift(feii_idx_all, zabs_pred, tol)

        # --- MgII candidates (independent, no system building) ---
        mgii_idx = np.where((ion_pred == "MgII") & np.isin(wl_pred, [2796, 2803]))[0]

        if feii_systems and len(mgii_idx) > 0:

            for fe_sys in feii_systems:

                z_fe = np.nanmedian(zabs_pred[fe_sys])

                for r in mgii_idx:

                    lam_obs = cmedian[r]

                    # --- test SII hypotheses ---
                    z_sii_1250 = lam_obs / LAM[("SII", 1250)] - 1.0
                    z_sii_1253 = lam_obs / LAM[("SII", 1253)] - 1.0
                    z_sii_1259 = lam_obs / LAM[("SII", 1259)] - 1.0

                    # determine which transition fits FeII system, and
                    # is EW-consistent with any companion SII line
                    # already assigned at this same redshift (e.g. from
                    # another MgII candidate converted in this loop)
                    if abs(z_sii_1250 - z_fe) <= tol and \
                       ew_ordering_ok("SII", 1250, r, z_sii_1250):
                        ion_pred[r]   = "SII"
                        wl_pred[r]    = 1250
                        zabs_pred[r]  = z_sii_1250
                        score_pred[r] = -1.0

                    elif abs(z_sii_1253 - z_fe) <= tol and \
                         ew_ordering_ok("SII", 1253, r, z_sii_1253):
                        ion_pred[r]   = "SII"
                        wl_pred[r]    = 1253
                        zabs_pred[r]  = z_sii_1253
                        score_pred[r] = -1.0

                    elif abs(z_sii_1259 - z_fe) <= tol and \
                         ew_ordering_ok("SII", 1259, r, z_sii_1259):
                        ion_pred[r]   = "SII"
                        wl_pred[r]    = 1259
                        zabs_pred[r]  = z_sii_1259
                        score_pred[r] = -1.0

        # =====================================================
        # FINAL CONSISTENCY PASS: optically thin/thick limits
        #
        # Each stage above only checks the pairs it creates, so a
        # system assembled from several matches (a 3+-line multiplet,
        # overlapping doublet matches, post-aggregation relabelling)
        # can still hold a pair no stage compared directly. Re-check
        # every pair of distinct transitions of every ion system (same
        # ion, zabs clustered within `tol`); a system with any pair
        # outside the limits is unclassified.
        # =====================================================
        for ion in {x for x in ion_pred if x is not None}:
            if strength_by_ion.get(ion) is None:
                continue
            members = np.where((ion_pred == ion) & np.isfinite(wl_pred))[0]
            for sys_idx in cluster_by_redshift(members, zabs_pred, tol):
                consistent = True
                for p, ra in enumerate(sys_idx):
                    for rb in sys_idx[p + 1:]:
                        if int(wl_pred[ra]) == int(wl_pred[rb]):
                            continue
                        ia = transition_idx(ion, wl_pred[ra])
                        ib = transition_idx(ion, wl_pred[rb])
                        if ia is None or ib is None:
                            continue
                        if not pair_ok(ion, ia, ra, ib, rb):
                            consistent = False
                            break
                    if not consistent:
                        break
                if not consistent:
                    for r in sys_idx:
                        ion_pred[r]   = None
                        wl_pred[r]    = np.nan
                        tran_pred[r]  = np.nan
                        zabs_pred[r]  = np.nan
                        score_pred[r] = np.nan

        # ---------------------------
        # Save
        # ---------------------------
        # ew_arr/dew_arr are integrated over `wave` (QSO rest frame);
        # convert to the absorber rest frame, W_r = W_qso (1+zqso)/(1+zabs),
        # wherever the region has a zabs (as in LineClassifier's copy of
        # this function, whose mocks define C(z, W) in the absorber frame).
        # Unclassified regions keep the QSO-frame value. SNR_line and the
        # EW-ratio tests above are frame-independent.
        zabs_f = np.asarray(zabs_pred, dtype=float)
        has_z = np.isfinite(zabs_f)
        frame = np.where(has_z, (1 + z) / (1 + np.where(has_z, zabs_f, 0.0)), 1.0)
        for r in range(R):
            results.append({
                "ID": id,
                "zqso": z,
                "lam": cmedian[r],
                "start": starts[r],
                "end": ends[r],
                "lam05": wave_sub[starts[r]] * (1 + z),
                "lam95": wave_sub[ends[r]] * (1 + z),
                "ion_pred": ion_pred[r],
                "wl_pred": wl_pred[r],
                "zabs": zabs_pred[r],
                "score": score_pred[r],
                "SNR_line": snr_arr[r],
                "EW": ew_arr[r] * frame[r],
                "dEW": dew_arr[r] * frame[r],
                "dEW_stat": stat_arr[r] * frame[r],
                "dEW_cont": cont_frac_err * csum_arr[r] * frame[r],
                "EW_qso": ew_arr[r],
                "dEW_qso": dew_arr[r],
            })

    return pd.DataFrame(results)
