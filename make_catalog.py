"""
Absorber catalogue of a file of quasar spectra with the functions of catalog_utils.py
(continuum autoencoder, U-Net line finder, EW measurement and classification).

  python make_catalog.py SPECTRA_FILE --spectra-out OUT_SPECTRA.h5 --catalog-out OUT_CATALOG.h5
                         [--v-tol 250] [--ew-sigma-margin 3] [--n-max N] [--chunk 10000]
                         [--cont-frac-err X]

SPECTRA_FILE: HDF5 in the input format of catalog_utils.read_spectra (ids, zqso, norm, flux,
ivar, nan_mask on the grid of data/rest_wave.h5), e.g. data/desi_dr1_subsample500.h5 or the
DESI DR1 parent file desi_selected_with_continuum_labels.h5.

Writes (catalog_utils.build_catalog)
  --spectra-out : continuum, continuum_scaled, labels_pred (n, 16384), links to the input datasets
  --catalog-out : pred_df (one row per region) and spectra (one row per spectrum), pandas HDF5,
                  with the classifier settings v_tol, ew_sigma_margin as attributes of pred_df
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_utils as cu

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("spectra", help="input spectra file (catalog_utils.read_spectra format)")
ap.add_argument("--spectra-out", required=True, help="output: continuum and U-Net labels")
ap.add_argument("--catalog-out", required=True, help="output: the absorber catalogue")
ap.add_argument("--v-tol", type=float, default=cu.V_TOL,
                help=f"velocity tolerance of the line associations [km/s] (default {cu.V_TOL:g})")
ap.add_argument("--ew-sigma-margin", type=float, default=cu.EW_SIGMA_MARGIN,
                help=f"sigma margin of the EW ordering / ratio tests (default {cu.EW_SIGMA_MARGIN:g})")
ap.add_argument("--n-max", type=int, default=None, help="only the first N spectra (tests)")
ap.add_argument("--chunk", type=int, default=cu.CHUNK, help=f"spectra per chunk (default {cu.CHUNK})")
ap.add_argument("--cont-frac-err", type=float, default=None,
                help="constant fractional continuum error (default: S/N dependent)")
args = ap.parse_args()

for out in (args.spectra_out, args.catalog_out):
    if os.path.abspath(out) == os.path.abspath(args.spectra):
        sys.exit(f"output {out} would overwrite the input")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

print(f"input {args.spectra} ({cu.n_parent_spectra(args.spectra)} spectra), v_tol {args.v_tol:g} km/s, "
      f"EW sigma margin {args.ew_sigma_margin:g}, started {time.ctime()}", flush=True)
cu.build_catalog(args.spectra_out, args.catalog_out, n_max=args.n_max, chunk=args.chunk,
                 cont_frac_err=args.cont_frac_err, parent=args.spectra, v_tol=args.v_tol,
                 ew_sigma_margin=args.ew_sigma_margin)
print(f"finished {time.ctime()}")
