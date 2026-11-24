# DESI DR1 absorber catalogue pipeline

Builds a catalogue of metal absorbers in quasar spectra: continuum from a fine-tuned
autoencoder, absorption pixels from a U-Net, then equivalent widths and ion / transition
classification by redshift coincidence with curve-of-growth constraints.

- `catalog_pipeline.ipynb`: the pipeline step by step on 500 DESI DR1 spectra, a check against
  the production catalogue, and how to build a catalogue of your own spectra.
- `catalog_utils.py`: all the functions (import it from this directory).
- `data/`: the wavelength grid, transition list and the 500 example spectra.
- `models/`: the two networks, downloaded from Zenodo by `fetch_models.py`: the fine-tuned
  continuum autoencoder (https://doi.org/10.5281/zenodo.24500565) and the U-Net line finder
  (https://doi.org/10.5281/zenodo.22711575, `UNet.h5`, saved as
  `models/best_unet_nonoise_snr0_longrun.h5`).

## Quick start

```bash
pip install -r requirements.txt
python fetch_models.py          # downloads the networks into models/ (~70 MB)
jupyter notebook catalog_pipeline.ipynb
```

## Your own catalogue

Write your spectra to an HDF5 file in the format of `data/desi_dr1_subsample500.h5`
(datasets `ids`, `zqso`, `norm`, `flux`, `ivar`, `nan_mask` on the grid of
`data/rest_wave.h5`; see the `catalog_utils` docstring), then

```python
import catalog_utils as cu
pred_df, spectra = cu.build_catalog(parent="my_spectra.h5",
                                    spectra_out="my_continuum_labels.h5",
                                    catalog_out="my_catalog.h5")
```

or, from the command line,

```bash
python make_catalog.py my_spectra.h5 --spectra-out my_continuum_labels.h5 --catalog-out my_catalog.h5 \
    [--v-tol 250] [--ew-sigma-margin 3] [--n-max N]
```

The networks were trained for DESI quasars at 2.5 < z < 4 on this rest-frame grid.
