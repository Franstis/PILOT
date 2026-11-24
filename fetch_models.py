"""
Download the two networks of the pipeline from Zenodo into models/ and check their MD5.

  python fetch_models.py

- fine-tuned continuum autoencoder: https://doi.org/10.5281/zenodo.24500565
- U-Net line finder (UNet.h5 on Zenodo): https://doi.org/10.5281/zenodo.22711575
"""
import hashlib
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    # local name: (Zenodo URL, md5)
    "autoencoder_finetuned_nmf.h5": (
        "https://zenodo.org/records/24500565/files/autoencoder_finetuned_nmf.h5?download=1",
        "958feaa549ec2c430ea1eabed4a92cbc"),
    "best_unet_nonoise_snr0_longrun.h5": (
        "https://zenodo.org/records/22711575/files/UNet.h5?download=1",
        "f6cae071d5c36c9fd2dc7dc49501f0dd"),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out_dir = os.path.join(HERE, "models")
    os.makedirs(out_dir, exist_ok=True)
    for name, (url, checksum) in MODELS.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and md5(path) == checksum:
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, path + ".part")
        if md5(path + ".part") != checksum:
            os.remove(path + ".part")
            raise RuntimeError(f"{name}: MD5 mismatch, download failed")
        os.replace(path + ".part", path)
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
