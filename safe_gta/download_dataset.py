"""
Download MetaDrive offline safe RL dataset (from HuggingFace mirror).
Run this in your Anaconda terminal:

  conda activate safe_gta
  python download_dataset.py
"""
import os
import urllib.request
import ssl

DATA_DIR = os.path.join(os.path.dirname(__file__), "safe_gta", "data")
os.makedirs(DATA_DIR, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main"

# Only download mediumsparse for now (124 MB) — enough for Baseline 1 & Safe-GTA
DATASETS = {
    "metadrive_mediumsparse.hdf5": f"{HF_BASE}/SafeMetaDrive-mediumsparse-v0-50-1000.hdf5",
    "metadrive_mediummean.hdf5":   f"{HF_BASE}/SafeMetaDrive-mediummean-v0-50-1000.hdf5",
    "metadrive_mediumdense.hdf5":  f"{HF_BASE}/SafeMetaDrive-mediumdense-v0-50-1000.hdf5",
}

# Allow unverified SSL (some machines have corporate proxy cert issues)
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / 1024 / 1024
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


for fname, url in DATASETS.items():
    out_path = os.path.join(DATA_DIR, fname)
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Already exists: {fname}  ({size_mb:.1f} MB)")
        continue

    print(f"\nDownloading {fname} ...")
    print(f"  URL: {url}")

    # Try with requests first (better progress + redirect handling)
    try:
        import requests
        r = requests.get(url, stream=True, verify=False, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(downloaded / total * 100, 100)
                    print(f"\r  {pct:5.1f}%  {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB", end="", flush=True)
                else:
                    print(f"\r  {downloaded/1024/1024:.1f} MB downloaded", end="", flush=True)
        print()
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
        continue
    except Exception as e:
        print(f"\n  requests failed ({type(e).__name__}), trying urllib...")

    # Fallback: urllib
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, out_path, reporthook=reporthook)
        print()
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        print()
        print("  *** Manual download instructions ***")
        print(f"  1. Open this URL in your browser: {url}")
        print(f"  2. Save the file as: {out_path}")
        print()

print("\nDone!")
print("Files in safe_gta/data/:")
for f in sorted(os.listdir(DATA_DIR)):
    mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1024 / 1024
    print(f"  {f}  ({mb:.1f} MB)")
