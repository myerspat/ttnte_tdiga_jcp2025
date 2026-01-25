import requests
import hashlib
import os
import json
import time
from tqdm import tqdm

# --- SETTINGS ---
USE_SANDBOX = False
ACCESS_TOKEN = "CjZ6TEA8wOpOwl4OMGw6vt7Hrq3TBXzuaMcnJA60iVpLoavJxPEpRQwLJtRq"
FILE_PATH = "ttnte_tdiga_jcp2025_data.tar.gz"
MAX_RETRIES = 10  # How many times to restart the 39GB upload if it fails
CHUNK_SIZE = 5 * 1024 * 1024

if USE_SANDBOX:
    BASE_URL = "https://sandbox.zenodo.org/api/deposit/depositions"
    SITE_URL = "https://sandbox.zenodo.org"
else:
    BASE_URL = "https://zenodo.org/api/deposit/depositions"
    SITE_URL = "https://zenodo.org"

PARAMS = {"access_token": ACCESS_TOKEN}


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="1. Local Checksum"
        ) as pbar:
            for chunk in iter(
                lambda: f.read(5 * 1024 * 1024), b""
            ):  # 1MB chunks for speed
                hash_md5.update(chunk)
                pbar.update(len(chunk))
    return hash_md5.hexdigest()


def upload_file_with_retries(bucket_url, file_path):
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n3. Upload Attempt {attempt}/{MAX_RETRIES}...")
            with open(file_path, "rb") as f:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc="   Uploading",
                    leave=False,
                ) as pbar:
                    # Wrapper to update progress bar
                    class ProgressFile:
                        def __init__(self, file, size, pbar):
                            self.file = file
                            self.size = size
                            self.pbar = pbar

                        def read(self, n):
                            # We ignore 'n' and force our high-speed CHUNK_SIZE
                            chunk = self.file.read(10 * 1024 * 1024)
                            if chunk:
                                self.pbar.update(len(chunk))
                                # Optional: uncomment the next line to debug in nohup.out
                                # print(f"Sent {len(chunk)} bytes...")
                            return chunk

                        def __len__(self):
                            return self.size

                        # These stubs help 'requests' treat it like a real file
                        def seek(self, offset, whence=0):
                            pass

                        def tell(self):
                            return 0

                    # Direct PUT to the bucket
                    r = requests.put(
                        f"{bucket_url}/{filename}",
                        data=ProgressFile(f, file_size, pbar),
                        params=PARAMS,
                        timeout=None,
                    )
                    r.raise_for_status()
                    return r  # Success!

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"\n⚠️ Network error on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                wait = attempt * 10  # Wait 10s, 20s, 30s...
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print("❌ Max retries reached. Check your connection.")
                raise e


def run_deposition():
    # 1. Local Checksum
    local_hash = calculate_md5(FILE_PATH)

    # 2. Create Draft
    print("2. Creating Zenodo Draft...")
    r = requests.post(BASE_URL, params=PARAMS, json={})
    r.raise_for_status()
    depo_data = r.json()
    bucket_url = depo_data["links"]["bucket"]
    depo_id = depo_data["id"]

    # 3. Robust Upload
    try:
        res = upload_file_with_retries(bucket_url, FILE_PATH)

        # 4. Verify
        server_hash = res.json()["checksum"].replace("md5:", "")
        if local_hash == server_hash:
            print(f"✅ Success! MD5 Verified.")
        else:
            print("❌ Checksum mismatch!")
            return

        # 5. Apply Metadata
        authors = [
            {
                "name": "Myers, Patrick",
                "affiliation": "https://ror.org/00jmfr291",
                "orcid": "0000-0003-0261-5474",
                "type": "Work package leader",
            },
            {
                "name": "Radaideh, Majdi I",
                "affiliation": "https://ror.org/00jmfr291",
                "orcid": "0000-0002-2743-0567",
                "type": "Supervisor",
            },
            {
                "name": "Kiedrowski, Brian",
                "affiliation": "https://ror.org/00jmfr291",
                "orcid": "0000-0001-8517-4410",
                "type": "Supervisor",
            },
        ]
        metadata = {
            "metadata": {
                "title": 'Raw Data for "Tensorized Discontinuous Isogeometric Analysis Method for the 2-D Time-Independent Linearized Boltzmann Transport Equation"',
                "upload_type": "dataset",
                "description": "Raw data upload",
                "creators": authors,
                "contributors": authors,
                "version": "1.0.0",
                "publication_date": date.today().isoformat(),
                "license": "cc-by-4.0",
                "access_right": "open",
                "keywords": [
                    "Transport (physics)",
                    "Tensor Networks",
                    "Tensor Trains",
                    "Discrete Ordinates",
                    "Isogeometric Analysis",
                    "Numerical Methods",
                    "Discontinuous Galerkin",
                ],
                "language": "eng",
            }
        }

        requests.put(
            f"{BASE_URL}/{depo_id}", params=PARAMS, json=metadata
        ).raise_for_status()

        print(f"\n🎉 DRAFT SAVED: {SITE_URL}/deposit/{depo_id}")

    except Exception as e:
        print(f"Critical Failure: {e}")


if __name__ == "__main__":
    run_deposition()
