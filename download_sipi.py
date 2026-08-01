import urllib.request
import zipfile
import os

os.makedirs('data/usc_sipi', exist_ok=True)
url = 'https://sipi.usc.edu/database/misc.zip'
zip_path = 'data/usc_sipi/misc.zip'

print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, zip_path)

print(f"Extracting {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('data/usc_sipi')

print("Done!")
