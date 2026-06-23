import requests, tarfile, os

URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
os.makedirs("data/raw", exist_ok=True)

print("Downloading BSD500...")
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(URL, stream=True, headers=headers)
response.raise_for_status()

with open("data/raw/BSR_bsds500.tgz", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Extracting...")
with tarfile.open("data/raw/BSR_bsds500.tgz", "r:gz") as tar:
    tar.extractall("data/raw/")
print("Done.")
