import os, wget
from tqdm import trange

url = 'https://aclanthology.org/2022.findings-emnlp.'
download_path = '/content/drive/MyDrive/NLPPapers/EMNLP/2022/findings/'
for i in trange(1, 548):
    full_url = url + str(i) + '.pdf'
    wget.download(full_url, out=download_path)
