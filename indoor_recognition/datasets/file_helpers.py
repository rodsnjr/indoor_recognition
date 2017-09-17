from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm, trange

import zipfile

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def downloader(url, directory='/', filename='dataset.zip', desc='Dataset'):
    """
        Downloader function with progress bar
        for Jupyter Notebooks
    """
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
        urlretrieve(url, directory, filename, pbar.hook)

def extract_zip(filename, directory):
    
    zf = zipfile.ZipFile(filename)
    uncompress_size = sum((item.file_size for item in zf.infolist()))
    extracted_size = 0
    
    with tqdm(total=0) as pbar:
        for item in zf.infolist():
            extracted_size += item.file_size
            percentage = extracted_size * 100/uncompress_size
            zf.extract(item, directory)
            pbar.update(percentage)