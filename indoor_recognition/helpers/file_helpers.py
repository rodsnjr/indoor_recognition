from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm, trange
import os
import zipfile

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def downloader(url, directory, filename='dataset.zip', description='Dataset'):
    """
        Downloader function with progress bar
        for Jupyter Notebooks
    """
    download_description = str(description)
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=download_description) as pbar:
        urlretrieve(url, os.path.join(directory, filename), pbar.hook)

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

def find_images_directory(directory):
    " Return Generator with each image in directory, and a label (image contained in subdirectory) "
    return (0,0)