import zipfile
import tarfile
import wget
import os

from greetings import cfg


if __name__ == '__main__':
    if not os.path.exists(cfg.RAW_DATA):
        os.makedirs(cfg.RAW_DATA)
    if not os.path.exists(cfg.SAVE_LOC):
        os.makedirs(cfg.SAVE_LOC)
    raw_data = os.path.join(cfg.RAW_DATA, "greetings.tar.gz")
    glove_zip = os.path.join(cfg.SAVE_LOC, "glove.840B.300d.zip")
    glove_unzip = os.path.join(cfg.SAVE_LOC, "glove")
    wget.download(cfg.DATASET, raw_data)
    wget.download(cfg.GLOVE, glove_zip)

    with zipfile.ZipFile(glove_zip) as f:
        f.extractall(glove_unzip)
    os.remove(glove_zip)

    with tarfile.open(raw_data) as f:
        f.extractall(cfg.RAW_DATA)
    os.remove(raw_data)
