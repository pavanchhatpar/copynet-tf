from setuptools import setup, find_packages

# read the contents of your README file
from os import path
par_directory = path.abspath(path.dirname(__file__))
with open(path.join(par_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="copynet-tf",
    description="CopyNet with TensorFlow 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/pavanchhatpar/copynet-tf",
    packages=find_packages(),
    version="0.1.5",
    author='Pavan Chhatpar',
    author_email='pavanchhatpar@gmail.com'
)
