from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'pandas',
    'openpyxl',
]

setup(
    name="py-cutter",
    version="0.1.0",
    author="Rian Bogley",
    author_email="rianbogley@gmail.com",
    description="A simple tool to cut audio and video files using timestamps from a CSV or Excel file",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RianBogley/py-cutter",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires
)