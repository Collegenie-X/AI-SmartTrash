from setuptools import setup, find_packages

setup(
    name="smart-trash",
    version="0.1",
    packages=find_packages(),
    install_requires=["streamlit", "tensorflow", "pillow", "numpy", "opencv-python"],
)
