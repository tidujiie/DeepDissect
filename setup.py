from setuptools import setup, find_packages

setup(
    name='deep_dissect',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "pycocotools>=2.0.7",
        "scipy>=1.10.1",
        "numpy>=1.24.1",
        "scikit-learn>=1.3.2",
        "Pillow>=9.3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.3",
        "matplotlib>=3.7.3",
        "ipywidgets>=8.0.4",
        "IPython>=8.12.3",
        "ipykernel>=6.29.5"
    ],
    description='Deep Dissect Library for ablation studies on mmdetection models',
    author='anonymous',
    keywords='deep dissect'
)