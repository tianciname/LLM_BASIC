from setuptools import setup, find_packages

setup(
    name="trainmon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8",
        "tensorboard>=2.8"
    ],
    author="Your Name",
    description="A lightweight training monitor with TensorBoard and CSV logging",
    python_requires=">=3.7",
)