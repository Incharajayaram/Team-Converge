"""ECDD Experimentation - Setup Configuration.

This allows the package to be installed in development mode:
    pip install -e .

Or installed directly:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="ecdd-experimentation",
    version="0.1.0",
    description="ECDD Deepfake Detection Experimentation Framework",
    author="Team Converge",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "training": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "matplotlib>=3.7.0",
        ],
        "experiments": [
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
            "scikit-learn>=1.3.0",
        ],
        "federated": [
            "flask>=2.3.0",
            "requests>=2.31.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ],
        "tflite": [
            "tensorflow>=2.13.0",
            "onnx>=1.14.0",
            "onnx-tf>=1.10.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
            "flask>=2.3.0",
            "requests>=2.31.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecdd-run-experiments=run_all_tests:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
