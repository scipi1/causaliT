"""
Setup configuration for proT package.

This minimal setup.py allows editable installation (pip install -e .)
without duplicating dependency management. Dependencies should be managed
via requirements.txt or your existing environment setup.
"""

from setuptools import setup, find_packages

setup(
    name="proT",
    version="0.1.0",
    description="Process Transformer for sequence prediction",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    # No install_requires - dependencies managed separately in requirements.txt
    # This allows pip install -e . without reinstalling packages
)
