"""
R3MES Python SDK Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="r3mes-sdk",
    version="0.1.0",
    author="R3MES Team",
    author_email="dev@r3mes.network",
    description="Official Python SDK for the R3MES decentralized AI training network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/r3mes/sdk-python",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],
        "wallet": [
            "bip39>=2.0.0",
            "cosmpy>=0.8.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/r3mes/sdk-python/issues",
        "Documentation": "https://docs.r3mes.network/sdk/python",
        "Source": "https://github.com/r3mes/sdk-python",
    },
)
