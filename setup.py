"""Setup script for Garmin Analyser."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="garmin-analyser",
    version="1.0.0",
    author="Garmin Analyser Team",
    author_email="support@garminanalyser.com",
    description="Comprehensive workout analysis for Garmin data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/garmin-analyser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Sports/Healthcare",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "pdf": ["weasyprint>=54.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "garmin-analyser=main:main",
            "garmin-analyzer-cli=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "garmin_analyser": ["config/*.yaml", "visualizers/templates/*.html", "visualizers/templates/*.md"],
    },
)