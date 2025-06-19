from setuptools import setup, find_packages

setup(
    name="mushkil-viz",
    version="0.1.0",
    description="Intelligent Tabular Data Analysis & Visualization System",
    author="Waleed Hashmi and Abdullah Athar",
    author_email="ama86@cantab.ac.uk",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black",
            "flake8",
            "isort",
            "pre-commit",
        ],
    },
)
