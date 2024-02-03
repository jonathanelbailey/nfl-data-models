from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    'numpy',
    'pandas',
    'scikit-learn',
    'clearml',
    'xgboost'
]

dev_requirements = [
    "pytest",
    "sphinx"
]

setup(
    name="src",
    version="0.1.0",
    author="Jonathan Bailey",
    author_email="jonathan@jelbailey.com",
    description="A Python package for predicting NFL game outcomes using advanced data models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonathanelbailey/nfl-data-models",
    project_urls={
        "Bug Tracker": "https://github.com/jonathanelbailey/nfl-data-models/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    include_package_data=True,
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
)
