from os import path
from codecs import open
from setuptools import setup, find_packages

# We are using a README.md, if you do not have this in your folder, replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Here you specify the meta-data of your package. The `name` argument is needed in some other steps.
setup(
    name="v6-descriptive-statistics",
    version="2.0.0",
    description="Vantage6 algorithm that retrieves descriptive statistics ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/STRONGAYA/v6-descriptive-statistics",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vantage6-algorithm-tools",
        "pandas",
        "requests",
        "numpy",
        "vantage6-strongaya-general @ git+https://github.com/STRONGAYA/v6-tools-general.git@v1.0.1",
        "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v0.1.3",
    ],
)
