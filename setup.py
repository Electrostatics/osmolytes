"""Set up the osmolytes package."""
import setuptools


with open("README.md", "r") as readme:
    LONG_DESCRIPTION = readme.read()


setuptools.setup(
    name="osmolytes",
    version="0.0.1",
    description=(
        "This code attempts to predict the influence of osmolytes on protein "
        "stability"
    ),
    long_description=LONG_DESCRIPTION,
    python_requires=">=3.6",
    license="BSD",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    tests_require=["pytest"],
)
