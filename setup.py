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
    long_description=(
        "This code attempts to predict the influence of osmolytes on protein "
        "stability, using the methods in:  Auton M, Bolen DW. Predicting the "
        "energetics of osmolyte-induced protein folding/unfolding. "
        "*Proc Natl Acad Sci* 102:15065 (2005) "
        "https://doi.org/10.1073/pnas.0507053102. Other models may be added "
        "in the future."
    ),
    python_requires=">=3.6",
    license="CC0-1.0",
    author="Nathan Baker",
    author_email="nathanandrewbaker@gmail.com",
    url="https://github.com/Electrostatics/osmolytes",
    packages=setuptools.find_packages(),
    package_data={
        "": ["data/*.yaml", "tests/data/*.json", "tests/data/*.yaml"]
    },
    install_requires=["numpy", "scipy", "pyyaml", "pandas"],
    tests_require=["pytest"],
    entry_points={"console_scripts": ["mvalue=osmolytes.main:main",]},
    keywords="science chemistry biophysics biochemistry",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Common Public License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
