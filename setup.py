from setuptools import setup

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

REQUIREMENTS = ["numpy", "matplotlib", "swyft==0.4.4", "h5py", "scipy", "torch"]

setup(
    name="CADDENA",
    version="0.1",
    description="Combined ANalysis of Direct Detection Experiments.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Martin de los Rios, Andres Perez & David Cerdeño",
    author_email=" martindelosrios13@gmail.com ",
    url=" https://github.com/martindelosrios/CADDENA",
    py_modules=["ez_setup"],
    packages=["CADDENA", "CADDENA.dataset"],
    #    exclude_package_data={"": ["tests"]},
    include_package_data=True,  # < - - - - - -- solo si hay datos
    package_data={"CADDENA": ["dataset/*"]},
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["CADDENA", "Dark matter"],
    classifiers=[
        " Development Status :: 4 - Beta",
        " Intended Audience :: Education",
        " Intended Audience :: Science/Research",
        " License :: OSI Approved :: MIT License",
        " Operating System :: OS Independent",
        " Programming Language :: Python",
        " Topic :: Scientific/Engineering",
    ],
)
