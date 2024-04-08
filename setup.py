from setuptools import setup

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

REQUIREMENTS = ["numpy", "matplotlib", "swyft==0.4.4"]

setup(
    name="BATMAN",
    version="0.1",
    description="BAyesian Toolkit for Machine learning ANalysis of direct detection experiments.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Martin de los Rios, Andres Perez & David Cerdeño",
    author_email=" martindelosrios13@gmail.com ",
    url=" https://github.com/martindelosrios/BATMAN",
    py_modules=["ez_setup"],  # < - - - - - - - aca van los modulos
    packages=["BATMAN", "BATMAN.dataset"],  # < - -- - - - - aca van los paquetes
    exclude_package_data={"": ["tests"]},
    include_package_data=True,  # < - - - - - -- solo si hay datos
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=["BATMAN", "Dark matter"],
    classifiers=[
        " Development Status :: 4 - Beta",
        " Intended Audience :: Education",
        " Intended Audience :: Science/Research",
        " License :: OSI Approved :: MIT License",
        " Operating System :: OS Independent",
        " Programming Language :: Python",
        " Programming Language :: Python :: 3.8",
        " Topic :: Scientific/Engineering",
    ],
)
