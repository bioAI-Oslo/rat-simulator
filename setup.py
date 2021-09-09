import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rat-simulator",
    version="1.0.1",
    author="Vemund S. Schoyen",
    author_email="vemund@live.com",
    description="A rat in an environment simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vemundss/rat-simulator",
    packages=setuptools.find_packages(),
    install_requires=['scipy',
                      'numpy',                     
                      ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
