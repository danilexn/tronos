import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tronos-danilexn",
    version="0.1.0",
    author="Daniel Leon-Perinan",
    author_email="daniel@ilerod.com",
    description="Scripts for TRONOS: TRacking of Nuclear OScillations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danilexn/tronos",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS"
    ],
    python_requires='>=3.7',
)