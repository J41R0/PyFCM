import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='py_fcm',
    version='1.0.0',
    scripts=[],
    author="Jairo Lefebre",
    author_email="jairo.lefebre@gmail.com",
    description="Fuzzy cognitive maps python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/J41R0/PyFCM",
    packages=setuptools.find_packages("py_fcm", exclude=["tests"]),
    install_requires=[
        'pandas >= 0.24.2',
        'matplotlib >= 3.1.0',
        'networkx >= 2.3',
        'numpy >= 1.19.1',
        'numba >= 0.51.2',
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
