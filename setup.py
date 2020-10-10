import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='py_fcm',
    version='0.2.5',
    scripts=[],
    author="Jairo Lefebre",
    author_email="jairo.lefebre@gmail.com",
    description="Fuzzy cognitive maps python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/J41R0/PyFCM",
    packages=setuptools.find_packages(exclude="tests"),
    install_requires=[
        'pandas >= 0.24.2',
        'matplotlib >= 3.1.0',
        'networkx >= 2.3',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
