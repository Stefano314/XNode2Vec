import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Xnode2vec",
    version="0.0.4.2",
    author="Stefano Bianchi",
    author_email="stefanobianchi314@gmail.com",
    description="Implementation of fastnode2vec algorithm for networkx objects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stefano314/XNode2Vec",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["Xnode2vec"],
    install_requires=["fastnode2vec"],
    python_requires=">=3.6",
)
