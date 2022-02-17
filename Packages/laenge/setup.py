import setuptools

setuptools.setup(
    name="LaengeNet",
    version="0.1.0",
    author="Daniel Schatzl",
    author_email="danielschatzl16@gmail.com",
    description="Rebuild of the Network to approximate Gibbs energy, entropy, enhalpy and heat capacity of iron "
                "published by Länge M. https://doi.org/10.1007/s00500-019-04663-3",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'sgte'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    package_data={'': ['models/*.*']},
)