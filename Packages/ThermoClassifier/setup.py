import setuptools

setuptools.setup(
    name="ThermoClassifier",
    version="0.1.0",
    author="Daniel Schatzl",
    author_email="danielschatzl16@gmail.com",
    description="Classifies thermodynamic measurement data of the properties Gibbs energy, entropy, enthalpy or heat-"
                "capacity into the element and phases the measurements have been taken from.",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'pandas', 'torch', 'SGTE'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    package_data={'': ['data/*.xlsx']},
)