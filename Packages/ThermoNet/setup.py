import setuptools

setuptools.setup(
    name="thermonet",
    version="0.1.0",
    author="Daniel Schatzl",
    author_email="danielschatzl16@gmail.com",
    description="Approximates the thermodynamic functions Gibbs energy, entropy, enthalpy and heat capacity.",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'sgte', 'thermoclassifier'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    package_data={'': ['models/*.*', 'data/*.xlsx']},
)