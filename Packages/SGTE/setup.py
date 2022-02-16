import setuptools

setuptools.setup(
    name="sgte",
    version="0.1.0",
    author="Daniel Schatzl",
    author_email="danielschatzl16@gmail.com",
    description="Converts the sgte data coefficients stored in data into (temperature, value) pairs.",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'pandas'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    package_data={'': ['data/*.xlsx']},
)