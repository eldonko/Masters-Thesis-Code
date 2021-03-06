import setuptools

setuptools.setup(
    name="utils",
    version="0.1.0",
    author="Daniel Schatzl",
    author_email="danielschatzl16@gmail.com",
    description="utils",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'thermonet', 'pandas', 'PyPDF2', 'matplotlib', 'numpy'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)