from setuptools import setup, find_packages

setup(
    name='pca',
    version='0.1',
    description='Package to perform pca on either data, images or a cube of images',
    url='https://github.com/jmilou/pca',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='image processing data analysis',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)
