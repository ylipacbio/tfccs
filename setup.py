import os

from setuptools import find_packages, setup


def _get_local_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def _get_description(file_name):
    with open(file_name, 'r') as f:
        _long_description = f.read()
    return _long_description


setup(
    name='tfccs',
    version='0.1.0',  # don't forget to update pbsvtools/__init__.py too
    packages=find_packages(),
    license='BSD',
    author='yli',
    author_email='yli@pacificbiosciences.com',
    description='CCSQV using tensorflow',
    setup_requires=['setuptools'],
    # Maybe the pbtools-* should really be done in a subparser style
    entry_points={'console_scripts': [
        'fextract2numpy=tfccs.fextract2numpy:main',
        'fextract2stat=tfccs.fextract2stat:main',
        'multinomial=tfccs.train:multinomial_main',
        'cnn=tfccs.train:cnn_main',
        'evalmodel=tfccs.evalmodel:main',
        'sampling=tfccs.sampling:main',
    ]},
    install_requires=[],
    tests_require=['pytest'],
    long_description=_get_description(_get_local_file('README.md')),
    classifiers=['Development Status :: 4 - Beta'],
    include_package_data=True,
    zip_safe=False,
)
