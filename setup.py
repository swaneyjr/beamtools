import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name='beamtools-crayfis',
        author='Jeff Swaney',
        author_email='jswaney@uci.edu',
        description='Analysis tools for beam data',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=setuptools.find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
        ],
)
