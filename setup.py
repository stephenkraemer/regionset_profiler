from setuptools import find_packages, setup

setup(
        name='region_set_profiler',
        version='0.1',
        author='Stephen Kraemer',
        author_email='stephenkraemer@gmail.com',
        license='MIT',
        package_dir={'': 'src'},
        packages = find_packages(where='src', exclude=['contrib', 'docs', 'tests*']),
        python_requires='>=3.6',
        install_requires=[
            'scipy',
            'statsmodels',
            'numpy',
            'pandas',
            'more_itertools',
        ],
        extras_require={
            'dev': [
                'pytest',
            ]
        }
)
