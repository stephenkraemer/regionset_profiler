from setuptools import setup

setup(
        name='region_set_profiler',
        version='0.1',
        author='Stephen Kraemer',
        author_email='stephenkraemer@gmail.com',
        license='MIT',
        package_dir={'': 'src'},
        packages=['region_set_profiler'],
        python_requires='>=3.6',
        install_requires=[
            'scipy',
            'statsmodels',
            'FisherExact',
            'numpy',
            'pandas>=0.23.4',
            'more_itertools',
        ],
        extras_require={
            'dev': [
                'pytest',
            ]
        }
)
