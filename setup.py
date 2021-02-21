from setuptools import find_packages, setup

setup(
        name='region_set_profiler',
        version='0.1',
        author='Stephen Kraemer',
        author_email='stephenkraemer@gmail.com',
        license='MIT',
        package_dir={'': 'src'},
        packages = find_packages(where='src', exclude=['contrib', 'docs', 'tests*']),
        python_requires='>=3.8',
        install_requires=[
            'attrs',
            'codaplot @ git+https://github.com/stephenkraemer/codaplot.git',
            'joblib',
            'more_itertools',
            'numpy',
            'pandas',
            'scipy',
            'statsmodels',
        ],
        extras_require={
            'dev': [
                'pytest',
            ]
        }
)
