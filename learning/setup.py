from setuptools import setup

setup(
    name='aido1-LF1-baseline-IL-logs-tensorflow',
    version='2018.10.19',
    keywords='duckietown, logs, imitation learning, tensorflow',
    install_requires=[
        'h5py',
        'numpy',
        'pandas==0.23.0',
        'tables>=3.4.3',
    ],
    entry_points={
        'console_scripts': [
            'LF_IL_tensorflow-start=AIDO18_LF_IL_tensorflow.launcher:main',
        ],
    },
)
