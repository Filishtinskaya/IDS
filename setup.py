from setuptools import setup, find_packages

setup(
    name='IDS',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
      'click',
      'numpy',
      'scapy',
      'scikit-learn',
      'cryptography'
    ],
    entry_points='''
        [console_scripts]
        IDS=cli:main
    ''',
)