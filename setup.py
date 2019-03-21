"""
This setup script is based on pypa's template.
https://github.com/pypa/sampleproject
"""


from setuptools import setup, find_packages


setup(
    name='chainer_profutil',
    version='0.0.2',
    description='Chainer utility tool for GPU profiling by nvprof.',
    author='Kazuhiro Yamasaki',
    author_email='yamasaki.k.1101@gmail.com',
    license='MIT License',
    packages=find_packages(exclude=['docs', 'tests']),
)