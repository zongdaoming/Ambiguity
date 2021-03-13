# _*_ coding: utf-8 _*_
# @author  :   naive dormin
# @time    :   2021/03/13 21:21:09

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'dm-sonnet==1.35', 'tensorflow==1.14',
                     'tensorflow-probability==0.7.0']

setup(
    name='hpu_net',
    version='0.1',
    description='A library for the Hierarchical Probabilistic U-Net model.',
    url='https://github.com/deepmind/deepmind-research/hierarchical_probabilistic_unet',
    author='dorming',
    author_email='ecnuzdm@gmail.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)