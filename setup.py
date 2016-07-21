#!/usr/bin/env python

from distutils.core import setup

setup(name='context2vec',
      version='1.0',
      description='Bidirectional-LSTM sentnetial context embeddings',
      author='Oren Melamud',
      url='https://www.github.com/orenmel/context2vec/',
      packages=['context2vec','context2vec.common','context2vec.eval','context2vec.eval.wsd','context2vec.train'],
      license=' Apache 2.0'
     )