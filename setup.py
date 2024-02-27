from setuptools import setup, find_packages


setup(name='ontoEval',
      version="1.0.0",
      description='Ontology Evaluation using Transformers',
      author='Antonio Zaitoun',
      packages=find_packages(exclude=('bert-pre-training', 'comparison', 'figures')),
      install_requires=['rdflib', 'pandas', 'spacy', 'transformers', 'torch']
      )
