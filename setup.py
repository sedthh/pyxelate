from setuptools import setup

setup(name='pyxelate',
      version='1.2.1',
      description='Pyxelate is a Python class that converts images into tiny pixel arts with limited color palettes.',
      url='http://github.com/sedthh/pyxelate',
      author='sedthh',
      license='MIT',
      packages=[''],
      zip_safe=False,
      install_requires=[
          'scikit-image==0.16.2', 'scikit-learn==0.22.1'
      ],
      )
