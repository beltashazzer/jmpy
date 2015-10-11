from setuptools import setup

setup(name='jmpy',
      version='0.10.0',
      description='Plotting and Modeling Package',
      url='',
      author='David Daycock',
      author_email='daycock@gmail.com',
      license='MIT',
      packages=[
        'jmpy',
        'jmpy.common',
        'jmpy.plotting',
        'jmpy.modeling'
        ],
      package_data={'jmpydev': ['*.html', '*.ipynb', '*.rst']},
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'scipy',
          'patsy',
          'statsmodels',
          'pymc'
      ],
      zip_safe=True)