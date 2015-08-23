from setuptools import setup

setup(name='jmpy',
      version='0.9.0',
      description='JMP Style Plotting and Modeling',
      url='',
      author='David Daycock',
      author_email='ddaycock@micron.com',
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
      ],
      zip_safe=True)