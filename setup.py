from setuptools import setup

setup(name='jmpy',
      version='0.10.0',
      description='Plotting and Modeling Package',
      url='https://github.com/beltashazzer/jmpy',
      download_url = 'https://github.com/beltashazzer/jmpy/tarball/0.1',
      author='David Daycock',
      author_email='daycock@gmail.com',
      license='MIT',
      packages=[
        'jmpy',
        'jmpy.common',
        'jmpy.plotting',
        'jmpy.modeling',
        'jmpy.bayes'
        ],
      package_data={'jmpy': ['*.html', '*.ipynb', '*.rst', '*.md']},
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