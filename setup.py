from setuptools import setup

setup(name='jmpy',
      version='0.10.4',
      description='JMP Style Plotting and Modeling',
      url='https://github.com/beltashazzer/jmpy',
      author='David Daycock',
      author_email='daycock@gmail.com',
      license='MIT',
      packages=[
        'jmpy',
        'jmpy.common',
        'jmpy.plotting',
        'jmpy.modeling',
        ],
      package_data={'jmpy': ['*.html', '*.ipynb', '*.rst']},
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'scipy',
          'patsy',
          'statsmodels'
      ],
      zip_safe=True)
