from setuptools import setup, find_packages

setup(name='servier',
      version='1.0',
      description='A molecule classifier based on'
                  'ECFP fully-connected neural network',
      author='Hugo Defois',
      author_email='defois.hugo@gmail.com',
      packages=find_packages(),
      install_requires=["pandas", "numpy", "rdkit-pypi",
                        "sklearn", "tensorflow", "keras",
                        "keras-tuner"],
      python_requires=">=3.6",
      entry_points={'console_scripts': ['servier=servier:main.main']})
