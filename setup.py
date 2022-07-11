from setuptools import setup

setup(
   name='bayesinverse',
   version='0.1',
   description='',
   author='Robert Maiwald',
   author_email='rmaiwald@iup.uni-heidelberg.de',
   packages=['bayesinverse'],  #same as name
   install_requires=[], #external packages as dependencies
   scripts=['python-scripts/control.py']
)