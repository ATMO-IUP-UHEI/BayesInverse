from setuptools import setup

setup(
   name='bayesinverse',
   version='1.1',
   description='',
   author='Robert Maiwald',
   author_email='rmaiwald@iup.uni-heidelberg.de',
   packages=['bayesinverse'],  #same as name
   install_requires=["scipy", "numpy"], #external packages as dependencies
   scripts=[]
)