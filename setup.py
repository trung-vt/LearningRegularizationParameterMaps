from setuptools import setup

setup(
   name='learning_regularisation_parameter_map',
   version='1.0.0',
   description='Learn regularisation parameter map for image reconstruction using deep learning.',
   author='trung-vt',
   author_email='trung.vuthanh24@gmail.com',
   packages=['networks', 'data', 'metrics', 'scripts', 'results'],
#    packages=['learning_regularisation_parameter_map'], #same as name
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
   install_requires=['wheel', 'torch'], #external packages as dependencies
    # scripts=[
    #     'networks/pdhg',
    # ]
)