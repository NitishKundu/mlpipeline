from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    Reads requirements.txt file and returns a list of required packages
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements



setup(
    name='E2EMLPipeline',
    version='0.1',
    author="Nitish",
    author_email="nitishkundu1993@gmail.com",
    description="End to End ML Pipeline",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
            
)