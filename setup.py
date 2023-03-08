from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[requirement.replace('\n', '') for requirement in requirements ]

        if "-e ." in requirements:
            requirements.remove('-e .')
setup(
    name = 'ML-project',
    version = '0.0.1',
    author = 'Shuchita',
    author_email= 'mishra.shu@northeastern.edu',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)
