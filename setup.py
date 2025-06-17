from setuptools import setup,find_packages
from typing import List

def get_requirements()->List[str]:
    requirement_list:List[str]=[]
    try:
        with open("requirements.txt","r") as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")
    return requirement_list

setup(
    name='Network_security',
    version='1.0.0',
    packages=find_packages(),
    install_requires=get_requirements(),
    author='Gaurav',
)