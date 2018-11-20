from setuptools import setup, find_packages

install_requires = [
    # "gurobipy",  	# install this manually
    # "alib",      	# install this manually
    # "vnep_approx" , 	# install this manually 
    "matplotlib",
    "numpy",
    "click",
    "pyyaml",
    "jsonpickle",
]

setup(
    name="evaluation-acm-ccr-2019",
    # version="0.1",
    packages=["evaluation_acm_ccr_2019"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "evaluation-acm-ccr-2019 = evaluation_acm_ccr_2019.cli:cli",
        ]
    }
)
