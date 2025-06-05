from setuptools import setup, find_packages

setup(
    name="ml-pipeline-orchestrator",
    version="0.1.1",
    author="Darshan Rajopadhye",
    author_email="rajopadhye.d@northeastern.edu",
    description="A lightweight ML pipeline orchestrator for managing workflow execution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/therrshan/ml-orchestrator",
    packages=find_packages(),
    package_data={
        'ml_orchestrator': ['../examples/**/*'],  # Include examples
    },
    include_package_data=True,
    install_requires=[
        "PyYAML>=6.0",
        "click>=8.0.0",
        "colorama>=0.4.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ml-orchestrator=ml_orchestrator.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)