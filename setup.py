"""Python setup.py for helicon package"""
import io
import os
from setuptools import find_packages, setup

def read_file(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

def read_version(path):
    return read_file(path).split("\n")[0].split("=")[-1].strip().strip('"')

def read_requirements(path):
    return [
        line.strip()
        for line in read_file(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name="helicon",
    version=read_version("src/helicon/__version__.py"),
    description="A collection of tools for cryo-EM 3D reconstruction of helical structures",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Jiang Lab", author_email="",
    url="https://github.com/jianglab/helicon",
    package_dir={'': 'src'},
    packages=find_packages(exclude=["tests", ".github", ".git", ".sync"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["helicon = helicon.helicon:main"]
    },
    extras_require={},
)
