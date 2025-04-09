from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ezprompt",
    version="0.0.1",
    author="Jataware Corp",
    author_email="ben@jataware.com",
    description="Minimial prompting library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jataware/ezprompt",
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.12",
    install_requires=[
        "litellm",
        "pydantic",
        "rich",
        "tqdm",
        "tenacity",
        "json_repair"
    ],
)