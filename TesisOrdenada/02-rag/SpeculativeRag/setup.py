from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="speculative_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Author",
    author_email="teogutter@gmail.com",
    description="Speculative RAG implementation with Pinecone",
    keywords="rag, speculative, pinecone, nlp, ai",
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "speculative-rag=speculative_rag.cli:main",
        ],
    },
) 