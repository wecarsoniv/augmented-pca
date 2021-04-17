import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    readme_description = fh.read()

setuptools.setup(
    name="augmented-pca", # Replace with your own username
    version="0.0.1",
    author="Billy Carson",
    author_email="wec14@duke.edu",
    description="Implementations of adversarial and supervised linear factor models.",
    long_description=readme_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Issue Tracker": "https://github.com/wecarsoniv/augmented-pca/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

