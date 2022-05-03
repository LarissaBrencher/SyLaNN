from setuptools import setup
import setuptools
import versioneer

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name="SyLaNN",
    version= "0.0.1", # versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Larissa Brencher",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/LarissaBrencher/SyLaNN.git",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=["numpy", "torch", "sympy", "json", "random", "datetime", "time", "inspect", "os"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: Microsoft :: Windows"
    ]
)