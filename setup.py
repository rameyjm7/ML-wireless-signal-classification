from setuptools import setup, find_packages

setup(
    name="ml_wireless_classification",
    version="0.1.0",
    description="ML Wireless Classification",
    author="Jacob Ramey",
    author_email="rameyjm7@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "scikit-learn",
        "tensorflow",
        "numpy",
        "matplotlib",
        "PyWavelets",
    ],
    entry_points={
        "console_scripts": [
            "start=ml_wireless_classification.__init__:create_app",
        ],
    },
)
