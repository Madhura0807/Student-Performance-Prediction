from setuptools import find_packages, setup

setup(
    name="student-performance-prediction",
    version="1.0.0",
    description="Flask app for predicting student academic performance",
    packages=find_packages(),
    install_requires=[
        "Flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "catboost",
        "xgboost",
        "dill",
        "joblib",
    ],
)
