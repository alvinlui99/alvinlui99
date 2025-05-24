from setuptools import setup, find_packages

setup(
    name="crypto_trading_bot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "python-binance",
        "python-dotenv",
    ],
    python_requires=">=3.8",
) 