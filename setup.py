from setuptools import setup, find_packages

setup(
    name="yoyo-agent",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "yoyo=batch_agent.cli:main",
            "batch-agent=batch_agent.cli:main",
        ],
    },
)
