from setuptools import setup, find_packages


setup(name="conflux",
      version=0.1,
      description="time series utility library for Python",
      url="https://github.com/kweimann/conflux",
      author="Kuba Weimann",
      author_email="kuba.weimann@gmail.com",
      license="MIT",
      packages=find_packages(),
      install_requires=["numpy"],
      zip_safe=False)
