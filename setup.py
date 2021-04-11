import setuptools

#from setuptools import setup

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='pyangstromRT',
    url='https://github.com/yuanyuansjtu/Angstrom-method',
    author='Yuan Hu',
    author_email='yuanyuansjtu@yahoo.com',
    # Needed to actually package something
    #packages=['pyangstromHT'],

    packages=setuptools.find_packages(),

    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.0',
    # The license can be anything you like
    license='MIT',
    description='A python code to execute Angstrom method at room temperatures',
    classifiers = [
                      "Programming Language :: Python :: 3",
                      "License :: OSI Approved :: MIT License",
                      "Operating System :: OS Independent",
                  ],
    python_requires='>=3.6',
)
