# Angstrom-method for Thermal Diffusivity Measurement using Infrared Thermography
This code process temperatures obtained from IR thermography videos, and applies a modified Angstrom method (E Lopez-Baeza et al 1987 J. Phys. D: Appl. Phys. 20 1156) to calculate thermal diffusivity based on the recording. Here I demonstrate the how to work with Flir software, but it works for any IR software so long as you can export IR video as multiple csv files.

For details, please check the Jupyter notebook for instructions how to use this code.

My journal paper regarding experimental section of this code is https://doi.org/10.1115/1.4047145 (see citation detail below). If you find it beneficial for your application, I encourage to cite my current paper along the original paper which for modified Angstrom method theory. If you find this code useful, please also cite this code, the DOI is: https://doi.org/10.5281/zenodo.4587863.

Hu, Y., and Fisher, T. S. (May 29, 2020). "Accurate Thermal Diffusivity Measurements Using a Modified Ångström's Method With Bayesian Statistics." ASME. J. Heat Transfer. July 2020; 142(7): 071401. https://doi.org/10.1115/1.4047145


Installation:

pip install git+https://github.com/yuanyuansjtu/Angstrom-method.git#egg=pyangstromRT

You can import the useful function:from pyangstromHT import high_T_angstrom_method.Then you can call functions high_T_angstrom_method.interpolate_light_source_characteristic so that this code can do stuff for you.

Just in case the repository is updated, please use this command to keep your local copy up to date:

pip install --upgrade git+https://github.com/yuanyuansjtu/Angstrom-method.git#egg=pyangstromRT

It can also be installed to use on colab:

!pip install git+https://github.com/yuanyuansjtu/Angstrom-method.git#egg=pyangstromRT
