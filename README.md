
This project includes all the codes and files for development of classical pair potentials based on force matching to DFT calculations. The examples shown here are silica and silicates, but it can be also applied to other materials as long as the pair approximation stands.

Please refer to the published paper for more details. 
Zheng Yu, Ajay Annamareddy, Dane Morgan, Bu Wang; How close are the classical two-body potentials to ab initio calculations? Insights from linear machine learning based force matching. J. Chem. Phys. 7 February 2024; 160 (5): 054501. https://doi.org/10.1063/5.0175756 

Python packages required: numpy, scipy, sklearn

# Data 
Data for training the force fields (FF) for silica and silicates. They are available for download from Zenodo via [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8329308.svg)](https://doi.org/10.5281/zenodo.8329308). 


# Data_extraction
An example AIMD run with VASP

Codes for 1) collecting forces (output) from AIMD simulations; 2) computing input features based on the local structures 

# Potential

- Codes for training the Ridge model and evaluating the atomic pair interactions

- Codes for generating the FF based on the weights of the Ridge model

- FF for silica (checked and can be used directly with LAMMPS)

- FF for sodium silicates and boron silicates (can be used but might need more benchmark testing)

- An example MD run with generated FF 
