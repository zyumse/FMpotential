
This project includes all the codes and files for development of classical pair potentials based on force matching to DFT calculations. The examples shown here are silica and silicates, but it can be also applied to other materials as long as the pair approximation stands.

Author: Zheng Yu and Bu Wang

Please refer to reference XXX for more details. 

# Data 
Includes data for training the FFs (silica and silicates)

# Prep
codes for 1) collecting forces (output) from VASP AIMD simulations; 2) collect input features based on the local structures

# Potential

- ML codes for training the model and generating the FF 

- FF for silica (well-developed and checked)

- FF for sodium silicates and boron silicates (can be used but might need more benchmark testing)

- example MD run with generated FF 
