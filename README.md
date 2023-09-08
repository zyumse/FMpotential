
This project includes all the codes and files for development of classical pair potentials based on force matching to DFT calculations. The examples shown here are silica and silicates, but it can be also applied to other materials as long as the pair approximation stands.

Please refer to our paper XXX for more details. 

# Data 
Data for training the force fields (FF) for silica and silicates

# Prep
An example AIMD run with VASP

Codes for 1) collecting forces (output) from AIMD simulations; 2) computing input features based on the local structures 

# Potential

- ML codes for training the model and generating the FF 

- FF for silica (checked and can be used directly with LAMMPS)

- FF for sodium silicates and boron silicates (can be used but might need more benchmark testing)

- An example MD run with generated FF 
