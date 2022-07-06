# Maxwellfirstorder
non symmetric first order in time FEM BEM coupling for the full space Maxwell system 


Further details can be found in my thesis https://publikationen.bibliothek.kit.edu/1000133728

run errorplottime.py or errorplotspace.py for a experimental order of convergence experiment either in time or in space.  
The computations are done in the MLLGfunc... files. Inputs are discretization parameters and material parameters, output is the (coefficients of the) solution on the corresponding time points.  

The software runs with 
Fenics 2019.1.0 
Bempp 3.3.4
