# Using-SVISE-for-State-Estimation

-> TODO:
1. The datagiven to the traiing set needs to be switched with the data obtained
2. The current data is dx = alpha*x - beta*y | dy = delta*x - gamma*y
3. This needs to be replaced by the displacement of sensors 1,2,3......n as a function of distance first and then time.
4. A drift function for the same (probably euler bending equations, FEM equations etc.) need to be modelled.
5. Get inti the SVISE library and check what needs to be manipulated to adjust our equation into the library.
6. Stochastic variables appropriate for this model need to be inserted.
7. Model needs to be tested for observed and unobserved values of x and estimation of displacement needs to be made.
8. This would let us verify if the model is applicable to real world sytems where force and displacement are unknown.

# The following repository has been used as a reference:
https://github.com/coursekevin/svise/blob/tutorials/tutorials/1-intro-to-svise.ipynb
