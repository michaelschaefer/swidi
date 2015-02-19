# SwiDi

**Please be advised that this software is still under heavy development so that interfaces may change at any time.**

SwiDi is a Python framework for numerical simulations of stochastic differential equations with Markovian switching. For
a detailed description of the underlying mathematical model as well as the API we refer to the official documentation.

## Documentation

Currently, the documentation must be built by the user himself. This is achieved by calling
  
    make <target>
    
inside the `docs/` folder of the repository. `<target>` can be replaced by many values, the most eminant examples 
probably being `html` and `latexpdf`. Please be advised that you may need additional third-party software in order to
build the chosen target (like, obviously, a LaTeX environment for the target `latexpdf`). 