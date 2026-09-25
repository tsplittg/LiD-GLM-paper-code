# lidglm

A compromise between generalized linear models and unconstrained neural networks. Code for the paper "LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models" (https://arxiv.org/abs/2608.16340)

## Installation

The code was developed under python 3.11.10 , ideally the code should therefore be run in a fresh (conda) virtual environment
using this python version.

The package was developed using poetry and can therefore simply be installed calling 

```bash
$ poetry install
```

in the root folder. Alternatively, 

```bash
$ pip install .
```

can be used.

Some scripts that test semi-structured deep distributional regression models as developed by Rügamer et al. need the
python implementation of these models to function. These can be found as the `PySddr` package under https://github.com/HelmholtzAI-Consultants-Munich/PySDDR . 
Simply install their package using 

```bash
$ pip install .
```

into the same virtual environment as ours.

The script that computes the exact Lipschitz constant for the LiD-GLM on the "Cars" data in Section 4.1.2 (***applications-> Cars_application -> cars_exlipbab_computation.py***)
needs the `exlipbab` package (https://github.com/tsplittg/ExLipBaB_Code) to compute the constant from previoulsy saved network weights. As this script in turn does not need the lidglm package
to run, it is likely easiest to simply install `exlipbab` into a separate virtual should package version issues arise. The installation of
`exlipbab` is detailed on their Github page.

## Replication of the Experiments:

The folder ***applications*** contains the scripts which were used to create the illustrations in our paper as well as the code of our experiments. Under ***src->lidglm*** is our implementation of LiD-GLM.

1) The code for the simulation study to reconstruct known linear effects (Sec. 4.3.1) can be found under ***applications-> PHO_simul_study***. First, the ***pho_linear_simul*** script needs to be run which saves the results in the ***outputs*** folder.
The ***PHO_linear_plots*** script reads these results and creates a corresponding boxplot.
2) 

## License

`lidglm` was created by Tom Splittgerber. It is licensed under the terms of the MIT license.

## Credits

The folder layout for `lidglm` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
