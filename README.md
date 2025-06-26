# Cascading Marginal Emissions Factor Managed Charging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15748366.svg)](https://doi.org/10.5281/zenodo.15748366)

## Problem Statement

Managing electric vehicle (EV) charging to reduce emissions benefits both EVs and the grid: (1) to further decarbonize transportation and make the switch from an ICE to an EV better; and (2) to use the flexibility in EV charging as a tool to acclerate the grid's transition and the integration of lower carbon resources. 

Many authors have argued that marginal and average emission factors (MEFs and AEFs) are the best signal for this managed charging. In this research project we aim to answer the questions: 
- How effective are the MEF and AEF methods in present day and future grid conditions?
- Until what level of adoption are EVs on the margin? 
- What happens with the MEF method once EVs are no longer a marginal load? 

For scenarios when following the MEF and AEF methods fail, we present the Cascading MEF method which ensures emissions reductions in a WECC case study.

## About

Please contact Sonia Martin (soniamartin@stanford.edu) with questions about this repository. 

Research team: Sonia Martin (Stanford), Siobhan Powell (ETH Zurich), and Ram Rajagopal (Stanford) 

This research project was funded by Volkswagen and the Stanford Bits & Watt's EV50 Initiative. 

This code accompanies a paper submitted to Nature Communications entitled "Cascading Marginal Emissions Signals for Green Charging with Growing Electric Vehicle Adoption".

## License 

This code is licensed under the CC BY-NC-SA 4.0 license. The legal code can be found here: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

## Code

### Initializing Python Environment

Instructions are provided to create a virtual environment created with .venv in Visual Studio code on a Windows PC.

1) Download Python 3.9 at https://www.python.org/downloads/release/python-390/ or from the Microsoft Store.
2) Download VSCode with build tools at https://code.visualstudio.com/download (works for Windows or Mac)
3) To create a new virtual environment on VS Code, press View -> Command Palette and search for Python:Create New Environment. Click on Venv for the .venv virtual environment. 
4) Download the correct package versions using the commands below:

pip install pandas==1.3.1

pip install scipy==1.6.3

pip install cvxpy==1.2.1

pip install matplotlib==3.7

pip install scikit-learn==1.4.0

pip install ipykernel

pip install openpyxl

pip install mosek

#### Package Versions

If creating your own virtual environment, please use these packages to ensure correct dependencies. Note: the exact Python and Pandas versions are important for the grid object "pickling". After downloading the packages, you should have Numpy version 1.22.4. This should be installed automatically with the scikit-learn installation; this is why packages must be installed in the specified order.

Python 3.9

Pandas 1.3.1

Scipy 1.6.3

CVXPY 1.2.1

Matplotlib 3.7

Scikit-learn 1.4.0

ipykernel

Openpyxl

Mosek 11.0.22

### Optimization Solver

Optimization with CVXPY is run with the MOSEK solver. The license is available for free for academic users and offers a 30 day free trial for private users. Please see https://www.mosek.com/resources/getting-started/ to download a license. The .lic file from MOSEK must be stored in the correct folder for CVXPY to correctly run. There is a code block that will print an error message if the license is not present. 


### Structure

This repository contains three folders: 
1. GridInputData
2. Data
3. Figures

The GridInputData folder includes EIA, CEMS, eGRID, FERC, and other data needed to generate the grid model. The output object is saved in the Data folder.

The Data folder contains synthetic EV data and, once the optimization is run, managed charging optimization results. For privacy and legal reasons, we cannot publish the actual raw or processed data used for this project.
 
The Figures folder contains notebooks to generate the figures contained in the paper submission. The figures (except figure S4) are coded to plot agnostic of existence of a charging timer in the results.

In the main repository we have the two main model classes: 
1. charging.py
2. simple_dispatch.py

create_synthetic_data.py is the file that generates the provided synthetic data file.

There are also files labeled 1 through 5 for data preprocessing and the algorithm files, starting with "run_optimization". Lastly, all code can be called through the file 0_run_experiment.ipynb.

### Running Instructions

Run all code cells in 0_run_experiment.ipynb. The file requires the user to choose whether or not a charging timer should be incorporated, add synthetic data file paths, and choose which type of optimization to run.

Note that the Mosek check is important to ensure correct results. (The optimization code will not throw an error if the Mosek file is missing, so you should confirm with the code block check instead.)

By default, the simulation will run for the set of the number of EVs in the paper and will run for 15 trials. These values can be changed in the "run_optimization" .py files themselves.

Once the code has completed, the results will be saved in named folders under the Data folder.

