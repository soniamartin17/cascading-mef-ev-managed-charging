# Cascading Marginal Emissions Factor Charging Control

[![DOI](https://zenodo.org/badge/706858319.svg)](https://zenodo.org/doi/10.5281/zenodo.13356990)

## Problem Statement

Managing electric vehicle (EV) charging to reduce emissions benefits both EVs and the grid: (1) to further decarbonize transportation and make the switch from an ICE to an EV better; and (2) to use the flexibility in EV charging as a tool to acclerate the grid's transition and the integration of lower carbon resources. 

Many authors have argued that marginal and average emission factors (MEFs and AEFs) are the best signal for this managed charging. In this research project we aim to answer the questions: 
- How effective are the MEF and AEF methods in present day and future grid conditions?
- Until what level of adoption are EVs on the margin? 
- What happens with the MEF once EVs are no longer a marginal load? 

For scenarios when the MEF and AEF methods fail, we present the Cascading MEF method which ensures emissions reductions in a WECC case study..

## About

Please contact Sonia Martin (soniamartin@stanford.edu) with questions about this repository. 

Research team: Sonia Martin (Stanford), Siobhan Powell (ETH Zurich), and Ram Rajagopal (Stanford) 

This research project was funded by Volkswagen and the Stanford Bits & Watt's EV50 Initiative. 

This code accompanies a paper submitted to Nature Energy entitled "Beyond Marginal: Emissions Signals for Green Charging with Growing Electric Vehicle Adoption".

## License 

This code is licensed under the CC BY-NC-SA 4.0 license. The legal code can be found here: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

## Code

### Structure

This repository contains two folders: 
1. GridInputData
2. Figures

The GridInputData folder includes EIA, CEMS, eGRID, FERC, and other data needed to generate the grid model. The output object is saved in the Data folder.

The Figures folder contains notebooks and figures contained in the paper submission.

To run the code, EV charging data must be placed in a folder called "Data". For privacy and legal reasons, we cannot publish raw or processed data used for this project.

In the main repository we have the two main model classes: 
1. charging.py
2. simple_dispatch.py

There are also files labeled 1 through 5 for data preprocessing and the algorithm files, starting with "run_optimization". 

### Running Instructions

Run files 1-5 for data preprocessing, adding in correct data paths as needed. Note that the desired date range used for the rest of the simulation must be specified in "5_calculate_uncontrolled.py".
Choose which of the optimization scenarios to run: 2020 (unlabeled) or 2030, and "MEF, AEF, Cascade, or Daytime". Run file!

## Package Versions

Python 3.9.0

Pandas 1.3.1

Scipy 1.6.3

CVXPY 1.2.1

Scikit-learn: 1.4.0

Note: the exact Python and Pandas versions are important for the grid object "pickling".