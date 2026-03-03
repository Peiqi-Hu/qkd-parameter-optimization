# QKD Parameter Optimization
## Overview
This project implements a research-oriented framework for parameter optimization in BB84 quantum key distribution (QKD) protocol. The goal is to investigate how classical optimization and machine learning techniques can accelerate parameter tuning in QKD systems. In practical QKD deployments, key rate depends on multiple physical and prrotocol parameters (e.g., channel loss, basis bias, error rates). It is computationally expensive to identify optimal parameters using traditional brute-force approach. 

Inspired by *Wang(2019) on machine learning for optimal parameter prediction in QKD*(1) , this project builds a small-scale experimental pipeline that consists of followings:
1. Simulates BB84 QKD
2. Generates optimization datasets via classical search
3. Trains machine learning model to predict near-optimal parameters
4. (optional) Optimization
5. Extends toward Finite-key analysis

The framework is designed as a small-scale research prototype suitable for further extension toward realistic QKD modeling.

## Motivation
In QKD systems, the secret key rate depends on: (reference 2)
- Quantum bit error rate (QBER)
- Channel transmittance
- Basis choice probability
- Signal intensity
- Finite-key statistical fluctuations (for finite key analysis)

Traditional optimization relies on: (!!!!!!! to add more details)
- Brute-force search or local search algorithm

Wang's 2019 (1) proposed a neural network that offers a acceleration in parameter optimization by learning a mapping from system parameters (input) to optimal parameters (output) based on dataset generated using traditional optimization method. 

This project experimentally explores the integration of a physically simulated BB84 protocol with the machine learning framework proposed by Wang (2019), using a small-scale dataset, with potential extensions to incorporate different statistical fluctuation bounds.

## Project Stages
#### brief
- Implement BB84 at circuit level using Qiskit
- Verify QBER trends under:
  - Depolarizing noise
  - Basis bias
  - Channel loss (simulated)
- Compare Qiskit QBER to analytical QBER
- Show they match within statistical error (rephrase)
- Switch to analytical model for dataset generation
- Apply ML
- Validate ML optimal parameters using Qiskit again

### Stage 1 - Parameterized BB84 Simulator
- Implement theoretical key rate function (asymptotic ideal BB84 formula)
- Implement analytical noisy channel model
  - Plot:
    - Key rate vs QBER
    - Key rate vs transmittance
  - Validate monotonicity:
    - Increasing noise → decreasing key rate
    - Increasing loss → decreasing key rate
- Implement BB84 using Qiskit
  - Confirm QBER simulation matches theory.
- Parameterized basis bias probability
- Noise model injection
- Compute:
  - QBER
  - Gain
  - Secret key rate (asymptotic at this stage)
- Validate simulation by:
  - Compare with theoretical Gain, QBER expections
  - 
- Output:
  - Structured simulation results
  - Configurable experiment settings

### Stage 2 - Dataset Generation via Local Search
- Define small parameter space
- Apply:
  - Grid Search? Hill climbing refinement?
- Record:
  - Input parameters
  - Observed QBER ? and more?
  - Computed key rate
  - Best-found parameters 
- Goal: Generate a small but structured dataset for machine learning modeling in step 3.

### Step 3 - Machine Learning Model
- Train model
  - Neural network?
  - regression?
- Evaluate:
  - Mean squared error
- Analyze:
  - 
- Goal: Evaluate whether ML can approximate optimal parameter mapping.

### Step 4 (optional) - Bayesian? Optimization

### Step 5 - Feed optimal parameters back to qiskit
- using qiskit for validation 

### Step 6 (Future work) - Finite-key Analysis
Extend key rate calculation to include:
- Statistical fluctuation bounds
- Security parameter ε

- Goal: Make the model closer to real QKD research conditions.


## Architecture
(diagram)

## Concepts
(formulas + papers)

## Main functions and demo screenshots

## Result 
(plots)

## Summary

## Technical report
(4-6 pages pdf with results and methods)

## Reference
(1) Wang et al. (2019)
(2) QKD parameters reference
