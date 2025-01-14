# Probabilistic Models of Travel Times in Shared Mobility

This repository contains the implementation of probabilistic machine learning models for analyzing travel time variability in Shared Mobility Services (SMS) and evaluating its impact on accessibility. The research focuses on comparing different ML approaches to predict travel time distributions and assess transportation accessibility in low-density areas.

## Overview

This project addresses the challenges of evaluating accessibility in Shared Mobility Services by:
- Implementing probabilistic ML models to predict travel time distributions
- Creating time-expanded graphs for accessibility analysis
- Combining conventional public transport and SMS networks
- Evaluating accessibility improvements in low-density areas

## Models Implemented

The following probabilistic ML models are implemented to learn travel time distributions:

1. **Linear Regression with Gamma Distribution (LRGD)**
   - Basic linear regression predicting Gamma distribution mean
   - Uses route distance as single feature

2. **Conditional Kernel Density Estimation (CKDE)**
   - Non-parametric method grouping similar routes
   - Fits Gamma kernels to each group's travel times

3. **Random Forest Regressor with Gamma Distribution (RFRGD)**
   - Combines multiple trees for Gamma distribution fitting
   - Uses shape α = (µ/σ)² and rate β = µ/σ² parameters

4. **Bayesian Neural Network (BNN)**
   - Two-layer neural network with Student's t-distribution priors
   - Estimates Gamma distribution parameters using ELBO loss

### Algorithm Details

#### Algorithm 1: Linear Regression with Gamma Distribution
<img src="algorithms/algo1_lrgd.png" alt="Linear Regression with Gamma Distribution Algorithm" width="800"/>

#### Algorithm 2: Conditional Gamma Kernel Density Estimation
<img src="algorithms/algo2_ckde.png" alt="Conditional Gamma Kernel Density Estimation Algorithm" width="800"/>

#### Algorithm 3: Random Forest with Gamma Distribution
<img src="algorithms/algo3_rfrgd.png" alt="Random Forest with Gamma Distribution Algorithm" width="800"/>

#### Algorithm 4: Bayesian Neural Network with Gamma Distribution
<img src="algorithms/algo4_bnn.png" alt="Bayesian Neural Network with Gamma Distribution Algorithm" width="800"/>

For implementation details, please refer to the source code in the respective model directories.
Algorithm 4 Bayesian Neural Network with Gamma Distribution
Require: X = feature matrix, y = travel times, h = hidden dimension
 1: function InitializeBNN(input_dim, hidden_dim):
 2:     // Initialize network with Student-T priors
 3:     df ← 9.82, loc ← 0, scale ← 1.004
 4:     W₁ ~ StudentT(df, loc, scale)hidden_dim × input_dim
 5:     b₁ ~ StudentT(df, loc, scale)hidden_dim
 6:     W₂ ~ StudentT(df, loc, scale)2 × hidden_dim
 7:     b₂ ~ StudentT(df, loc, scale)2
 8: end function
 9: function Forward(x):                              ▷ Single forward pass
10:     h ← ReLU(W₁x + b₁)                           ▷ Hidden layer
11:     [log_mean, log_shape] ← W₂h + b₂             ▷ Output layer
12:     // Transform parameters
13:     mean ← exp(log_mean)
14:     shape ← exp(softplus(log_shape)) + 1.0
15:     // Compute rate with constraints
16:     rate ← clamp(shape/mean, 10⁻³, 100)
17:     shape ← mean × rate                          ▷ Adjust shape
18:     return shape, rate
19: end function
20: function Train(X, y, epochs, batch_size):
21:     guide ← AutoDiagonalNormal(model)            ▷ Variational distribution
22:     optimizer ← Adam(lr)
23:     for epoch ← 1 to epochs do
24:         for (Xb, yb) in BatchLoader(X, y, batch_size) do
25:             loss ← -ELBO(Xb, yb)                 ▷ Negative ELBO
26:             optimizer.step(loss)
27:         end for
28:     end for
29: end function
30: function PredictDistribution(Xnew):
31:     // Generate multiple samples using guide
32:     for s ← 1 to num_samples do
33:         shapes, rates ← Forward(Xnew)
34:         ys ~ Gamma(shapes, rates)
35:     end for
36:     μ ← mean(ys)                                 ▷ Mean prediction
37:     σ ← std(ys)                                  ▷ Uncertainty
38:     return ys, μ, σ
39: end function
```

## Visual Overview

![Accessibility Analysis Example](images/accessibility_map.png)
<!-- Local image from your repository -->

<img src="images/travel_time_plot.png" alt="Travel Time Distribution" width="600"/>
<!-- Local image with custom size -->

![Model Comparison](https://raw.githubusercontent.com/username/repo/main/images/model_comparison.png)
<!-- Image from URL -->

You can also add image descriptions and titles:

| Accessibility Map | Travel Time Distribution |
|:----------------:|:-----------------------:|
| ![](images/map1.png) | ![](images/plot1.png) |
| *Figure 1: Spatial distribution of accessibility scores* | *Figure 2: Travel time variability* |

## Key Features

- **Travel Time Analysis**: Captures day-to-day, within-the-day, and vehicle-to-vehicle variability
- **Accessibility Calculation**: Uses isochrone-based measures with hexagonal grid tessellation
- **Time-Expanded Graphs**: Combines CPT and SMS networks for comprehensive accessibility analysis
- **Walking Integration**: Incorporates walking as a transportation mode with configurable parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/reliable-accessibility-framework.git

# Navigate to project directory
cd reliable-accessibility-framework

# Install required dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example code for model training and evaluation
from models import LRGD, CKDE, RFRGD, BNN
from utils import evaluate_model

# Train models
model = RFRGD()  # or any other implemented model
model.fit(X_train, y_train)

# Evaluate performance
results = evaluate_model(model, X_test, y_test)
```

## Data Requirements

The framework expects the following data:
- SMS trip data with timestamps and geolocation
- GTFS data for conventional public transport
- Opportunity locations (schools, workplaces, etc.)
- Area tessellation into hexagonal grid cells

## Performance Metrics

Models are evaluated using multiple metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Negative Log-Likelihood (NLL)
- Kullback-Leibler Divergence (KLD)

## Results

The RFRGD model showed the best overall performance:
- MAE: 176.33 seconds
- RMSE: 255.42 seconds
- NLL: 6.97
- KLD: 10.04

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ourahou2024probabilistic,
  title={Probabilistic Models of Travel Times in Shared Mobility for Evaluating Accessibility},
  author={Ourahou, Mohamed and Araldo, Andrea and Zigrand, Louis and Carneiro Viana, Aline},
  journal={Transportation Research},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work has been funded by Région Île-de-France. Data was provided by Padam Mobility.

## Contact

- Mohamed Ourahou - mohamed.ourahou@telecom-sudparis.eu
- Project Link: [https://github.com/medourahou/Reliable-Accessibility-framework](https://github.com/medourahou/Reliable-Accessibility-framework)
