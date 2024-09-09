# Hypothesis Testing

This project focuses on the generation of artificial population data derived from scatter joint probabilities of wave properties. Hypothesis testing is applied to selected samples of varying sizes to assess the minimum sample size required for testing the sea development threshold.

This guide outlines the minimum workflow required to conduct the project presented via [jupyter-notebook](https://github.com/pciuh/hypothesis/blob/main/wave_scatter.ipynb).

The data used in this project comes from the scatter diagram of wave heights and periods. The source is from the paper: *J. Prendergast, M. Li, and W. Sheng, "A Study on the Effects of Wave Spectra on Wave Energy Conversions," in IEEE Journal of Oceanic Engineering, vol. 45, no. 1, pp. 271-283, Jan. 2020, [doi: 10.1109/JOE.2018.2869636](https://doi.org/10.1109/JOE.2018.2869636).*

## Workflow

1. **Define Statistical Distribution**:
   - The loaded scatter diagram is assumed to follow a normal joint probability distribution (bivariate)
   - To properly generate samples for hypothesis testing, the covariance matrix and mean values of wave heights and periods were calculated based on the scatter data.
   - The number of samples is split into large and small groups, denoted as *L* and *S*, respectively.

2. **Generate Samples**:
    - A different number of samples, depending on the selected option *L* or *S*, is generated from pre-defined, multivariate distribution
    - Noise is applied to the generated data to make it more realistic.

3. **Choose a Statistical Test**:
   - The statistical one-tailed z-test is used according to the nature of the considered data.

4. **Determine Minimal Sample Size**:
   - Identify the minimal sample size at which your results are statistically significant at a p-value of < 0.05

5. **Visualize Results**:
   - Prepare a plot that illustrates the scatter data and generated population samples
