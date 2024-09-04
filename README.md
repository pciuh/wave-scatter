# Hypothesis Testing

This project focuses on scatter joint probabilities, properties, and hypothesis testing. The goal is to determine whether significant results can be expected based on the assumed statistical distribution, sample size, error levels, and other relevant factors. This guide outlines the minimum workflow required to conduct the project.

The data used in this project comes from the scatter diagram of wave heights and periods. The source is from the paper: *J. Prendergast, M. Li, and W. Sheng, "A Study on the Effects of Wave Spectra on Wave Energy Conversions," in IEEE Journal of Oceanic Engineering, vol. 45, no. 1, pp. 271-283, Jan. 2020, [doi: 10.1109/JOE.2018.2869636](https://doi.org/10.1109/JOE.2018.2869636).*

## Workflow

1. **Define Statistical Distribution**:
    - Start by defining the statistical distribution of your samples in at least two groups (e.g., treatment and control or treatment1 and treatment2).
    - This is a crucial step, so choose the distribution based on your best knowledge and intuition. You may choose a multivariate distribution if applicable.

2. **Generate Samples**:
    - Generate a number of samples for each group based on the defined distribution.

3. **Add Distortion**:
    - You can introduce some form of distortion in each group, such as noise or measurement error, to simulate real-world data.

4. **Choose a Statistical Test**:
    - Select an appropriate statistical test based on your discipline and the nature of your data.

5. **Check Assumptions**:
    - Ensure that your samples satisfy the assumptions of the chosen statistical test. If they do not, consider selecting an alternative test.

6. **Evaluate Significance**:
    - Determine whether the treatment and control groups show significant differences.

7. **Determine Minimal Sample Size**:
    - Identify the minimal sample size at which your results are statistically significant at a p-value of < 0.05.

8. **Visualize Results**:
    - Prepare a plot that illustrates the experiment and the statistical test results.