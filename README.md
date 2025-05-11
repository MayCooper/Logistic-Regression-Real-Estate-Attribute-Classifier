
# Structural Feature Inference for Real Estate - Logistic Regression

## Overview

This project applies supervised machine learning, specifically Logistic Regression, to predict whether a residential property has a garage based solely on its square footage. This type of binary classification is relevant in real estate analysis where certain features, such as garage availability, may influence buyer decisions and property value. By building a statistically sound logistic model, we aim to explore the relationship between home size and structural features.

---

## Project Objectives

- **Primary Goal**: To investigate and model how square footage alone contributes to the likelihood that a home includes a garage.
- **Use Case**: Assist housing developers, appraisers, and real estate professionals in understanding how property size correlates with garage availability, informing both market pricing and development strategy.
- **Methodology**: The project follows a complete statistical workflow including data inspection, exploratory analysis, binary encoding of target variables, training/testing split, logistic model fitting using MLE (maximum likelihood estimation), and interpretation of coefficients and significance.

---

## Research Question

Can the square footage of a house be used to reliably predict the presence or absence of a garage?

This question is significant in assessing whether size alone can be an effective indicator of property features or if multiple variables are needed to make useful inferences in housing analytics.

---

## Dataset Summary

The dataset includes 5,600 individual housing records with relevant attributes for logistic regression. The analysis focuses on the following:

- **SquareFootage**: A continuous variable representing the total indoor living area of the home.
- **Garage**: A binary categorical variable indicating whether the house has a garage (Yes or No).

The dataset is balanced with nearly equal representation of homes with and without garages. This makes it suitable for binary classification without needing resampling techniques.

### Key Statistics

| Metric               | Value       |
|----------------------|-------------|
| Observations         | 5,600       |
| Avg Square Footage   | ~1,050 ft²  |
| Garage (Yes %)       | ~54.2%      |
| Min Square Footage   | ~500 ft²    |
| Max Square Footage   | ~2,800 ft²  |

---

## Data Exploration and Preparation

### Key Insights

Initial analysis revealed that square footage is moderately right-skewed, with a concentration of homes between 800–1,200 ft². The binary target variable, `Garage`, was distributed fairly evenly across the dataset.

### Feature Distributions

#### Distribution of Price

![image](https://github.com/user-attachments/assets/50e61bfe-c3b6-4106-acbb-5d6ff3854538)

The distribution of housing prices is right-skewed, with most properties clustered between \$150K and \$400K. A long tail extends beyond \$600K, indicating outliers or luxury homes.

---

#### Distribution of SquareFootage

![image](https://github.com/user-attachments/assets/86079345-d6a7-4b4a-b35d-8e40e24f703c)

Square footage is also right-skewed, with a high density of homes around 550 ft², followed by a gradual decline. This skew suggests that many homes are compact or mid-sized, with fewer large properties.

---

#### Count of Garage Presence

![image](https://github.com/user-attachments/assets/a8c04593-a3b0-4cc9-b813-f841f2444f1e)

There is a class imbalance, with more homes lacking a garage than those with one. This impacts classification performance and may require resampling techniques in future models.

---

#### Price Distribution by Garage

![image](https://github.com/user-attachments/assets/dd5b8bc5-7d5d-4dd0-bc7b-9a054a03bb94)

Homes with and without garages show similar price distributions, though the median and upper quartile prices appear slightly higher for garage-equipped homes. However, variance is wide, and overlap is substantial.

---

#### SquareFootage by Garage

![image](https://github.com/user-attachments/assets/3c4af1f7-ad54-4acb-b204-2a283df8a873)

Homes with garages tend to have slightly larger square footage on average, but the distribution overlap indicates that square footage alone is not a strong distinguishing factor.

---

#### Price vs. Square Footage Colored by Garage

![image](https://github.com/user-attachments/assets/75071142-1bb3-4264-ad8a-49fdab6c6e8e)

This scatter plot shows that price generally increases with square footage, but there's no clear visual separation between homes with and without garages. Garage presence is not strongly tied to price or size alone.

### Preprocessing Steps

1. **Data Cleaning**: Verified there were no missing values in the key columns.
2. **Encoding**: The `Garage` variable was encoded as binary (1 = Yes, 0 = No).
3. **Splitting**: Data was randomly split into a **training set (70%)** and **testing set (30%)** to evaluate generalization.

These steps ensured that the data was properly structured for modeling and evaluation.

---

## Modeling Approach

A logistic regression model was used due to the binary nature of the target variable (`Garage`). The logistic function models the log-odds of the target being 1 (garage present), which is appropriate for classification problems with a probabilistic interpretation.

### Final Model Equation

```
Logit(P(Garage=1)) = -0.4205 - 0.0001 * SquareFootage
```

### Coefficient Interpretation

- The intercept term (-0.4205) represents the baseline log-odds of having a garage when square footage is 0.
- The negative coefficient for `SquareFootage` suggests that as square footage increases, the likelihood of having a garage slightly decreases, which may seem counterintuitive and warrants further investigation or additional variables.

---

## Model Evaluation

### Performance Metrics

| Metric            | Value         |
|-------------------|---------------|
| Log-Likelihood    | -3659.0       |
| Pseudo R-squared  | 0.00066       |
| p-value (model)   | 0.027         |
| Converged         | Yes           |

### Interpretation

- The model's p-value (0.027) indicates statistical significance at the 5% level, meaning the relationship is unlikely due to random chance.
- However, the **pseudo R-squared** value is very low, implying that square footage explains only a tiny portion of the variance in garage presence.
- This suggests the need to include more explanatory features for a useful predictive model.

---

## Key Insights

### For Developers and Analysts

- Relying solely on square footage is insufficient for determining garage presence.
- Future work should incorporate additional predictors such as number of bedrooms, property type, or zoning classification.

### For Real Estate Market Applications

- Structural size (square footage) should not be used in isolation for property classification tasks.
- Logistic models can be useful for classification when properly specified with multiple meaningful predictors.

---

## Tools and Technologies

- **Pandas, NumPy** – For data manipulation and basic exploration.
- **Matplotlib, Seaborn** – Used to create histograms and distribution plots for exploratory data analysis.
- **Statsmodels** – Employed for fitting the logistic regression model and obtaining summary statistics such as p-values and confidence intervals.
- **Scikit-learn** – Used for splitting the dataset and general machine learning workflow support.

---

## Results Interpretation

### Logistic Regression Output

The final logistic regression model was trained using **SquareFootage** as the sole predictor of whether a home has a **Garage** (binary: Yes or No). Below is a summary of the statistical output:

| Coefficient     | Estimate   | Std. Error | z-value | p-value | 95% CI                |
|----------------|------------|------------|---------|---------|------------------------|
| Intercept      | -0.4205    | 0.074      | -5.669  | 0.000   | [-0.566, -0.275]       |
| SquareFootage  | -0.0001    | 0.0000658  | -2.200  | 0.028   | [-0.00016, -0.0000158] |

### Key Takeaways

- The **negative coefficient** on `SquareFootage` suggests that, within this dataset, as square footage increases, the odds of a property having a garage **slightly decrease**.
- The **p-value for `SquareFootage` is 0.028**, which is **statistically significant at the 5% level**, meaning there is evidence to suggest that square footage has a real, albeit very small, effect on garage presence.
- The **pseudo R² is extremely low** (0.00066), indicating that the model explains **less than 0.1%** of the variance in garage presence. In practical terms, the model has **very limited predictive power**.

---

## Model Optimization Insights

Three model selection strategies were evaluated:

- **Forward Stepwise Selection** (Best AIC = 7322.01)
- **Backward Stepwise Elimination**
- **Recursive Feature Elimination**

All methods selected the same model with `SquareFootage` as the only predictor. Recursive elimination could not improve beyond the null model, and all methods converged on the same AIC/BIC, reinforcing the low informativeness of square footage alone.

---

## Test Set Performance

Using the test set (30% of the dataset):

### Confusion Matrix

|              | Predicted: No | Predicted: Yes |
|--------------|----------------|-----------------|
| Actual: No   | 908            | 0               |
| Actual: Yes  | 492            | 0               |

### Accuracy: **64.86%**

- The model **always predicted "No"** for garage presence on the test set. While the overall accuracy appears decent due to the **class imbalance** (64% of houses don't have a garage), the model **fails completely** to identify any homes with a garage.
- This confirms that the model is **biased toward the majority class** and fails to generalize in a balanced way.

---

## Conclusion of Results

- The logistic regression model is statistically valid but practically ineffective.
- The relationship between square footage and garage presence is statistically significant but **too weak** to be useful alone.
- The confusion matrix shows **zero sensitivity** to the positive class (garage = Yes), indicating the model is **not viable in real-world applications** without incorporating additional features.

For this project to yield meaningful predictions, it is essential to include other potentially important predictors like **home type**, **number of floors**, **price**, or **zoning attributes**.

---

## Model Metrics Overview

| Metric                  | Before Optimization | After Optimization |
|-------------------------|---------------------|--------------------|
| Pseudo R²               | —                   | 0.00066            |
| Log-Likelihood          | -3661.4             | -3659.0            |
| AIC                     | —                   | 7322.01            |
| BIC                     | —                   | 7335.27            |
| Test Set Accuracy       | —                   | 64.86%             |

### Interpretation

- **Pseudo R² improved** slightly after modeling, indicating a very marginal increase in explanatory power.
- **Log-Likelihood increased** from -3661.4 to -3659.0, indicating a better model fit.
- **AIC and BIC decreased**, which is favorable and confirms that the selected model is statistically preferred over the null.
- **Accuracy (64.86%)** reflects prediction on the test set. However, due to the model predicting only the majority class, this number is misleading. It underscores the limitation of using accuracy in imbalanced datasets.

---

## Sample of the Dataset

Below is a preview of a few records from the dataset used in this project:

| ID   | Price       | SquareFootage | NumBathrooms | NumBedrooms | BackyardSpace | CrimeRate | SchoolRating | AgeOfHome | DistanceToCityCenter | EmploymentRate | PropertyTaxRate | RenovationQuality | LocalAmenities | TransportAccess | Fireplace | HouseColor | Garage | Floors | Windows | PreviousSalePrice | IsLuxury |
|------|-------------|----------------|---------------|--------------|----------------|------------|----------------|------------|------------------------|----------------|------------------|--------------------|-----------------|------------------|-----------|-------------|--------|--------|----------|--------------------|----------|
| 4922 | 255614.90   | 566.62         | 1.0           | 4            | 779.42         | 20.56      | 5.62           | 39.46      | 10.08                  | 97.29          | 1.84             | 4.93               | 4.44            | 4.55             | Yes       | Blue        | No     | 1      | 13       | 181861.54           | 0        |
| 5009 | 155586.09   | 1472.34        | 1.0           | 2            | 656.13         | 15.62      | 5.63           | 40.51      | 7.89                   | 93.22          | 0.95             | 4.08               | 5.56            | 6.83             | No        | Green       | No     | 1      | 17       | 50042.60            | 0        |
| 4450 | 131050.83   | 550.00         | 1.78          | 3            | 754.57         | 12.47      | 9.20           | 48.38      | 23.74                  | 96.60          | 1.87             | 4.26               | 8.07            | 8.48             | Yes       | Green       | Yes    | 2      | 34       | 48400.34            | 0        |
| 1070 | 151361.71   | 941.81         | 2.04          | 2            | 439.59         | 22.22      | 7.08           | 94.67      | 5.22                   | 91.45          | 1.45             | 4.45               | 5.00            | 6.27             | Yes       | Red         | No     | 1      | 14       | 84594.12            | 0        |
| 400  | 113167.61   | 550.00         | 1.06          | 3            | 353.03         | 8.28       | 5.93           | 16.80      | 43.13                  | 86.50          | 1.26             | 3.36               | 5.46            | 6.99             | No        | White       | Yes    | 1      | 21       | 22934.60            | 0        |
| 5979 | 224973.41   | 1474.99        | 1.86          | 2            | 774.45         | 25.39      | 4.92           | 57.62      | 19.71                  | 94.71          | 1.49             | 6.15               | 0.98            | 1.90             | No        | White       | Yes    | 2      | 36       | 160664.13           | 0        |
| 3703 | 169471.52   | 1069.49        | 1.23          | 3            | 757.58         | 37.84      | 4.40           | 46.86      | 5.73                   | 80.90          | 1.57             | 5.82               | 4.25            | 5.12             | No        | White       | No     | 1      | 10       | 89824.61            | 0        |
| 2260 | 265497.89   | 550.00         | 1.0           | 1            | 636.64         | 48.26      | 4.32           | 60.29      | 11.35                  | 94.18          | 0.72             | 5.81               | 5.43            | 6.03             | No        | Yellow      | Yes    | 1      | 15       | 177283.20           | 0        |

This subset highlights the diversity in housing characteristics and pricing used for training and testing the model.

---

## Conclusion

This project demonstrates the practical use of logistic regression to address a binary classification problem in the housing domain—predicting whether a residential property has a garage based on its square footage. The results confirmed a statistically significant relationship between home size and garage presence, establishing a foundational understanding of how structural features can inform classification tasks in real estate analytics.

While square footage alone offers only a modest predictive signal, the project successfully implemented a full data science pipeline: from exploratory data analysis and preprocessing to statistical modeling and evaluation. This process illustrates the power of interpretable models and the importance of grounding predictions in real-world data.

### Key Achievements

- Developed a clean and reproducible modeling workflow using logistic regression.
- Demonstrated statistically significant predictors of garage presence.
- Implemented forward, backward, and recursive feature selection techniques.
- Highlighted the interpretability and transparency of coefficient-based models.

### Future Enhancements

- **Feature Expansion**: Including more property and neighborhood features can enhance predictive accuracy and reveal deeper insights.
- **Advanced Modeling**: Exploring tree-based and ensemble models (e.g., Random Forest, XGBoost) can further improve performance and capture complex patterns.
- **Balanced Evaluation**: Incorporating metrics like precision, recall, and ROC-AUC will provide a more complete view of model effectiveness.
- **Class Balance Strategies**: Resampling techniques such as SMOTE can help improve sensitivity to underrepresented cases.

### Final Thoughts

This project lays a strong foundation for more sophisticated housing classification systems. By combining interpretable modeling with statistical rigor, it opens the door for further exploration in real estate prediction, with potential applications for housing developers, market analysts, and smart property platforms. With expanded features and enhanced techniques, this approach has strong potential to evolve into a powerful and practical decision-support tool.
