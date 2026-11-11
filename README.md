# Ride-Hailing Fare Prediction and Statistical Analysis

### Transportation Data Analytics Project

<p align="center">
  <b>Exploratory Data Analysis | Statistical Inference | Feature Engineering | Machine Learning | Ensemble Learning</b>
</p>

---

## Overview

This project investigates ride-hailing prices and develops machine learning models to predict transportation fares using historical Uber and Lyft ride data.

The analysis uses a dataset containing approximately 693,000 observations and 39 original attributes, combining temporal, geographical, service-related, and weather information.

The main objective is to understand the factors influencing ride prices and develop predictive models capable of estimating fares based on trip characteristics and environmental conditions.

The project follows a comprehensive data mining workflow, progressing from exploratory data analysis and statistical hypothesis testing to feature selection, predictive modeling, hyperparameter optimization, and ensemble learning.

Particular attention is given to comparing traditional linear regression approaches with nonlinear tree-based models and investigating whether combining multiple predictors can improve forecasting accuracy.

## Project Objectives

The main objectives of this project are:

- Investigate the relationship between ride characteristics and transportation fares.
- Analyze pricing differences between Uber and Lyft service categories.
- Examine the influence of trip distance, time, location, and weather conditions on ride prices.
- Perform statistical hypothesis testing to evaluate differences between service categories.
- Identify relevant predictors through correlation analysis and feature selection.
- Develop and compare multiple regression models.
- Optimize model hyperparameters using GridSearchCV.
- Investigate ensemble learning and residual-based model improvement.
- Evaluate the predictive performance of individual and combined models.

## Dataset Description

The project uses the `rideshare_kaggle.csv` dataset, containing ride-hailing information from Uber and Lyft.

The original dataset consists of:

- 693,071 observations.
- 39 attributes.
- Numerical and categorical variables.
- Temporal, geographical, transportation, and meteorological information.

Each observation represents an individual ride option offered by a transportation service at a particular time and location.

### Feature Categories

**1. Temporal Features**

| Feature | Description |
|---|---|
| `hour` | Hour at which the ride information was recorded. |
| `day` | Day of the month. |
| `month` | Month of the year. |

These features allow the investigation of time-dependent pricing patterns and potential demand variations.

**2. Geographical Features**

| Feature | Description |
|---|---|
| `source` | Starting location of the trip. |
| `destination` | Destination of the trip. |
| `latitude` | Latitude coordinate. |
| `longitude` | Longitude coordinate. |
| `distance` | Trip distance in miles. |

These attributes describe the geographical characteristics of individual trips.

**3. Transportation and Service Features**

| Feature | Description |
|---|---|
| `cab_type` | Transportation provider (Uber or Lyft). |
| `product_id` | Internal service identifier. |
| `name` | Human-readable ride category. |
| `price` | Observed ride fare. |
| `surge_multiplier` | Surge pricing multiplier. |

These variables capture the economic and operational characteristics of ride-hailing services.

**4. Weather and Environmental Features**

The dataset also contains meteorological information, including:

- Temperature and apparent temperature.
- Humidity.
- Wind speed and wind gust.
- Visibility.
- Precipitation intensity and probability.
- Atmospheric pressure.
- Cloud cover.
- UV index.
- Dew point.
- Ozone concentration.
- Moon phase.

These features allow the investigation of potential relationships between weather conditions and transportation pricing.

## Data Preprocessing

Before developing predictive models, several preprocessing operations are performed.

### 1. Data Inspection

The initial analysis examines:
- Dataset dimensions and structure.
- Attribute types.
- Missing observations.
- Descriptive statistics.
- Numerical and categorical feature distributions.

### 2. Missing Value Treatment

Observations belonging to the `Taxi` service category are removed because their fare information is unavailable.

Missing fare values are subsequently imputed using the mean fare of the corresponding ride category.

For ride category $k$, the imputation rule is:

$$
\hat{p}_k =
\frac{1}{|I_k|}
\sum_{i\in I_k}p_i
$$

Where $I_k$ represents observations belonging to category $k$ with available fare information.

### 3. Categorical Encoding

Categorical variables are converted into numerical representations using dummy variables and one-hot encoding.

This transformation is applied to variables such as:
- Ride category.
- Transportation provider.
- Source location.
- Destination location.
- Other categorical attributes used in the predictive models.

### 4. Outlier Detection

Two approaches are investigated for identifying unusually high or low ride prices.

**Interquartile Range (IQR)**

For a given ride category:

$$
IQR=Q_3-Q_1
$$

The lower and upper bounds are defined as:

$$
LB=Q_1-1.5(IQR)
$$

$$
UB=Q_3+1.5(IQR)
$$

Observations outside these bounds are identified as potential outliers.

**Percentile-Based Filtering**

An alternative approach removes observations outside specified percentile boundaries.

The two methods are compared to investigate their effects on the distribution of ride prices.

## Exploratory Data Analysis

The exploratory analysis focuses on understanding the structure of ride prices and their relationships with different explanatory variables.

### 1. Price Distribution

The distribution of ride prices is examined using histograms and boxplots.

These visualizations help identify:
- Price concentration.
- Distribution asymmetry.
- Extreme observations.
- Differences between ride categories.

### 2. Ride Category Analysis

Different service categories are compared to investigate their pricing behavior.

The analysis examines differences between shared, standard, premium, and larger-capacity transportation services.

### 3. Statistical Hypothesis Testing

One-way Analysis of Variance (ANOVA) is employed to investigate whether mean ride prices differ significantly between service categories.

The hypotheses are:

$$
H_0:\mu_1=\mu_2=\cdots=\mu_k
$$

$$
H_1:\exists i,j:\mu_i\neq\mu_j
$$

Where $\mu_i$ represents the mean fare of ride category $i$.

Following the ANOVA analysis, Tukey's Honestly Significant Difference (HSD) test is used for pairwise comparisons between service categories.

This provides a more detailed investigation of which categories exhibit statistically significant differences in their average fares.

### 4. Correlation Analysis

Correlation matrices are generated to investigate relationships between numerical and encoded categorical variables.

The analysis focuses on identifying features associated with ride prices, particularly:
- Trip distance.
- Ride category.
- Transportation provider.
- Temporal characteristics.
- Environmental conditions.

## Feature Engineering and Selection

A significant part of the project investigates the selection of relevant explanatory variables for regression modeling.

The objective is to reduce unnecessary dimensionality while retaining predictive information.

Three feature selection approaches are implemented.

### 1. Correlation-Based Selection

Features are ranked according to their correlation with the target variable.

This provides an initial indication of which variables exhibit stronger linear relationships with ride prices.

### 2. Forward Selection

Forward selection begins with an empty feature set and progressively adds predictors.

At each iteration, the feature that produces the greatest improvement in model performance is selected.

The procedure continues until additional features no longer improve the selected evaluation criterion.

### 3. Backward Elimination

Backward elimination begins with the complete set of candidate predictors and progressively removes features.

At each iteration, the effect of removing individual predictors is evaluated.

The procedure continues until removing additional features no longer improves the model's performance.

The three approaches are compared to investigate their effects on predictive accuracy and model complexity.

## Regression Models

The main predictive task is estimating the price of a ride based on its characteristics.

Let:

- $X$ represent the matrix of explanatory features.
- $y$ represent the observed ride prices.
- $\hat{y}$ represent the predicted ride prices.

The regression problem is expressed as:

$$
\hat{y}=f(X)
$$

Different regression algorithms are investigated to capture both linear and nonlinear relationships between explanatory variables and ride prices.

### 1. Multiple Linear Regression

Multiple Linear Regression is used as a baseline predictive model.

The model assumes a linear relationship between the explanatory variables and the target:

$$
\hat{y}=
\beta_0+\sum_{j=1}^{p}\beta_jx_j
$$

Where:
- $\beta_0$ is the intercept.
- $\beta_j$ represents the coefficient of predictor $j$.
- $x_j$ represents an explanatory variable.

The model is evaluated using different feature selection approaches, allowing comparison between full and reduced feature sets.

### 2. Random Forest Regression

Random Forest is implemented as an ensemble of decision trees.

Each tree generates an individual prediction, and the final prediction is obtained by averaging the outputs:

$$
\hat{y}(X)=
\frac{1}{T}
\sum_{t=1}^{T}f_t(X)
$$

Where:
- $T$ is the number of trees.
- $f_t(X)$ is the prediction of tree $t$.

The model is investigated using different configurations of:
- Number of estimators.
- Maximum tree depth.
- Minimum samples required for splitting.

GridSearchCV is used to identify suitable hyperparameter configurations.

### 3. XGBoost Regression

XGBoost is implemented as a gradient-boosted decision tree model.

Unlike Random Forest, which builds an ensemble of independently trained trees, gradient boosting constructs trees sequentially to improve the predictions of the existing ensemble.

The general prediction structure is:

$$
\hat{y}_i=
\sum_{t=1}^{T}f_t(x_i)
$$

The model is optimized through hyperparameter tuning, considering:

- Number of estimators.
- Learning rate.
- Maximum tree depth.
- Subsample ratio.
- Column sampling ratio.

GridSearchCV with three-fold cross-validation is used to investigate different parameter combinations.

## Ensemble Learning and Model Combination

An additional part of the project investigates whether combining multiple predictive models can improve fare estimation.

Two approaches are explored.

### 1. Residual-Based Random Forest Correction

A hybrid approach is developed using two Random Forest models.

The first model generates the initial fare prediction:

$$
\hat{y}^{(1)}=f_1(X)
$$

The residuals on the training data are then calculated:

$$
r_i=y_i-\hat{y}_i^{(1)}
$$

A second Random Forest model is trained to predict these residuals:

$$
\hat{r}=f_2(X)
$$

The final prediction is obtained by adding the residual correction to the initial prediction:

$$
\boxed{
\hat{y}=
f_1(X)+f_2(X)
}
$$

The purpose of this approach is to capture systematic prediction errors that remain after the first model.

### 2. Stacking Regression

A stacking architecture is implemented using Random Forest and XGBoost as base learners.

A Ridge Regression model serves as the meta-learner.

The architecture consists of two levels:

**Level 0 — Base Models**

- Random Forest Regressor.
- XGBoost Regressor.

**Level 1 — Meta-Model**

- Ridge Regression.

The base models generate predictions that are combined by the meta-model:

$$
\hat{y}=
\beta_0+
\beta_1\hat{y}_{RF}+
\beta_2\hat{y}_{XGB}
$$

The Ridge model estimates the combination coefficients while applying regularization to reduce excessive coefficient magnitudes.

Five-fold cross-validation is used within the stacking procedure to generate training predictions for the meta-learner.

## Model Evaluation

Regression models are evaluated using several complementary metrics.

### 1. Coefficient of Determination (R²)

$$
R^2=
1-
\frac{
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
$$

R² measures the proportion of variation in the target variable explained by the model.

### 2. Mean Squared Error (MSE)

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### 3. Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{MSE}
$$

### 4. Mean Absolute Error (MAE)

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

These metrics provide different perspectives on predictive accuracy and the magnitude of fare estimation errors.

## Model Comparison and Results

The project compares linear regression, Random Forest, XGBoost, and ensemble-based approaches.

The following results are reported in the notebook for one of the final comparison experiments:

| Model | Test R² | Test MSE |
|---|---:|---:|
| Random Forest | 0.9564 | 3.8632 |
| XGBoost | 0.9609 | 3.4640 |
| Stacking Regressor | **0.9623** | **3.3391** |

<text>These values are taken from the notebook's recorded outputs. They correspond to a particular experimental split and should not be interpreted as independently reproduced benchmark results.</text>

The stacking model achieves the best reported performance in this comparison.

The results suggest that combining predictions from different learning algorithms can improve predictive performance compared with using either base learner independently.

The project also investigates the effect of residual-based correction.

| Model | Test R² | Test MSE |
|---|---:|---:|
| Base Random Forest | 0.9296 | 6.2314 |
| Residual-Corrected Random Forest | 0.9566 | 3.8377 |

## Additional Weather Classification Experiment

Besides the primary fare prediction task, the notebook contains a supplementary classification experiment based on weather conditions.

The target variable is constructed as:

$$
Rain=
\begin{cases}
1 & \text{if precipitation intensity}>0\\
0 & \text{otherwise}
\end{cases}
$$

Two classification approaches are investigated:

- Logistic Regression.
- Linear Support Vector Machine (Linear SVM).

Both models are evaluated using precision, recall, F1-score, and accuracy.

This experiment is separate from the primary ride-price prediction task and uses the weather attributes available in the dataset.

## Repository Structure

```text
transportationProject/
│
├── Main.ipynb
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `Main.ipynb` | Main Jupyter Notebook containing data exploration, statistical analysis, preprocessing, feature selection, regression modeling, hyperparameter optimization, and ensemble learning. |
| `README.md` | Project documentation. |

**Dataset requirement:** The notebook expects `rideshare_kaggle.csv` in its working directory. The dataset is not currently included in the repository.

## Installation and Requirements

The project is implemented in Python.

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost jupyter
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/AbbyPolaris/transportationProject.git
cd transportationProject
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost jupyter
```

Download the `rideshare_kaggle.csv` dataset and place it in the project directory.

Launch Jupyter Notebook:

```bash
jupyter notebook Main.ipynb
```

Execute the notebook cells sequentially to reproduce the analysis and modeling workflow.

## Technologies

- **Python** — Main programming language.
- **Pandas** — Data manipulation and preprocessing.
- **NumPy** — Numerical computation.
- **Scikit-learn** — Regression, feature selection, model evaluation, and ensemble learning.
- **XGBoost** — Gradient-boosted decision trees.
- **SciPy** — Statistical hypothesis testing.
- **Statsmodels** — Statistical modeling and post-hoc analysis.
- **Matplotlib** — Data visualization.
- **Seaborn** — Statistical visualization.
- **Jupyter Notebook** — Interactive analysis and experimentation.

## Team

Transportation Project — Fall 2025

Developed by:

- Abbas Ebrahimi
- Mahan Kazemi
- Amirmahdi Bigdeli

---

<p align="center">
  <b>Transportation Data Analytics</b><br>
  Statistical Analysis • Predictive Modeling • Ensemble Learning
</p>
