# Introduction

Linear regression is one of the simplest and most widely used techniques in statistical modeling and machine learning. It models the relationship between a dependent variable \( y \) and one or more independent variables \( x \) using a linear equation.

---

## Simple Linear Regression

In simple linear regression, the relationship between one independent variable and the dependent variable is modeled as:

\[
y = f(x) = a x + b
\]

Where:
- \( x \) is the feature
- \( y \) is the target
- \( a \) is the slope of the regression line
- \( b \) is the intercept (bias)

The goal is to find the best-fitting line that minimizes the error between the predicted values and the actual values.

---

## Multiple Linear Regression

Multiple linear regression extends the concept of simple linear regression to multiple features. The equation is given by:

\[
y = f(x) = a_0 + a_1 x_1 + a_2 x_2 + \cdots + a_d x_d
\]

Where:
- \( a_0 \) is the bias (intercept)
- \( x_1, x_2, ..., x_d \) are the \( d \) input features
- \( a_1, a_2, ..., a_d \) are the corresponding coefficients that determine the contribution of each feature to the output

---

## Model Training

The objective of training a linear regression model is to find the optimal values of the parameters \( a_0, a_1, ..., a_d \) that minimize the error. This is commonly achieved using the **Ordinary Least Squares (OLS)** method, which minimizes the sum of squared residuals:

\[
J(a) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Where:
- \( y_i \) is the actual value
- \( \hat{y}_i \) is the predicted value

---

## Assumptions of Linear Regression

1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Independence**: The observations are independent of each other.
3. **Homoscedasticity**: The variance of residuals is constant across all levels of the independent variables.
4. **Normality**: The residuals follow a normal distribution.
5. **No Multicollinearity**: Independent variables should not be highly correlated with each other.

---

## Evaluating the Model

To assess the performance of a linear regression model, we use the following metrics:

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

### R-squared (\( R^2 \))

\( R^2 \) measures the proportion of variance in \( y \) explained by the model:

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Where:
- \( SS_{res} \) is the sum of squared residuals
- \( SS_{tot} \) is the total sum of squares

---

## Training a Linear Regressor

### Normal Equations (Least Square Method)

### Pseudo-Inverse

### Batch Gradient Descent Method

### Mini-Batch Gradient Descent Method

### Stochastic Gradient Descent Method
