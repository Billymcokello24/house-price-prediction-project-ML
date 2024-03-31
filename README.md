

 House Price Prediction Project using ML
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The project involves the following key steps:
1. Data preprocessing, including handling missing values, converting categorical variables to numerical values, and scaling features.
2. Implementing linear regression, Lasso regression, and Ridge regression models.
3. Extending the feature set by including polynomial features of degree 3.
4. Evaluating model performance using mean squared error (MSE) and R-squared on both training and testing datasets.
5. Creating a plot to visualize the relationship between "Overall Qual" and "SalePrice".

## Dataset

We utilize the Ames Housing dataset, which includes various features related to houses such as overall quality, living area, basement size, etc. The dataset is publicly available and can be accessed [here](http://jse.amstat.org/v19n3/decock/AmesHousing.txt).

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Setup

1. Clone the repository to your local machine.
2. Ensure you have Python and the necessary libraries installed (Pandas, NumPy, Scikit-Learn, and Matplotlib). If not, install using pip:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```

## Usage

1. Run the provided notebook script `ames-housing.ipynb`.
2. The script will load the dataset, preprocess the data, train various regression models, and display MSE and R-squared values.
3. Additionally, it will create plots to visualize the relationship between "Overall Qual" and "SalePrice".

## Results

The project evaluates the performance of linear regression, Lasso regression, and Ridge regression models with and without polynomial features of degree 3. The MSE and R-squared values are displayed for both training and testing datasets, providing insights into the effectiveness of the models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

