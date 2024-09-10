# Direct Parameter Estimation-based Imputation for randomly missing data
This repository contains the **DPEImputation** package, designed for the imputation of missing data. The package is based on the methodology outlined in the paper **"Direct Parameter Estimation-based Imputation for Randomly Missing Data"** by Tuan L. Vo, Van Hua, Uyen Dang, and Thu Nguyen.

### Installation of package
To install the DPEImputation package, you can easily do so via GitHub. Run the following command in your environment:

`!pip install git+https://github.com/DangLeUyen/DPEImputation.git`

After installation, you can import the **DPEImputer** class as follows:

`from DPEImputation import DPEImputer`

### Usage Guide

#### 1. For a Dataset X with Labels y

```from DPEImputation import DPEImputer

# Create an instance of the DPEImputer class
imputer = DPEImputer()

# Fit the imputer on the incomplete dataset X with a specified window_size of 10.
# If window_size is not provided, it defaults to 7.
# If the number of features in the dataset is less than 7, and window_size is not provided, window_size will be automatically set to the number of features.
imputer.fit(X, y, window_size=10)

# Apply imputation to the missing data in X
X_imputed = imputer.transform(X, y)
```

#### 2. For a dataset X without label

```from DPEImputation import DPEImputer

# Create an instance of the DPEImputer class
imputer = DPEImputer()

# Fit and transform the imputer on the incomplete dataset X with window_size set to 10.
# If window_size is not provided, it defaults to 7.
# If the number of features in the dataset is less than 7 and window_size is not provided, window_size will be automatically set to the number of features.
X_imputed = imputer.fit_transform(X, window_size=10)
```

For additional examples and details, please refer to the `example.ipynb` file in the repository.
### Citation
If you use this package in your research, please cite the paper:

Tuan L. Vo, Van Hua, Uyen Dang, Thu Nguyen. Direct Parameter Estimation-based Imputation for Randomly Missing Data.

-------------------------------
This README provides users with a clear guide on installing and using the DPEImputation package while also giving proper credit to the research behind it.

