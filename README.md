# Direct Parameter Estimation-based Imputation for randomly missing data
This is an imputation package for missing data, which can be easily installed with pip. 

The DPEImputation package is associated with the paper "Direct Parameter Estimation-based Imputation for randomly missing data (Tuan L. Vo, Van Hua, Uyen Dang, Thu Nguyen)"

### Installation of package
You can install the DPEImputation package from Github

`!pip install git+https://github.com/DangLeUyen/DPEImputation.git`

### Implementation of simulation study

`from DPEImputation import DPEImputer
# Create an instance of the DPEImputation class
imputer = DPEImputer()

# Fit the imputer on the incomplete dataset X
imputer.fit(X, initializing=False)

# Apply imputation to the missing data that we want to impute 
X_imputed = imputer.transform(X)  
`
