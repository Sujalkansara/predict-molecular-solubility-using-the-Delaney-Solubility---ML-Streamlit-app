# predict-molecular-solubility-using-the-Delaney-Solubility-ML
built a regression model to predict molecular solubility using the Delaney Solubility Dataset, diving deep into the world of cheminformatics and data science.

# Molecular Solubility Prediction with Machine Learning

## Overview
This project builds a machine learning model to predict aqueous solubility (logS) of chemical compounds using the Delaney Solubility Dataset. It compares two regression models—Linear Regression and Random Forest—to explore their performance in a cheminformatics application. The project covers the full ML pipeline: data loading, preprocessing, model training, prediction, evaluation, and comparison.
this is a beginner-friendly dive into ML with real-world relevance in drug discovery.

## Dataset

- **Source**: Delaney Solubility Dataset
- **Description**: Contains ~1,128 compounds with 4 numerical features:
   1. MolLogP: Hydrophobicity (partition coefficient).
   2. MolWt: Molecular weight.
   3. NumRotatableBonds: Molecular flexibility.
   4. AromaticProportion: Proportion of aromatic atoms.
   5. Target: logS (logarithm of aqueous solubility, mol/L).
- **Format**: CSV, clean with no missing values.

## Project Workflow

### Importing the Dataset  
Loaded using Pandas: pd.read_csv(url).

### Data Separation  
Features (X): MolLogP, MolWt, NumRotatableBonds, AromaticProportion.  
Target (y): logS.
- **`y = f(x)`**

### Data Splitting  
Split into 80% training and 20% testing sets using `sklearn.model_selection.train_test_split.`

### Linear Regression Model  
Built with sklearn.linear_model.LinearRegression.  
Trained `(model.fit(X_train, y_train))`, predicted `(model.predict(X_test))`, and evaluated using Mean Squared Error (MSE) and R².

### Random Forest Model  
Built with `sklearn.ensemble.RandomForestRegressor`.  
Followed same steps: train, predict, evaluate.

### Model Comparison  
Compared MSE and R² scores. Random Forest typically outperformed Linear Regression due to its ability to capture non-linear relationships.


## Tech Stack

- Python: Core language.
- Pandas & NumPy: Data manipulation.
- scikit-learn: Model building and evaluation.
- Matplotlib/Seaborn: Data and result visualization.
- Jupyter Notebook: Interactive development.

## Installation

- Clone the repository:
``` git clone https://github.com/Sujalkansara/predict-molecular-solubility-using-the-Delaney-Solubility-ML-.git```

- Install dependencies:
```pip install pandas numpy scikit-learn matplotlib seaborn```

- Run the Jupyter Notebook:
```jupyter notebook solubility_prediction.ipynb```



## Usage

Open solubility_prediction.ipynb in Jupyter Notebook.
Run the cells sequentially to:
Load and preprocess the dataset.
Train and evaluate Linear Regression and Random Forest models.
Visualize results and compare model performance.

Modify hyperparameters or try other algorithms (e.g., XGBoost) for experimentation.

## Results

- Linear Regression: Simple, interpretable, but assumes linear relationships.
- Random Forest: Captures non-linear patterns, often yielding lower MSE and higher R².
- Visualizations (e.g., scatter plots of predicted vs. actual logS) highlight model performance.

## Learnings

Mastered the end-to-end ML pipeline.
Understood trade-offs: Linear Regression’s simplicity vs. Random Forest’s flexibility.
Learned to evaluate models with MSE and R².
Gained insights into cheminformatics for drug discovery.

## Future Improvements

Explore hyperparameter tuning (e.g., GridSearchCV for Random Forest).
Test advanced models like XGBoost or Neural Networks.
Incorporate additional molecular descriptors.

Contributing
Feel free to fork this repository, submit issues, or create pull requests with improvements!

