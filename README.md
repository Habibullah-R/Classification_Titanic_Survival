# Titanic Survival Prediction

This project focuses on predicting the survival of passengers aboard the Titanic using machine learning. The dataset contains passenger information such as age, gender, ticket class, and more. Various classification models are trained and evaluated to determine the best-performing one.

## Dataset
The dataset used is the famous Titanic dataset, which includes the following features:
- **Survived**: Binary indicator (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Preprocessing
- Dropped irrelevant columns: `PassengerId`, `Cabin`.
- Handled missing values for `Age` and `Embarked`.
- Split the data into training (80%) and testing (20%) sets.
- Applied feature transformations:
  - Numeric features: Imputed missing values with the median and scaled using `StandardScaler`.
  - Categorical features: Encoded using `OneHotEncoder`.

## Models Evaluated
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Decision Tree Classifier**

## Results
The models were evaluated based on accuracy, confusion matrix, and classification report. The best-performing model was **KNN** after hyperparameter tuning, achieving an accuracy of **83.8%**.

### Performance Summary
| Model                  | Accuracy | Precision (0/1) | Recall (0/1) | F1-Score (0/1) |
|------------------------|----------|------------------|--------------|----------------|
| Logistic Regression    | 81.6%    | 0.83 / 0.80      | 0.87 / 0.74  | 0.85 / 0.77    |
| Random Forest          | 81.0%    | 0.81 / 0.81      | 0.89 / 0.70  | 0.85 / 0.75    |
| KNN (Default)          | 82.1%    | 0.83 / 0.81      | 0.88 / 0.74  | 0.85 / 0.77    |
| SVM                    | 81.6%    | 0.82 / 0.81      | 0.88 / 0.73  | 0.85 / 0.77    |
| Decision Tree          | 81.6%    | 0.81 / 0.83      | 0.90 / 0.70  | 0.85 / 0.76    |
| **KNN (Tuned)**        | **83.8%**| **0.83 / 0.85**  | **0.90 / 0.74**| **0.87 / 0.79**|

## Hyperparameter Tuning
The KNN model was further optimized with the following parameters:
- `metric = 'manhattan'`
- `n_neighbors = 11`
- `p = 1`
- `weights = 'distance'`

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git

## Installation

2. Install the required dependencies:  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Run the Jupyter Notebook

   ```bash
   jupyter notebook main.ipynb
## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

## Acknowledgments

- Dataset sourced from [Kaggle](https://www.kaggle.com/competitions/titanic/data)
- Inspired by various machine learning tutorials and competitions
