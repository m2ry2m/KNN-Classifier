# KNN Classifier Project

This project demonstrates my skills in machine learning by building a K-Nearest Neighbors (KNN) classifier in Python. I created it to showcase my ability to work with data, train models, and evaluate performance.

## What this project does

- Loads the dataset (`KNN_Project_Data.csv`)
- Visualizes the data with a pairplot to explore relationships
- Scales the features to prepare them for modeling
- Trains a KNN model with K=1 and evaluates its performance
- Tests different K values (1 to 40) to find the optimal one
- Plots error rate vs K (see `error_rate.png`)
- Trains a final model with K=30 and shows the results

## Files

- `knn_classifier.py`: The Python code for the KNN classifier
- `KNN_Project_Data.csv`: The dataset used in the project
- `error_rate.png`: Plot of error rate for different K values
- `README.md`: This file

## How to run

You need Python and these libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
Install them with: ```bash pip install pandas numpy scikit-learn matplotlib seaborn ``` Then run: ```bash python knn_classifier.py ```

## What I learned

- How to preprocess data for machine learning models
- How to implement and tune a KNN classifier
- How to evaluate model performance using metrics like confusion matrix and classification report
- How to visualize results to make informed decisions

Thanks for checking out my project!
