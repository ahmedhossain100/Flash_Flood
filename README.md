# Rainfall Prediction Model

## Overview

This project implements a machine learning model to predict high rainfall events based on historical data.  The model uses a LightGBM classifier, optimized for recall, to identify periods of significant rainfall.  The project includes data preprocessing, feature engineering, model training, and evaluation, with a focus on practical application through threshold optimization.

## Files

*   `model.py`:  The main Python script containing the data loading, preprocessing, model training, and evaluation logic.
*   `output_statistics.xlsx`: The dataset used for training and evaluating the model (ensure this file is available).  It should contain historical weather data.
*   `best_model.pkl`:  A serialized (pickled) file containing the best-trained model from the grid search. This file is generated after running `model.py`.
*   `README.md`: This file, providing an overview of the project and instructions for usage.

## Data Description

The model is trained on `output_statistics.xlsx`, which should contain historical weather data. Key columns used include:

*   `Date`: Date of the observation.  This is converted to datetime and the month is extracted.
*   Other columns: Features relevant to rainfall prediction (all numerical columns except 'Date', 'Rainfall', and 'Extreme Rainfall' are used).
*   `Rainfall`: Rainfall amount (mm). This is used to create the binary target variable, where Rainfall > 204.5 mm is considered a high rainfall event.
*   `Extreme Rainfall`: This column is dropped from the features.

## Dependencies

The following Python libraries are required to run the script:

*   `numpy`
*   `pandas`
*   `scikit-learn` (`sklearn`)
*   `lightgbm`
*   `pickle`

You can install these dependencies using pip:


## Usage

1.  **Install Dependencies:**  Make sure you have all the required libraries installed.

2.  **Data Availability:** Ensure the `output_statistics.xlsx` file is in the same directory as the `model.py` script, or provide the correct path in the script.

3.  **Run the Script:** Execute the `model.py` script:

    ```
    python model.py
    ```

    This will:

    *   Load the data from `output_statistics.xlsx`.
    *   Preprocess the data, including date conversion, month extraction, and one-hot encoding of months.
    *   Split the data into training and testing sets (70/30 split).
    *   Scale the numerical features using `MinMaxScaler`.
    *   Train a LightGBM model using GridSearchCV to optimize for recall.  It uses `class_weight='balanced'` to address potential class imbalance.
    *   Save the best trained model to `best_model.pkl`.
    *   Predict probabilities and optimize the prediction threshold based on precision and recall.
    *   Evaluate the model on the test set, calculating accuracy, recall, precision, and ROC AUC, using the optimized threshold.
    *   Print the best parameters found by GridSearchCV and the final evaluation metrics.

4.  **Trained Model:** After running the script, the `best_model.pkl` file will contain the trained model.  This can be loaded and used for making predictions on new data.  Remember to apply the same preprocessing steps to any new data before making predictions.

## Model Details

The model is a LightGBM classifier optimized using GridSearchCV.  Key steps in the `model.py` script include:

*   **Feature Engineering:**  The 'Date' column is converted to datetime objects, and the month is extracted and used for one-hot encoding.
*   **Data Preprocessing:**  Numerical features are scaled using `MinMaxScaler`.
*   **Model Training:**  A `LGBMClassifier` is trained with `class_weight='balanced'` to handle potential class imbalance. GridSearchCV is used to find the best hyperparameters, focusing on maximizing recall.  The parameter grid explored includes:
    *   `num_leaves`: [31, 50]
    *   `max_depth`: [10, 20, 30]
    *   `learning_rate`: [0.01, 0.1, 0.2]
    *   `n_estimators`: [100, 200]
*   **Threshold Optimization:** The code calculates precision-recall pairs for different probability thresholds and selects the threshold that provides the best balance between precision and recall, given a minimum acceptable precision of 0.20.  This threshold is then used to make final predictions.

## Model Evaluation

The model's performance is evaluated using the following metrics:

*   **Accuracy:**  Overall correctness of the model, calculated using the optimized threshold.
*   **Recall:**  The ability of the model to identify all actual positive cases (high rainfall events), calculated using the optimized threshold.  This is the primary metric being optimized.
*   **Precision:**  The ability of the model to avoid classifying negative cases as positive, calculated using the optimized threshold.
*   **ROC AUC:**  Area under the Receiver Operating Characteristic curve, a measure of the model's ability to distinguish between positive and negative classes.

The script prints these metrics after training and evaluation.

## Future Improvements

*   **Feature Selection:** Experiment with different feature sets to improve model performance.  Consider using feature importance techniques to identify the most relevant features.
*   **More Data:**  Increase the size of the training dataset.
*   **Hyperparameter Tuning:**  Further refine the hyperparameters of the LightGBM model.  Consider using a wider range of parameter values and different search strategies (e.g., RandomizedSearchCV).
*   **External Data Sources:** Incorporate external data sources such as weather forecasts, satellite data, or climate indices.
*   **Automated Retraining:** Implement a system for automatically retraining the model with new data on a regular basis.
*   **Ensemble Methods:** Explore ensemble methods beyond LightGBM, such as Random Forests or Gradient Boosting Machines.
*   **More Sophisticated Threshold Optimization:** Explore more sophisticated methods for threshold optimization that take into account the specific costs and benefits of different types of errors.

## License

MIT License

Copyright (c) Ahmed Hossain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
