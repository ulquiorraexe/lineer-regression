# Linear Regression Model - Exam Score Prediction

## Description
This project applies a **Linear Regression** model to predict students' **exam scores** based on two features: **study hours** and **previous exam scores**. The dataset is synthetically generated, and the model is trained and evaluated using the **scikit-learn** library.

## Features
- **Data Generation**: Randomly generates data for study hours, previous scores, and exam scores.
- **Training and Testing**: The dataset is split into training and testing sets.
- **Model Training**: A linear regression model is trained on the training data.
- **Evaluation**: The model is evaluated using **Mean Squared Error (MSE)** and **R-squared (R²)** scores.
- **Visualization**: A scatter plot visualizing the relationship between actual and predicted exam scores.
- **Comparison**: Displays a table comparing actual vs predicted exam scores.

## Requirements
Make sure to have the following Python libraries installed to run the project:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   gh repo clone ulquiorraexe/lineer-regression
2. Navigate to the project directory:
   ```bash
   cd lineer-regression
3. Install the required Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
4. Run the script:
   ```bash
   python linear_regression.py
This will execute the linear regression model on the dataset, train the model, and output the results.

## Usage
Once the dependencies are installed, you can run the Python script to train and evaluate the linear regression model. The script will:
1. Generate synthetic data.
2. Split the data into training and testing sets.
3. Train the linear regression model.
4. Evaluate the model using MSE and R².
5. Display the actual vs predicted values in a scatter plot.

## Notes
- This is a basic example of using **linear regression** for predicting exam scores.
- The dataset used in this project is synthetic, and some Gaussian noise is added to make it more realistic.
- You can modify the dataset or use real data to experiment further with the model.

## License
This project does not have a license.
