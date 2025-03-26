import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


np.random.seed(42)
study_hours = np.random.randint(1, 10, 100)
previous_scores = np.random.randint(40, 100, 100)
exam_scores = 0.5 * study_hours + 0.5 * previous_scores + np.random.normal(0, 5, 100)

data = {
    'study_hours': study_hours,
    'previous_scores': previous_scores,
    'exam_scores': exam_scores
}

df = pd.DataFrame(data)


X = df[['study_hours', 'previous_scores']]
y = df['exam_scores']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.scatter(y_test, y_pred)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek ve Tahmin Edilen Değerler Arasındaki İlişki")
plt.show()


results = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
print(results)
