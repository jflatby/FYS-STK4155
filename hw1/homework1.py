import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sklearn.metrics as metrics

plt.style.use('ggplot')

## Task 2

x = np.sort(np.random.rand(100))
y = 5 * x**2 + 0.1 * np.random.randn(100)


design_matrix = np.zeros((len(x), 3))
design_matrix[:, 0] = 1
design_matrix[:, 1] = x
design_matrix[:, 2] = x**2

#parameters = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y)
#ytilde = design_matrix @ parameters

fit = np.linalg.lstsq(design_matrix, y, rcond = None)[0]
ytilde = np.dot(fit, design_matrix.T)


x_new = np.linspace(0, 1, 100)
plt.scatter(x, y)
plt.plot(x, ytilde)
plt.show()


# 2.2
skfit = skl.LinearRegression().fit(design_matrix, y)
ytilde = skfit.predict(design_matrix)

plt.scatter(x, y)
plt.plot(x, ytilde)
plt.show()

# 2.3
mse = metrics.mean_squared_error(y, ytilde)
r2_score = metrics.r2_score(y, ytilde)

print("Mean squared error: ", mse)
print("R2 score: ", r2_score)


