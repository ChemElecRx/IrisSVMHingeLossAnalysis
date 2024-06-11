import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss, accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Remove the class corresponding to Iris Setosa (class label 0)
removed_class = 0
X = X[y != removed_class]
y = y[y != removed_class]

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a LinearSVC classifier (uses hinge loss)
svm = LinearSVC(C=0.001)  # Lower C to increase regularization strength
svm.fit(X_train, y_train)

# Predict using the trained model
y_train_pred = svm.decision_function(X_train)
y_test_pred = svm.decision_function(X_test)

# Calculate hinge loss
train_hinge_loss = hinge_loss(y_train, y_train_pred)
test_hinge_loss = hinge_loss(y_test, y_test_pred)

# Cross-validation to get a more robust estimate
cross_val_scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
cross_val_mean = cross_val_scores.mean()

# Calculate training accuracy
y_train_pred_labels = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred_labels)

# Calculate test accuracy
y_test_pred_labels = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred_labels)

# Print out all the metrics
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Cross-Validation Mean Accuracy: {cross_val_mean:.4f}")
print(f"Train Hinge Loss: {train_hinge_loss:.4f}")
print(f"Test Hinge Loss: {test_hinge_loss:.4f}")

# Visualize the results
# We'll use only the first two features for 2D visualization
X_train_vis = X_train[:, :2]
X_test_vis = X_test[:, :2]

# Re-train the SVM on the reduced dataset for visualization
svm_vis = LinearSVC()
svm_vis.fit(X_train_vis, y_train)

# Plot the decision boundary
def plot_decision_boundary(clf, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
    ax.scatter([], [], c='yellow', label='Iris Versicolor (Class 1)', alpha=0.3)
    ax.scatter([], [], c='blue', label='Iris Virginica (Class 2)', alpha=0.3)
    ax.legend(loc="upper left")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the decision boundary in the first subplot
plot_decision_boundary(svm_vis, X_train_vis, y_train, ax1)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('SVM Decision Boundary with Iris Dataset (2 classes, 2 features)')

# Display class labels and hinge loss values in the second subplot
ax2.axis('off')  # Turn off the axis
class_labels_text = f'Class labels: {np.unique(y)}'
hinge_loss_text = (f'{class_labels_text}\nTrain Hinge Loss: {train_hinge_loss:.4f}\n'
                   f'Test Hinge Loss: {test_hinge_loss:.4f}\n'
                   f'Cross-Validation Mean Accuracy: {cross_val_mean:.4f}\n'
                   f'Train Accuracy: {train_accuracy:.4f}\n'
                   f'Test Accuracy: {test_accuracy:.4f}')
ax2.text(0.5, 0.5, hinge_loss_text, horizontalalignment='center', verticalalignment='center', fontsize=12)

plt.tight_layout()
plt.show()
