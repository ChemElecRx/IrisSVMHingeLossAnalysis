from sklearn.svm import SVC

# Sample data
X = [[162.5, 57.1], [163.8, 47.8], [167.1, 48.6], [164.6, 54.4], [159.9, 50.9], 
     [171.4, 53.3], [157.7, 44.5], [160.2, 56.2], [158.3, 45.7], [165.9, 46.8], 
     [172.3, 52.0], [156.8, 59.5], [168.7, 49.3], [173.2, 55.7], [170.6, 58.0], 
     [181.5, 66.8], [168.8, 64.3], [174.2, 57.9], [179.3, 69.2], [185.7, 56.7], 
     [170.4, 59.1], [167.9, 68.4], [175.6, 72.0], [188.2, 71.1], [186.9, 60.5], 
     [169.1, 73.6], [172.3, 58.2], [183.0, 67.5], [177.8, 63.4], [180.6, 70.8]]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



# Train SVM model
model = SVC(kernel='linear')
model.fit(X, y)

# New data point to predict the class label for
new_data = [[163, 47]]

# Predict the class label for the new data point
predicted_label = model.predict(new_data)

# Display the predicted label
print('Predicted label:', predicted_label)