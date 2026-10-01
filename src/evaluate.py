from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

yhat = gb.predict(X_test)
print("Gradient Boosting Classifier", accuracy_score(y_test, yhat))

print(classification_report(y_test, yhat))
