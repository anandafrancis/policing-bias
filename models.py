from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn import svm


def decisionTree(df, xCols, yCol):
    df = df.dropna()
    X = df[xCols]
    y = df[yCol]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X_train, y_train)

    # plot the tree
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, filled=True, ax=ax, feature_names=X.columns, class_names=y[yCol[0]].unique())
    st.pyplot(fig)

    importances = clf.feature_importances_
    features = X.columns

    # Create dictionary of feature importances and corresponding feature names
    imp_dict = dict(zip(features, importances))

    # Sort dictionary by importance value
    sorted_imp_dict = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}

    # Create bar chart of feature importances
    fig = px.bar(x=list(sorted_imp_dict.keys()), y=list(sorted_imp_dict.values()))
    fig.update_layout(title='Feature Importances of Decision Tree', xaxis_title='Feature', yaxis_title='Importance')
    st.plotly_chart(fig)

    # get metrics
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the precision
    precision = precision_score(y_test, y_pred, average='weighted')

    # Calculate the recall
    recall = recall_score(y_test, y_pred, average='weighted')

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print the metrics
    st.write(f'Accuracy Score: {round(accuracy, 4)}')
    st.write(f'Precision Score: {round(precision, 4)}')
    st.write(f'Recall Score: {round(recall, 4)}')
    st.write(f'F1 Score: {round(f1, 4)}')


def svm_model(df, xCols, yCol):
    df = df.dropna()
    X = df[xCols]
    y = df[yCol]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a support vector machine classifier with a radial basis function (RBF) kernel
    svm_classifier = svm.SVC(kernel='linear', C=1.0, decision_function_shape='ovr', random_state=42)

    # Fit the SVM classifier to the training data
    svm_classifier.fit(X_train, y_train)

    # Predict the class labels for the test data
    y_pred = svm_classifier.predict(X_test)


    # Get feature importances
    coefficients = svm_classifier.coef_[0]
    features = X.columns

    # Create dictionary of feature importances and corresponding feature names
    imp_dict = dict(zip(features, coefficients))

    # Sort dictionary by importance value
    sorted_imp_dict = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}

    # Create bar chart of feature importances
    fig = px.bar(x=coefficients, y=features, orientation='h')
    fig.update_layout(title='Feature Importances of SVM', xaxis_title='Importance', yaxis_title='Feature')
    st.plotly_chart(fig)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the precision
    precision = precision_score(y_test, y_pred, average='weighted')

    # Calculate the recall
    recall = recall_score(y_test, y_pred, average='weighted')

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print the metrics
    st.write(f'Accuracy Score: {round(accuracy, 4)}')
    st.write(f'Precision Score: {round(precision, 4)}')
    st.write(f'Recall Score: {round(recall, 4)}')
    st.write(f'F1 Score: {round(f1, 4)}')
