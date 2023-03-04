from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, chi2, SelectKBest
import pandas as pd
import numpy as np
import streamlit as st

def featureSelector(df, xCols, yCol, key):

    df = df.dropna()
    X = df[xCols]
    y = df[yCol]

    methods = ['SelectPercentile', 'Mutual Information', 'SelectKBest']
    model = st.selectbox('Choose Which Feature Selection Method You Want to Use', methods, key=key)

    if model == 'SelectKBest':
        selector = SelectKBest(chi2, k=5)
        selector.fit_transform(X, y)
        scores = selector.scores_
        

    elif model == 'SelectPercentile':
        selector = SelectPercentile(f_classif, percentile=50)
        selector.fit(X, y)
        scores = selector.scores_

    else:
        scores = mutual_info_classif(X, y)
        


    st.write(f'Feature Selection Results using {model}')
    df = pd.DataFrame({'features': xCols, 'scores': scores})
    df = df.sort_values(by='scores', ascending=False).reset_index(drop=True)
    st.write(df)
