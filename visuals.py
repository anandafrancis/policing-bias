import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import numpy as np
import streamlit as st
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

def heatmap(data, x_col, y_col):
    
    # Calculate chi-squared p-values
    p_values = []
    for col in data.columns:
        if col != y_col:
            contingency_table = pd.crosstab(data[col], data[y_col])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            p_values.append(p)
    p_values = pd.DataFrame({'p-value': p_values}, index=data.drop(y_col, axis=1).columns)

    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=p_values.values.T, 
        x=p_values.index, 
        y=[y_col], 
        colorscale='Viridis', 
        colorbar=dict(title='p-value'), 
        hovertemplate='p-value: %{z:.2f}<extra></extra>'
    ))

    # Add title and axis labels
    fig.update_layout(title=f'Chi-Squared Test Results Between {y_col.upper()} and X Variables', xaxis_title='X Variables', yaxis_title='Y Variable')


    # Show the plot
    st.plotly_chart(fig)

def scatter2D(df, x, y, _class):

    trace = go.Scatter(
        x=df[x],
        y=df[y],
        mode='text',
        text=df[_class],
       textfont=dict(
        color='red'
    )
    )

    # create the layout for the scatterplot
    layout = go.Layout(
        title=f'Relationship Between {x.upper()} and {y.upper()}',
        xaxis=dict(title=x),
        yaxis=dict(title=y)
    )

    # create the figure with the trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot
    st.plotly_chart(fig)
        
def barChart(df, x_col, y_col, title, x_label, y_label):

    fig = go.Figure([go.Bar(x=df[x_col], y=df[y_col])])
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    st.plotly_chart(fig)

def customBarChart(ycol, xcol, countCol, df):

    # Create a drop-down menu for grouping variable selection
    group_var = st.selectbox('Select a grouping variable', df[ycol].unique())

    # Filter and group your data based on the user's selection
    grouped_data = df.groupby([ycol, xcol]).count().reset_index()
    var_data = grouped_data[grouped_data[ycol] == group_var]
    fig = go.Figure([go.Bar(x=var_data[xcol],
                                 y=var_data[countCol])])
    fig.update_layout(title=f'Frequency of {group_var.upper()} Committed By {xcol.upper()}', 
                      xaxis_title=xcol, yaxis_title=ycol)
    
    # Display the bar graph in the Streamlit app
    st.plotly_chart(fig)

def scatter3D(df, x, y, z, _class):

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x],
        y=df[y],
        z=df[z],
        mode='text',
        marker=dict(
            size=12,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df[_class],
        textfont=dict(
        color='red'
    )
    )])

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    st.plotly_chart(fig)
    
    

def bubbleChart(df, x, y, size, _class):
    df = df.dropna(subset=size)
    fig = px.scatter(df, x=x, y=y,
         size=size, color=_class,
                 hover_name=_class)
    
    fig.update_layout(title=f'Relationships Between {x.upper()} and {y.upper()} with Size Being {size.upper()} Freq')
    st.plotly_chart(fig)


def pieChart(valCol, nameCol, df):

    fig = px.pie(df, values=valCol, names=nameCol, color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(title=f'Breakdown of {valCol.upper()} by {nameCol.upper()} ',
                     font=dict(size=20))
    st.plotly_chart(fig)

def customPieChart(valCol, nameCol, df, valTitle):

    groupData = df.groupby(nameCol).count().reset_index()
    fig = px.pie(df, values=valCol, names=nameCol, color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(title=f'Breakdown of {valTitle} by {nameCol.upper()} ',
                     font=dict(size=20))
    st.plotly_chart(fig)


def dropdrownPieChart(valCol, nameCol, df, options, key1, key2):

    group_var = st.selectbox('Select a grouping variable', options, key=key1)
    
    groupData = df.groupby([nameCol, group_var]).count().reset_index().rename(columns={valCol: 'count'})
    group_subset = st.selectbox('Choose a subset within the grouping variable',
                                groupData[group_var].unique(), key=key2)
    
    subsetData = groupData[groupData[group_var] == group_subset]

    fig = px.pie(subsetData, values='count', names=nameCol, color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(title=f'Breakdown of {nameCol.upper()} from {group_subset.upper()} Within {group_var.upper()} Variable',
                     font=dict(size=20))
    st.plotly_chart(fig)
