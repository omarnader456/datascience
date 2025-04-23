
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data_and_create_figure():
    
    
    df = pd.read_csv('amazoncleaned2.csv')
    df2 = pd.read_csv('laptopscleaned2.csv')

    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=('Price vs Rating', 
                                        'Price vs Brand', 
                                        'Price vs RAM', 
                                        'Screen Size vs Weight'),
                        vertical_spacing=0.15,  
                        horizontal_spacing=0.15)

    fig.add_trace(
        go.Scatter(x=df['Price'], y=df['Rating'], mode='markers'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=df['Brand'], y=df['Price']),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=df2['RAM'], y=df2['Price (USD)'], mode='markers'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df2['Screen Size'], y=df2['Weight'], mode='markers'),
        row=2, col=2
    )

    corr_matrix1 = df.select_dtypes(include=[np.number]).corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix1.values, 
            x=corr_matrix1.columns, 
            y=corr_matrix1.columns, 
            colorscale='RdBu',
            text=corr_matrix1.values.round(2), 
            texttemplate="%{text}", 
            textfont={"size": 10},
            colorbar=dict(len=0.45, y=0.21, yanchor='middle')
        ),
        row=3, col=1
    )

    corr_matrix2 = df2.select_dtypes(include=[np.number]).corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix2.values, 
            x=corr_matrix2.columns, 
            y=corr_matrix2.columns, 
            colorscale='RdBu',
            text=corr_matrix2.values.round(2), 
            texttemplate="%{text}", 
            textfont={"size": 10},
            colorbar=dict(len=0.45, y=0.21, yanchor='middle')
        ),
        row=3, col=2
    )

    fig.update_layout(
        height=1200, 
        width=1200, 
        title_text="Multivariate Analysis",
        title_x=0.5, 
        showlegend=False
    )

    fig.update_xaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Rating", row=1, col=1)
    fig.update_xaxes(title_text="Brand", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=2)
    fig.update_xaxes(title_text="RAM", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1)
    fig.update_xaxes(title_text="Screen Size", row=2, col=2)
    fig.update_yaxes(title_text="Weight", row=2, col=2)

    return fig

fig = load_data_and_create_figure()

st.subheader('Exploratory Data Analysis')
st.plotly_chart(fig, use_container_width=True)


m = joblib.load('amazonproject_model.pkl')
m2 = joblib.load('laptops_model.pkl')

st.title('Predict RAM based on Price and Hard Drive Size')
col, col2 = st.columns(2)

with col:
    hds = st.number_input('Hard Disk Size (GB)', min_value=1, value=250)

with col2:
    prc = st.number_input('Price (USD)', min_value=208)
    lglprc = np.log1p(prc)

if st.button('Predict RAM'):
    input_data = pd.DataFrame([[hds]], columns=['Hard Disk Size'])
    input_data2 = pd.DataFrame([[lglprc]], columns=['LogPrice (USD)'])

    prediction = m.predict(input_data)[0]
    prediction2 = m2.predict(input_data2)[0]

    st.subheader('Prediction Result')

    if np.isclose(prediction, prediction2):
        st.write(f"Prediction for RAM: {prediction:.2f} GB")
    else:
        lower = min(prediction, prediction2)
        upper = max(prediction, prediction2)
        st.write(f"Prediction for RAM: {lower:.2f} – {upper:.2f} GB")
