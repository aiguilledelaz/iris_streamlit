import streamlit as st
import pandas as pd
import requests
import time

st.title('アヤメ分類器')

st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.sidebar.slider('sepal width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.sidebar.slider('petal width (cm)', min_value=0.0, max_value=10.0, step=0.1)

iris = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

targets = ['setosa', 'versicolor', 'virginica']

if st.sidebar.button('Predict'):
    st.write('## Input Value')
    iris_df = pd.DataFrame(iris, index=['data'])
    st.write(iris_df)

    # 予測の実行
    response = requests.post('https://test-iris-1z1u.onrender.com/predict', json=iris)
    time.sleep(10)
    prediction = response.json()["prediction"]
  

    # 予測結果の表示
    st.write('## Prediction')
    st.write(prediction)

    # 予測結果の出力
    st.write('## Result')
    st.write('このアイリスはきっと',str(targets[int(prediction)]),'です!')
