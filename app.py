import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

st.title("Analiza PCA danych spektroskopowych")

uploaded_file = st.file_uploader("Wgraj plik CSV z danymi (wiersze = próbki, kolumny = cechy)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Podgląd danych:")
    st.dataframe(df)

    try:
        n_components = st.slider("Liczba składowych PCA", 2, min(len(df.columns), 10), 2)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(df.values)

        explained_var = pca.explained_variance_ratio_

        st.subheader("Wykres PCA (pierwsze 2 składowe)")
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            labels={"x": "PC1", "y": "PC2"},
            title=f"PC1 ({explained_var[0]*100:.1f}%) vs PC2 ({explained_var[1]*100:.1f}%)"
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Błąd analizy PCA: {e}")
