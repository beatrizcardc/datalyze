# Versão Final v4 - Ajustada
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, f_oneway
from sklearn.preprocessing import OneHotEncoder

# Configuração da página
st.set_page_config(page_title="Datalyze - Análise Inteligente de Negócios", layout="wide")

# Título do App
st.title("📊 Datalyze - Análise Inteligente de Negócios")
st.write("Bem-vindo! Aqui você pode carregar seus dados e aplicar técnicas de análise para obter insights valiosos.")

# Função para carregar dados com seleção inteligente de planilha
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheets_validas = xls.sheet_names
            
            sheet_selecionada = st.sidebar.selectbox(
                "Selecione a planilha:", sheets_validas)
            
            df = pd.read_excel(xls, sheet_name=sheet_selecionada)
            
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
                data_min, data_max = df['data'].min(), df['data'].max()
                data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max])
                df = df[(df['data'] >= pd.Timestamp(data_inicio)) & (df['data'] <= pd.Timestamp(data_fim))]
            
            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None
    return None

# Função de previsão de vendas com múltiplas variáveis
def previsao_vendas(df):
    st.sidebar.subheader("⚙️ Variáveis de Influência")
    variaveis_disponiveis = ['dia_semana', 'produto', 'horario', 'temperatura']
    variavel_selecionada = st.sidebar.selectbox("Selecione a variável para análise:", variaveis_disponiveis)
    
    if variavel_selecionada not in df.columns:
        st.warning(f"⚠️ A variável `{variavel_selecionada}` não foi encontrada no DataFrame.")
        return
    
    df['temperatura'] = pd.to_numeric(df['temperatura'], errors='coerce')
    
    model = LinearRegression()
    X = df[[variavel_selecionada]]
    y = df['vendas']
    model.fit(X, y)
    df['previsao'] = model.predict(X)
    
    st.write("### 📈 Previsão de Vendas")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['data'], df['vendas'], label='Vendas Reais', marker='o')
    ax.plot(df['data'], df['previsao'], linestyle='--', color='red', label='Previsão', marker='s')
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    impacto = abs(model.coef_[0])
    if impacto > 1:
        impacto_texto = "Muito impacto"
    elif impacto >= 0.3:
        impacto_texto = "Impacto moderado"
    else:
        impacto_texto = "Impacto fraco ou irrelevante"
    
    st.write(f"### 🔍 Influência da variável `{variavel_selecionada}` sobre as vendas: {impacto_texto} (Coeficiente: {impacto:.2f})")
    
    st.write("### 📊 Estatísticas")
    st.write(f"Média de vendas: {df['vendas'].mean():.2f}")
    st.write(f"Variância das vendas: {df['vendas'].var():.2f}")

# Interface principal
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox(
    "Escolha uma análise", ["Previsão de Vendas", "Clusterização de Clientes", "Testes"])

df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head())
    
    if analise_selecionada == "Previsão de Vendas":
        if {'data', 'vendas'}.issubset(df.columns):
            previsao_vendas(df)
        else:
            st.warning("⚠️ Dados incompletos! Necessário colunas 'data' e 'vendas'.")






