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

# Função para carregar dados
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                xls = pd.ExcelFile(uploaded_file)
                planilhas = xls.sheet_names
                sheet_selecionada = st.sidebar.selectbox("Selecione a planilha:", planilhas)
                df = pd.read_excel(xls, sheet_name=sheet_selecionada)
            
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
                data_min, data_max = df['data'].min(), df['data'].max()
                st.sidebar.subheader("📆 Filtro de Período")
                data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max])
                df = df[(df['data'] >= pd.Timestamp(data_inicio)) & (df['data'] <= pd.Timestamp(data_fim))]
            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None
    return None

# Função de clusterização
def clusterizar_clientes(df):
    try:
        colunas_necessarias = {'idade', 'frequencia_compra', 'gasto_medio'}
        if not colunas_necessarias.issubset(df.columns):
            st.warning("⚠️ Dados incompletos! A planilha deve conter: 'idade', 'frequencia_compra' e 'gasto_medio'.")
            return None
        
        with st.spinner('Analisando padrões de compra...'):
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['cluster'] = kmeans.fit_predict(df[['idade', 'frequencia_compra', 'gasto_medio']])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        marcadores = ['o', 's', 'D']
        
        for cluster in sorted(df['cluster'].unique()):
            dados_cluster = df[df['cluster'] == cluster]
            ax.scatter(
                dados_cluster['idade'], 
                dados_cluster['gasto_medio'],
                s=100,
                c=cores[cluster],
                marker=marcadores[cluster],
                label=f'Grupo {cluster + 1}',
                alpha=0.7
            )
        
        ax.set_title('Segmentação de Clientes por Comportamento', pad=20, fontsize=16)
        ax.set_xlabel('Idade', labelpad=10, fontsize=12)
        ax.set_ylabel('Gasto Médio (R$)', labelpad=10, fontsize=12)
        ax.legend(title=' Grupos Identificados:', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        st.pyplot(fig)
        return df
    except Exception as e:
        st.error(f"⚠️ Ocorreu um erro na análise: {str(e)}")
        return None

# Interface principal
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox(
    "Escolha uma análise",
    ["Clusterização de Clientes"]
)

df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head())

    if analise_selecionada == "Clusterização de Clientes":
        if {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
            df = clusterizar_clientes(df)
        else:
            st.warning("⚠️ Dados incompletos! A planilha deve conter: 'idade', 'frequencia_compra' e 'gasto_medio'.")

# Botão de limpeza de dados
st.sidebar.button("🗑️ Limpar Dados", on_click=lambda: st.session_state.pop('df', None))

# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")
