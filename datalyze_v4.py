# Versão Final v4 - Datalyze Integrado sem Planilha de Testes
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, f_oneway

# Configuração da página
st.set_page_config(page_title="Datalyze - Análise Inteligente de Negócios", layout="wide")

# Título do App
st.title("📊 Datalyze - Análise Inteligente de Negócios")
st.write("Bem-vindo! Aqui você pode carregar seus dados e aplicar técnicas de análise para obter insights valiosos.")

# Função para carregar os dados
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, sheet_name="Vendas")  # Carregar apenas a aba "Vendas"
            
            df['Data'] = pd.to_datetime(df['Data'])  # Garantir que a coluna Data seja datetime
            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None
    return None

# Função para análise estatística e previsão de vendas
def analise_estatistica(df):
    st.header("📊 Estatísticas de Vendas")

    # Criando categorias para segmentação
    df['Categoria'] = df['Produto'].apply(lambda x: 'Categoria 1' if 'esp' in x.lower() else 'Categoria 2')

    # Média e Variância por Categoria
    media_por_categoria = df.groupby('Categoria')['Vendas'].mean()
    variancia_por_categoria = df.groupby('Categoria')['Vendas'].var()

    # Média e Variância por Dia da Semana
    df['DiaSemana'] = df['Data'].dt.day_name()
    media_por_dia = df.groupby('DiaSemana')['Vendas'].mean()
    variancia_por_dia = df.groupby('DiaSemana')['Vendas'].var()

    # Exibição dos dados no Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🏷️ Média e Variância por Categoria")
        st.dataframe(pd.DataFrame({'Média': media_por_categoria, 'Variância': variancia_por_categoria}))

    with col2:
        st.write("### 📅 Média e Variância por Dia da Semana")
        st.dataframe(pd.DataFrame({'Média': media_por_dia, 'Variância': variancia_por_dia}))

    # Teste T para Categoria 1 x Categoria 2
    vendas_um = df[df['Categoria'] == 'Categoria 1']['Vendas']
    vendas_dois = df[df['Categoria'] == 'Categoria 2']['Vendas']
    t_stat, p_valor = ttest_ind(vendas_esp, vendas_acougue, equal_var=False)

    # ANOVA para comparar os dias da semana
    grupos_vendas = [df[df['DiaSemana'] == dia]['Vendas'] for dia in df['DiaSemana'].unique()]
    anova_stat, anova_p_valor = f_oneway(*grupos_vendas)

    st.write("### 🔬 Testes Estatísticos")
    st.metric(label="p-valor do Teste T (Categoria 1 vs Categoria 2)", value=f"{p_valor:.4f}")
    st.metric(label="p-valor da ANOVA (Diferença entre Dias da Semana)", value=f"{anova_p_valor:.4f}")

    # Explicação para o usuário leigo
    st.write("🔍 **O que isso significa?**")
    if p_valor < 0.05:
        st.success("📌 Existe uma diferença significativa nas vendas entre Espetinhos e Açougue.")
    else:
        st.info("📌 Não foi encontrada diferença significativa entre Espetinhos e Açougue.")

    if anova_p_valor < 0.05:
        st.success("📌 Existe uma variação significativa nas vendas entre os dias da semana.")
    else:
        st.info("📌 Os dias da semana não apresentam grandes diferenças de vendas.")

# Função para previsão de vendas usando Regressão Linear
def previsao_vendas(df):
    st.header("📈 Previsão de Vendas")

    # Transformando os dados
    df['DiaSemana_Num'] = df['Data'].dt.weekday  # Segunda=0, Domingo=6
    df['Temperatura'] = np.random.randint(27, 39, size=len(df))  # Simulação de temperatura

    # Seleção de features para modelo preditivo
    X = df[['DiaSemana_Num', 'Temperatura']]
    y = df['Vendas']

    # Treinando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Criando datas futuras
    dias_futuros = pd.date_range(df['Data'].max() + pd.Timedelta(days=1), periods=7)
    df_futuro = pd.DataFrame({'Data': dias_futuros})
    df_futuro['DiaSemana_Num'] = df_futuro['Data'].dt.weekday
    df_futuro['Temperatura'] = np.random.randint(27, 39, size=len(df_futuro))

    # Fazendo previsões
    df_futuro['Previsao'] = model.predict(df_futuro[['DiaSemana_Num', 'Temperatura']])

    # Exibindo gráfico de tendência
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Data'], df['Vendas'], label="Vendas Passadas", marker='o')
    ax.plot(df_futuro['Data'], df_futuro['Previsao'], label="Previsão de Vendas", linestyle='--', marker='s', color='red')
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Explicação para o usuário
    st.write("🔍 **O que isso significa?**")
    st.write("Este gráfico exibe as **vendas passadas** e uma **previsão para os próximos 7 dias**.")
    st.write("Os valores futuros são estimados com base em padrões históricos e temperatura.")

# Interface principal
df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head())

    analise_estatistica(df)
    previsao_vendas(df)
else:
    st.warning("⚠️ Carregue um arquivo para começar.")


