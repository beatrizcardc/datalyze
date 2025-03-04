import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta

# Configuração da página
st.set_page_config(page_title="Datalyze - Análise Inteligente de Negócios", layout="wide")

# Título do App
st.title("📊 Datalyze - Análise Inteligente de Negócios")
st.write("Bem-vindo! Aqui você pode carregar seus dados e aplicar técnicas de análise para obter insights valiosos.")

# Função para carregar dados
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo XLSX", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            planilhas = xls.sheet_names
            
            if "Vendas" in planilhas and "Clientes" in planilhas:
                df_vendas = pd.read_excel(xls, sheet_name="Vendas")
                df_clientes = pd.read_excel(xls, sheet_name="Clientes")
                return df_vendas, df_clientes
            else:
                st.error("Erro: O arquivo deve conter as abas 'Vendas' e 'Clientes'.")
                return None, None
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None, None
    return None, None

# Carregar os dados
df_vendas, df_clientes = carregar_dados()

if df_vendas is not None and df_clientes is not None:
    # Exibir dados carregados
    st.write("### 📋 Dados de Vendas")
    st.dataframe(df_vendas.head())
    
    st.write("### 📋 Dados de Clientes")
    st.dataframe(df_clientes.head())
    
    # Processamento das datas
    df_vendas['data'] = pd.to_datetime(df_vendas['data'])
    df_vendas['dia_semana'] = df_vendas['data'].dt.dayofweek + 1  # 1 = Segunda-feira, 7 = Domingo

    # Criar modelo de previsão de vendas
    st.write("### 📈 Previsão de Vendas")
    
    # Definir variáveis para previsão
    variaveis = ['dia_semana', 'horario', 'temperatura']
    df_vendas = df_vendas.dropna(subset=['vendas'])
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_vars = encoder.fit_transform(df_vendas[['dia_semana', 'horario']])
    cols = encoder.get_feature_names_out(['dia_semana', 'horario'])
    df_encoded = pd.DataFrame(encoded_vars, columns=cols)
    df_vendas = pd.concat([df_vendas, df_encoded], axis=1)
    
    model = LinearRegression()
    X = df_vendas[cols]
    y = df_vendas['vendas']
    model.fit(X, y)
    df_vendas['previsao'] = model.predict(X)
    
    # Gráfico de tendências de vendas
    fig, ax = plt.subplots(figsize=(12, 6))
    df_vendas = df_vendas.sort_values('data')
    ax.plot(df_vendas['data'], df_vendas['vendas'], label='Vendas Reais', marker='o')
    ax.plot(df_vendas['data'], df_vendas['previsao'], linestyle='--', color='red', label='Previsão', marker='s')
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.write("### 🔍 Explicação da Previsão")
    st.markdown("""
    - O gráfico acima mostra a **tendência de vendas reais e as previsões geradas pelo modelo**.
    - Se a previsão acompanha os dados reais, o modelo está funcionando bem.
    - **Atenção**: Se houver grande discrepância, pode indicar eventos atípicos, como promoções ou sazonalidade.
    
    **Como usar essa previsão no seu negócio?**
    - Planejar estoque e produção conforme os períodos de alta demanda.
    - Ajustar promoções para aumentar vendas em dias de baixa previsão.
    - Melhorar estratégias de precificação para otimizar o faturamento.
    """)
    
    # Exibir coeficientes do modelo
    st.write("### 🔍 Influência das Variáveis")
    coeficientes = pd.DataFrame({'Variável': cols, 'Impacto': model.coef_}).sort_values('Impacto', ascending=False)
    st.dataframe(coeficientes.style.bar(color='#5fba7d', subset=['Impacto']))
    
    # Análise de Clusterização de Clientes
    st.write("### 🧩 Clusterização de Clientes")
    if {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df_clientes.columns):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_clientes['cluster'] = kmeans.fit_predict(df_clientes[['idade', 'frequencia_compra', 'gasto_medio']])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for cluster in sorted(df_clientes['cluster'].unique()):
            dados_cluster = df_clientes[df_clientes['cluster'] == cluster]
            ax.scatter(dados_cluster['idade'], dados_cluster['gasto_medio'], s=100, c=cores[cluster], label=f'Grupo {cluster + 1}', alpha=0.7)
        
        ax.set_xlabel("Idade")
        ax.set_ylabel("Gasto Médio (R$)")
        ax.legend(title="Grupos de Clientes")
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("""
        **Como interpretar?**
        - Clientes foram agrupados com base em idade, frequência de compras e gasto médio.
        - Isso ajuda a **identificar perfis de clientes** e criar estratégias direcionadas.
        
        **Exemplo de Estratégias:**
        - **Grupo 1:** Clientes jovens e frequentes → Promoções e descontos.
        - **Grupo 2:** Adultos de renda média → Programas de fidelidade.
        - **Grupo 3:** Clientes premium → Produtos exclusivos e atendimento especial.
        """)





