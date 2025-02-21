#Versão Final
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
def carregar_dados(analise_selecionada):
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                xls = pd.ExcelFile(uploaded_file)
                
                # Mapeamento de planilhas sugeridas por tipo de análise
                sheet_mapping = {
                    "Previsão de Vendas": ["Vendas"],
                    "Clusterização de Clientes": ["Clientes"],
                    "Testes": ["Testes"]
                }
                
                sheets_sugeridas = sheet_mapping.get(analise_selecionada, [])
                todas_sheets = xls.sheet_names
                sheets_validas = [s for s in todas_sheets if s in sheets_sugeridas] or todas_sheets
                
                sheet_selecionada = st.sidebar.selectbox(
                    "Selecione a planilha:",
                    sheets_validas,
                    help=f"Planilhas recomendadas para {analise_selecionada}: {', '.join(sheets_sugeridas)}"
                )
                
                df = pd.read_excel(xls, sheet_name=sheet_selecionada)
            
            # Processamento de datas
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

# Função de previsão de vendas com múltiplas variáveis
def previsao_vendas_avancada(df):
    st.sidebar.subheader("⚙️ Variáveis de Influência")

    # Definição das variáveis disponíveis
    variaveis_disponiveis = ['dia_semana', 'produto', 'horario', 'temperatura']
    
    # Widget para seleção de variáveis
    variaveis_selecionadas = st.sidebar.multiselect(
        "Selecione fatores de influência:",
        options=variaveis_disponiveis,
        default=['dia_semana']
    )

    # Verificação para evitar erro caso o usuário não selecione nada
    if not variaveis_selecionadas:
        st.warning("⚠️ Selecione pelo menos uma variável para análise.")
        st.stop()

    # Pré-processamento das variáveis
    features = []
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    for var in variaveis_selecionadas:
        if var in df.columns:
            if var in ['dia_semana', 'produto', 'horario']:  # One-Hot Encoding para variáveis categóricas
                encoded = encoder.fit_transform(df[[var]])
                cols = [f"{var}_{v}" for v in encoder.categories_[0]]
                df[cols] = encoded
                features.extend(cols)
            elif var == 'temperatura':  # Mantém variável numérica sem modificação
                df['temperatura'] = pd.to_numeric(df['temperatura'], errors='coerce')  # Converte para numérico
                features.append(var)
        else:
            st.warning(f"⚠️ A variável `{var}` não foi encontrada no DataFrame. Ignorando...")      

    # Exibir as variáveis processadas
    if features:
        st.write("### 🔍 Variáveis Processadas:", features)
    else:
        st.error("⚠️ Nenhuma variável foi processada. A modelagem não pode continuar.")
        return

    # Modelagem
    try:
        model = LinearRegression()
        model.fit(df[features], df['vendas'])
        df['previsao'] = model.predict(df[features])

        # Visualização
        st.write("### 📈 Modelo Preditivo Multivariado")

       # Mapeamento de números para os dias da semana
        dias_semana_mapeamento = {
            1: "Domingo", 2: "Segunda", 3: "Terça", 4: "Quarta", 
            5: "Quinta", 6: "Sexta", 7: "Sábado"
        }
        
        # Converter números para os nomes correspondentes
        df['dia_semana'] = df['dia_semana'].map(dias_semana_mapeamento)
        
        # Garantir que a ordem dos dias fique correta no gráfico
        dias_semana_ordem = ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"]
        df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_semana_ordem, ordered=True)
        
        # Agrupar os dados por dia da semana e calcular a média de vendas e previsão
        df_grouped = df.groupby('dia_semana')[['vendas', 'previsao']].mean().reindex(dias_semana_ordem)
        
        # Criar gráfico com os dias da semana no eixo X
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_grouped.index, df_grouped['vendas'], label='Vendas Reais', marker='o')
        ax.plot(df_grouped.index, df_grouped['previsao'], linestyle='--', color='red', label='Previsão', marker='s')
        
        ax.set_xlabel("Dia da Semana")
        ax.set_ylabel("Vendas")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)



        # Exibição dos coeficientes
        st.write("### 🔍 Influência dos Fatores")
        coeficientes = pd.DataFrame({
            'Variável': features,
            'Impacto': model.coef_
        }).sort_values('Impacto', ascending=False)

        st.dataframe(coeficientes.style.bar(color='#5fba7d', subset=['Impacto']))

    except Exception as e:
        st.error(f"Erro no modelo: {str(e)}")
        
#Novo Heatmap
    st.write("## 🔥 Mapa de Calor - Padrão Completo de Vendas")
    
    if {'horario', 'dia_semana', 'produto', 'temperatura', 'vendas'}.issubset(df.columns):
        
        # Criar uma tabela pivot com a média de vendas por combinação de horário, dia da semana e temperatura
        df_pivot = df.groupby(['dia_semana', 'horario']).agg({'vendas': 'sum', 'temperatura': 'mean', 'produto': 'count'}).reset_index()
    
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = sns.scatterplot(
            data=df_pivot, 
            x="dia_semana", 
            y="horario", 
            size="produto", # Tamanho representa a quantidade de produtos vendidos
            hue="temperatura", # Cor representa a temperatura
            palette="coolwarm", 
            sizes=(20, 500), # Define o tamanho dos pontos no gráfico
            edgecolor="black", 
            ax=ax
        )
    
        ax.set_title("📊 Mapa de Calor de Vendas por Dia da Semana e Horário")
        ax.set_xlabel("Dia da Semana")
        ax.set_ylabel("Horário")
        ax.grid(True, linestyle="--", alpha=0.5)
    
        # Criar uma legenda personalizada para representar a quantidade de vendas
        handles, labels = scatter.get_legend_handles_labels()
        labels[0] = "Quantidade de Produtos Vendidos"
        labels[len(labels)//2] = "Temperatura Média"
        ax.legend(handles, labels, title="Legendas", loc="upper right", fontsize="small")
    
        st.pyplot(fig)

# Interface principal
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox(
    "Escolha uma análise",
    ["Previsão de Vendas", "Clusterização de Clientes", "Testes"]
)

df = carregar_dados(analise_selecionada)

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head().style.format({"data": lambda t: t.strftime("%d/%m/%Y")}))

    if analise_selecionada == "Previsão de Vendas":
        if {'data', 'vendas'}.issubset(df.columns):
            previsao_vendas_avancada(df)
        else:
            st.warning("⚠️ Dados incompletos! Necessário colunas 'data' e 'vendas'.")

# Botão de limpeza de dados
st.sidebar.button("🗑️ Limpar Dados", on_click=lambda: st.session_state.pop('df', None))

# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")
