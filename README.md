# 📊 Datalyze - Análise Inteligente de Negócios

## 📝 Sobre o Datalyze

O **Datalyze** é um aplicativo desenvolvido para ajudar empresas a analisar dados de vendas e comportamento de clientes de maneira intuitiva. Com ele, é possível prever vendas, segmentar clientes e realizar testes estatísticos para melhorar a tomada de decisões.

## 🚀 Funcionalidades Principais

### 📈 1. Previsão de Vendas
- Modelo preditivo baseado em variáveis como dia da semana, horário e temperatura.
- Permite visualizar o impacto de cada variável sobre as vendas.
- Gráfico comparativo entre vendas reais e previstas.

### 👥 2. Clusterização de Clientes
- Segmentação de clientes baseada em idade, frequência de compra e gasto médio.
- Identifica perfis de clientes para estratégias de marketing personalizadas.
- Visualização gráfica dos grupos identificados.

### 🔬 3. Testes Estatísticos
- Comparação de vendas entre diferentes categorias de produtos.
- Indica se houve diferença significativa entre grupos.
- Utiliza testes estatísticos como **T-Student** e **ANOVA**.

### 🔎 4. Mapa de Correlação
- Exibe a relação entre diferentes variáveis.
- Ajuda a identificar fatores que mais influenciam as vendas.

## 📥 Como Usar

1. **Carregue seus dados** no formato CSV ou Excel.
2. **Escolha a análise desejada**: previsão de vendas, clusterização de clientes ou testes estatísticos.
3. **Explore os insights** gerados pelos gráficos e tabelas.
4. **Utilize os resultados** para otimizar suas estratégias de negócio.

## 📊 Formato dos Dados

### **Planilha 1: Vendas**
| data       | dia_semana | horario | temperatura | produto    | vendas | categoria_produto |
|------------|-----------|---------|-------------|------------|--------|-------------------|
| 10-03-2024 | 2         | 14      | 25.0        | Produto A  | 10     | Bebidas          |
| 10-03-2024 | 2         | 18      | 27.5        | Produto B  | 15     | Alimentos        |

### **Planilha 2: Clientes**
| Nome do Cliente | idade | frequencia_compra | gasto_medio |
|----------------|-------|-------------------|-------------|
| Cliente 1     | 30    | 1                 | 100,00      |
| Cliente 2     | 45    | 3                 | 250,00      |

## 📌 Interpretação dos Resultados

- **Previsão de Vendas**:
  - A linha **azul** representa as vendas reais.
  - A linha **vermelha pontilhada** representa a previsão feita pelo modelo.
  - Se houver grande diferença entre previsão e vendas reais, pode ser necessário revisar os fatores utilizados.

- **Clusterização de Clientes**:
  - **Grupo 1 (vermelho)**: Clientes mais jovens, menor gasto por compra e maior frequência.
  - **Grupo 2 (verde)**: Clientes de meia-idade, gasto moderado e compras regulares.
  - **Grupo 3 (azul)**: Clientes maduros, maior gasto por compra e menor frequência.

- **Testes Estatísticos**:
  - Se o **p-valor < 0.05**, há diferença significativa entre os grupos.
  - Se **p-valor ≥ 0.05**, os grupos são estatisticamente semelhantes.

## 🛠️ Tecnologias Utilizadas
- **Python**
- **Streamlit**
- **Pandas**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **SciPy**

## 📧 Contato

👥 **Beatriz Cardoso Cunha**  
📧 Email: beacarcun@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/beatriz-cardoso-cunha/)

