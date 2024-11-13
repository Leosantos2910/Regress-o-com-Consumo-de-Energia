# No terminal do PyCharm, instale as bibliotecas necessárias:
# pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregando o dataset
# Certifique-se de que o arquivo 'dados-para-o-dash.txt' esteja no mesmo diretório do projeto,
# ou forneça o caminho completo para o arquivo.
df = pd.read_csv('dados-para-o-dash.txt', sep=';', na_values='?')

# Pré-processamento de dados
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
df = df.dropna()

# Seleção de variáveis
X = df[['Global_active_power', 'Month', 'Hour']]
y = df['Global_intensity']

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Realizando previsões
y_pred = model.predict(X_test)

# Calculando o Erro Médio Quadrático
print("AMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualizando resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Previstos')
sns.scatterplot(x=y_test, y=y_test, color='red', label='Reais')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Previstos')
plt.legend()
plt.show()
