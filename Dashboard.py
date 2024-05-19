# Bibliotecas básicas de data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Streamlit
import streamlit as st

# Bibliotecas para scrapping do site IPEA
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Para machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Para deep learning
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.metrics import MeanSquaredError

# Função para prever os próximos 'num_prediction' pontos da série temporal
# Utiliza o modelo treinado para prever cada ponto sequencialmente
# A cada iteração, adiciona a previsão à lista 'prediction_list'

def predict(num_prediction, model):
    prediction_list = fechamento_s[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

# Função para gerar as datas dos próximos 'num_prediction' dias
# Assume que o DataFrame 'df' possui uma coluna 'Date' contendo as datas

def predict_dates(num_prediction):
    last_date = df_lstms['Data'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

##==========================================================================================
## Streamlit formatting
st.set_page_config(layout='wide')


coluna1, coluna2 = st.columns(2)
with coluna1:
    st.title(":fuelpump: PREÇO DO PETRÓLEO BRENT")
with coluna2:
    st.markdown('<h4 style="text-align: right;">FIAP PosTech - Tech Challenge - Fase 4</h4>', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: right;">Marcello Vartuli - RM352151</h4>', unsafe_allow_html=True)

##==========================================================================================

# Atualiza o dataframe com novos dados
def update_dataframe(df, df_novo):

    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    df_novo['Data'] = pd.to_datetime(df_novo['Data'], dayfirst=True)

    last_date = df['Data'].max()
    new_rows = df_novo[df_novo['Data'] > last_date]

    # Concatena os novos dados com o DataFrame existente se houver novas linhas
    if not new_rows.empty:
        updated_df = pd.concat([df, new_rows], ignore_index=True)
    else:
        updated_df = df

    return updated_df

# URL do site IPEADATA
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

# Faz uma requisição GET ao site e captura a resposta
response = requests.get(url)

# Verifica se a requisição foi bem sucedida
if response.status_code == 200:
    # Cria um objeto BeautifulSoup para analisar o HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    # Procura pela tabela no HTML analisado
    table = soup.find('table', {'id': 'grd_DXMainTable'})
    # Usa o pandas para ler a tabela HTML diretamente para um DataFrame
    new_df = pd.read_html(str(table), header=0)[0]
    new_df.rename(columns={"Preço - petróleo bruto - Brent (FOB)": "Fechamento"}, inplace=True)
    new_df['Fechamento'] = new_df['Fechamento']/100
  
    # Verifica se o arquivo do DataFrame existe e carrega, ou cria um novo DataFrame se não existir
    path = 'ipea.csv'
    try:
        existing_df = pd.read_csv(path)
    except FileNotFoundError:
        existing_df = new_df  # Se o arquivo não existir, considere os dados atuais como o DataFrame existente

    # Atualiza o DataFrame existente com novos dados (carga incremental)
    updated_df = update_dataframe(existing_df, new_df)

    # Salva o DataFrame atualizado para o arquivo
    updated_df.to_csv(path, index=False)

    # Mostra as primeiras linhas do DataFrame atualizado
    updated_df.set_index('Data')
    print(updated_df.head())
else:
    print('Falha ao acessar a página: Status Code', response.status_code)

df_eda = updated_df.set_index('Data').sort_index(ascending=True)

df_grafico = df_eda.reset_index()

fig = px.line(df_grafico, x="Data", y="Fechamento", title="Evolução de Preço Petróleo Brent")
fig.update_xaxes(title="Data")
fig.update_yaxes(title="Preço US$")


texto = '''
>
> ### Evolução dos preços
>
> ##### No gráfico abaixo podemos observar a evolução do preço em US$ do Petróleo Brent, nota-se a forte queda do preço do petróleo nos seguintes eventos de impacto global:
>
> **Juho/2008 a Dezembro/2008 -** A crise mundial provocada pela bolha imobiliaria no Estados Unidos seguida pela recessão provocou a desaceleração da economia e queda nos preços do petróleo.
>
> **Junho/2014 a Janeiro/2015 -** Aumento da produção de petróleo nas Áreas de Xisto dos Estados Unidos provoca aumento da oferta que não foi acompanhada pelo consumo causando queda nos preços.
>
> **Janeiro/2020 a Abril/2020 -** A pandemia de COVID-19 causou diminuição do consumo mundial devido as medidas de isolamento, causando a queda nos preços.
'''
st.markdown(texto)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

df_lstms = df_eda.reset_index('Data')

# Suazivando a série temporal
# Aplicando suavização exponencial
alpha = 0.09   # Fator de suavização
# O parâmetro alpha na suavização exponencial controla a taxa de decaimento dos pesos atribuídos às observações passadas.
# Determina o quão rapidamente o impacto das observações antigas diminui à medida que você avança no tempo.

df_lstms['Smoothed_Close'] = df_lstms['Fechamento'].ewm(alpha=alpha, adjust=False).mean()

df_lstms.drop(columns=['Fechamento'], inplace=True)

fechamento_s = df_lstms['Smoothed_Close'].values
fechamento_s = fechamento_s.reshape(-1,1) #transformar em array

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(fechamento_s)
fechamento_s = scaler.transform(fechamento_s)

split_percent = 0.80
split = int(split_percent*len(fechamento_s))

fechamento_s_train = fechamento_s[:split]
fechamento_s_test = fechamento_s[split:]

data_s_train = df_lstms['Data'][:split]
data_s_test = df_lstms['Data'][split:]

look_back = 5

train_generator = TimeseriesGenerator(fechamento_s_train, fechamento_s_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(fechamento_s_test, fechamento_s_test, length=look_back, batch_size=1)

## Carrega o modelo treinado
neural_model = model = tensorflow.keras.models.load_model('modelobrent')
    
prediction = neural_model.predict(test_generator)

fechamento_s_train = fechamento_s_train.reshape((-1))
fechamento_s_test = fechamento_s_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = data_s_train,
    y = fechamento_s_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = data_s_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = data_s_test,
    y = fechamento_s_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Predições da Petróleo Brent",
    xaxis = {'title' : "Data"},
    yaxis = {'title' : "Fechamento"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

texto = '''
>
> ### Predições dos preços
>
> ##### No gráfico a seguir podemos verificar o modelo de predição LSTM utilizado neste estudo. A partir de Julho/2016 podemos ver o desempenho do modelo comparado com os dados reais apurados.
> ##### Neste modelo foi aplicada a técnica de Suavização Exponencial.
'''
st.markdown(texto)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

fechamento_s = fechamento_s.reshape((-1))

num_prediction = 15 #definição dos próximos dias
forecast = predict(num_prediction, neural_model) #resultado de novos dias
forecast_dates = predict_dates(num_prediction)

trace1 = go.Scatter(
    x = data_s_test,
    y = fechamento_s_test,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Prediction'
)
layout = go.Layout(
    title = "Forecast Petróleo Brent",
    xaxis = {'title' : "Data"},
    yaxis = {'title' : "Fechamento"}
)
fig = go.Figure(data=[trace1, trace2], layout=layout)

texto = '''
>
> ### Predição dos próximos 15 dias
>
> ##### Neste gráfico podemos verificar a predição para os próximos 15 dias.
'''
st.markdown(texto)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

df_past = pd.DataFrame(df_lstms)
df_past = df_past[['Data','Smoothed_Close']]
df_past.rename(columns={'Smoothed_Close': 'Actual'}, inplace=True)         #criando nome das colunas
df_past['Data'] = pd.to_datetime(df_past['Data'])                          #configurando para datatime
df_past['Forecast'] = np.nan                                               #Preenchendo com NAs
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

# Faz a transformação inversa das predições
forecast = forecast.reshape(-1, 1) #reshape para array
forecast = scaler.inverse_transform(forecast)

df_future = pd.DataFrame(columns=['Data', 'Actual', 'Forecast'])
df_future['Data'] = forecast_dates
df_future['Forecast'] = forecast.flatten()
df_future['Actual'] = np.nan

# Concatenando os DataFrames usando concat
frames = [df_past, df_future]
results = pd.concat(frames, ignore_index=True).set_index('Data')

results2023 =  results.loc['2023-01-01':]

plot_data = [
    go.Scatter(
        x=results2023.index,
        y=results2023['Actual'],
        name='actual'
    ),
    go.Scatter(
        x=results2023.index,
        y=results2023['Forecast'],
        name='prediction'
    )
]

plot_layout = go.Layout(
        title='Forecast Petróleo Brent'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)

texto = '''
>
> ### Predição dos próximos 15 em US$
>
> ##### Neste gráfico podemos verificar a predição do valor em US$ para os próximos 15 dias.
'''
st.markdown(texto)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')