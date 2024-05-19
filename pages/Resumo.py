# Streamlit
import streamlit as st

##==========================================================================================
## Streamlit formatting
st.set_page_config(layout='wide')


coluna1, coluna2 = st.columns(2)
with coluna1:
    st.title(":fuelpump: PREÇO DO PETRÓLEO BRENT")
with coluna2:
    st.markdown('<h4 style="text-align: right;">FIAP PosTech - Tech Challenge - Fase 4</h4>', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: right;">Marcello Vartuli - RM352151</h4>', unsafe_allow_html=True)

st.markdown('---')

st.title("Resumo das conclusões do desafio")

st.markdown('---')

texto = '''
>
> #### Foram utilizados 3 métodos distintos para elaboração de um modelo preditivo para Petróleo Brent, tomando como base um intervalo de tempo disponível, iniciando em 20/05/1987.
>
> **LSTM** - Apresentou o melhor resultado pelos indicadores MAPE = 1.15% e Mean Square Error: 8.629660442238674e-05 com a série temporal suavizada.
> 
> **Prophet** - Apresentou o segundo melhor resultado com MAPE = 15.41%
> 
> **ARIMA** - Não provou ser um método adequado, tendo MAPE = 348.66%
>
'''

st.markdown(texto)