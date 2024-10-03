'''Verifique a previsão dos registros de Crimes violentos letais 
intencionais por BPM, se nos próximos 6 meses, os homicídios 
dolosos forem:
• Mês1: 50.000
• Mês2: 75.000
• Mês3: 80.000
• Mês4: 90.000
• Mês5: 95.000
• Mês6: 100.000'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auxiliar.conexoes import obter_dados_pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constante do Endereço dos dados
ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

# obter dados
# comma separeted values (CSV), em português é valores separados por vírgulas
# mas, que nem sempre são vírgulas. É preciso verificar
try:
    print('Obtendo dados de ocorrências...')

    # parâmentros: endereco_arquivo, nome_arquivo, tipo_arquivo, separador

    #df_ocorrencias = pd.read_csv(ENDERECO_DADOS, sep=';', encoding='LATIN-1')
    df_ocorrencias = obter_dados_pd(ENDERECO_DADOS,'','csv',';')

    df_ocorrencias = df_ocorrencias[(df_ocorrencias['ano']>=2022) &
                                    (df_ocorrencias['ano']>=2023)]

    #print(df_ocorrencias.head()) #head sem valor, trará as 5 primeiras linhas

    print('Dados obtidos com sucesso!')
except Exception as e:
    print('Erro ao obter dados: ', e)
    exit()

# delimitar somente as variáveis solicitadas e totalizar
try:
    print("inciando a delimitação das variáveis e a totalização...")
    
    # cidade e roubo de veículos
    #print(df_ocorrencias.columns) # exibir o nome de todas as colunas
    df_cvli = df_ocorrencias[['aisp','hom_doloso','cvli']]

    # totalizar o dataframe
    df_total_cvli = df_cvli.groupby('aisp')\
                            .sum(['hom_doloso','cvli']).reset_index()
    
    #print(df_total_cvli)

    print('Delimitação e totalização concluídas!')
except Exception as e:
    print("Erro ao delimitar o dataframe: ", e)
    exit()

# correlação
try:
    print('Correlacionando dados...')

    array_hom_dolosos = np.array(df_total_cvli['hom_doloso'])
    array_cvli = np.array(df_total_cvli['cvli'])

    correlacao = np.corrcoef(array_hom_dolosos,array_cvli)[0,1]

    print('Correlação: ', correlacao)

except Exception as e:
    print("Erro ao correlaciona dados: ", e)
    exit()

# Regressão linear
try:
    print('Iniciando a análise preditiva...')

    # dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        array_hom_dolosos,
        array_cvli,
        test_size=0.25,
        random_state=42
    )

    # normalizar os dados
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.reshape([-1,1]))
    X_test = scaler.transform(X_test.reshape([-1,1]))

    # treinar o modelo
    modelo = LinearRegression()

    # Aqui que é gerada a função linear
    modelo.fit(X_train, y_train)

    # Coeficiente de determinação (Score)
    r2_score = modelo.score(X_test, y_test)

    print('R² Score: ', r2_score)

    # aplicar a predição nos dados de teste
    predicao = modelo.predict(X_test)

    # Array com os dados de homicídio doloso para prever cvli
    '''
    •	Mês1: 50.000
    •	Mês2: 75.000
    •	Mês3: 80.000
    •	Mês4: 90.000
    •	Mês5: 95.000
    •	Mês6: 100.000

    '''
    array_hom_doloso_pred = np.array([50000,75000,80000,90000,95000,100000])

    # normalizar os dados de hom doloso
    array_hom_doloso_pred_scaled = scaler.transform(array_hom_doloso_pred.reshape(-1,1))

    # prever cvli
    cvli_pred = modelo.predict(array_hom_doloso_pred_scaled)

    print('Previsão CVLI: ', cvli_pred)
except Exception as e:
    print("Erro ao realizar a análise preditiva: ", e)
    exit()

# avaliação do modelo
try:
    print('Avaliando o modelo de previsões...')

    plt.subplots(2,2,figsize=(15,5))
    plt.suptitle('Avaliação do modelo de regressão')

    #posição 1: Gráfico de dispersão entre os arrays
    plt.subplot(2,2,1)

    sns.regplot(x=array_hom_dolosos,y=array_cvli)
    plt.title('Gráfico de dispersão')
    plt.xlabel('Hom. dolosos')
    plt.ylabel('CVLI')

    plt.text(min(array_hom_dolosos),
             max(array_cvli),
             f'Correlação: {correlacao}',
             fontsize=10)

    #posição 2: Gráfico de dispersão entre os dados reais e previsto
    plt.subplot(2,2,2)

    # retornar os dados de teste para escala real
    X_test = scaler.inverse_transform(X_test)

    # Gráfico de dispersão sem a linha de regressão
    plt.scatter(X_test,y_test, color='blue', label='Dados reais')
    plt.scatter(X_test,predicao, color = 'red', label='Previsões')

    plt.title('Dados reais x previstos')
    plt.xlabel('Hom. Dolosos')
    plt.ylabel('CVLI')

    plt.legend()

    #posição 3: Resíduos
    plt.subplot(2,2,3)

    # Os resíduos são a diferença entre os dados reais e previstos
    # cálculo dos resíduos
    residuos = y_test - predicao

    # plotar em gráfico de dispersão
    plt.scatter(predicao,residuos)

    # adicioanar uma linha constante no 0
    plt.axhline(y=0, color='black', linewidth=2)

    plt.title('Resíduos')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')

    # posição 4: dispersão dos valores simulados
    plt.subplot(2,2,4)
    plt.scatter(array_hom_doloso_pred,cvli_pred)

    plt.title('Recuperações de veículos simuladas')
    plt.xlabel('Roubo veículo simulado')
    plt.ylabel('Recuperação de veículo prevista')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Erro ao avaliar o modelo: ", e)
    exit()