'''Exercicio10:


Resposta: Não. Em razão do resultado da correlação ser forte e positiva, quando o roubo aumenta a recuperação de veiculos aumenta junto'''

###################################################################################
#----------------------------- IMPORTAR BIBLIOTECAS -------------------------------
###################################################################################

import pandas as pd
from auxiliar.conexoes import obter_dados_pd
import numpy as np


###################################################################################
#----------------------------------- OBTER DADOS ----------------------------------
###################################################################################

#Constante da URL dos dados
ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

try:

    #df_ocorrencias = obter_dados_pd(ENDERECO_DADOS,"", "csv",";")
    df_ocorrencias = pd.read_csv(ENDERECO_DADOS, sep=";", encoding="LATIN-1")
   
    print('Dados obtidos com sucesso!')
    #print(df_ocorrencias.head())

except Exception as e:
    print('Erro ao obter dados', e)
    exit()

###################################################################################
#-------------- Seleciona as colunas e obtem o toral roub.vei por mun -------------
###################################################################################

try:
    # Cidade e roubo de veículos
    #print(df_ocorrencias.columns)
    print('Iniciando o processo de seleção e agrupamento')
    df_recuperacao_veiculos = df_ocorrencias[['cisp', 'recuperacao_veiculos', 'roubo_veiculo']]

    #print(df_roubo_veiculo.head())
    df_total_recup_roubo = df_recuperacao_veiculos.groupby(['cisp']).sum(['recuperacao_veiculos', 'roubo_veiculo']).reset_index()
    #print(df_total_recup_roubo)
    #print(df_total_recuperacao_veiculos.sort_values(by='recuperacao_veiculos', ascending=False))
    print('Seleção e agrupamento concluído!')
except Exception as e:
    print('Erro ao selecionar ou agrupar dataset!', e)
    exit()

###################################################################################
#-------------------- CORRELAÇÃO ENTRE ROUBO E RECUPERAÇÃO ------------------------
###################################################################################

try:
    
    # Arrays das variáveis quantitativa do candidato filtrado
    array_recup = np.array(df_total_recup_roubo['recuperacao_veiculos'])
    array_roubo = np.array(df_total_recup_roubo['roubo_veiculo'])

    # Calcular o coeficiente de correalação de Pearson (r)
    correlacao = np.corrcoef(array_recup, array_roubo)[0,1]

    print(f'Correlação: {correlacao:.2f}')
    
except Exception as e:
    print('Erro ao selecionar colunas', e)
    exit()

###################################################################################
#---------------------- APLICAR MODELO DE REGRESSÃO LINEAR ------------------------
###################################################################################

try:
    
    #classe pra importação do modelo
    from sklearn.linear_model import LinearRegression

    #classe pra dividir os dados em treino e teste
    from sklearn.model_selection import train_test_split

    X = array_roubo
    y = array_recup
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42 #42 é o numero do guia dos mochileiro das galaxias
                                                        )

    # Normalização dos dados das variáveis independentes 
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.reshape(-1,1))

    X_test = scaler.transform(X_test.reshape(-1,1))

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    #R2 É O COEFICIENTE DE DETERMINAÇÃO
    r2_score = modelo.score(X_test, y_test)

    print('R2 score:', r2_score)
    
    # Prever a qntde de roubo de veículos com base nos valores de recuperção de veiculos 
    array_roubo_pred = np.array([400000, 50000, 600000])
    array_roubo_pred_scaled = scaler.transform(
                        array_roubo_pred.reshape(-1,1)
    )

    #prever o roubo
    recup_pred = modelo()
    # Lembre-se que os dados estão normalizados.



    
except Exception as e:
    print('ERRO AO APLICAR MODELO DE REGRESSÃO LINEAR', e)
    exit()
