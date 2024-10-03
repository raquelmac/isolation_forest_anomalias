import time
# Início da contagem de tempo
start_time_external = time.time()

###################################################################################
#---------------------------- IMPORTAR AS BIBLIOTECAS  ----------------------------
###################################################################################

try:

    # Início da contagem de tempo
    start_time = time.time()

    #import pandas as pd
    from auxiliar.conexoes import obter_dados_pl#, obter_dados_mysql
    import polars as pl
    #import numpy as np
    #import matplotlib.pyplot as plt
    #from dotenv import load_dotenv
    #import os # Obter as constantes do .env

    # carregando variáveis de ambiente .env
    # r (raw) string busca o arquivo na raiz do projeto
    # . vai direcionar a pasta atual
    #load_dotenv(r"./auxiliar/.env") # Navegue no prompt até a pasta UC2

    # Fim da contagem de tempo
    end_time = time.time()
    
    # Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'BIBLIOTECAS IMPORTADAS! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('Erro na importação das bibliotecas', e)
    exit()

###################################################################################
#----------------------------------- OBTER DADOS ----------------------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()
    
    # Diretório onde os arquivos estão
    ENDERECO_DADOS = r'./Dados/camada_raw/'

    # CAMADA RAW DA ARQUITETURA MEDALLION
    # Lista dos dados de bolsa família
    lista_arquivos = ['202404_NovoBolsaFamilia.csv',
                      '202405_NovoBolsaFamilia.csv']
    

    # Estrutura de repetição para percorrer a lista de arquivos
    for arquivo in lista_arquivos:

        df = obter_dados_pl(ENDERECO_DADOS + arquivo,'','csv',';') 
        
        # Append dos dados
        if 'df_bf' in locals():
            df_bf = pl.concat([df_bf, df])
        else:
            df_bf = df
    
    #print(df_bf['MÊS COMPETÊNCIA'].unique())
    #print(df_bf['UF'].value_counts().sort_values(ascending=True))
    #print(df_bf.head(2))

    # Fim da contagem de tempo
    end_time = time.time()

    # Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'DADOS OBTIDOS! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO OBTER DADOS', e)
    exit()

###################################################################################
#--------------------------- AJUSTAR O TIPO DAS COLUNAS ---------------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()
    
    # Substitui , para . na coluna VALOR PARCELA e a converte para float
    df_bf = df_bf.with_columns(
            pl.col('VALOR PARCELA').str.replace(',', '.').cast(pl.Float64)
    )
    

    print(f'TIPO DAS COLUNAS AJUSTADOS! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO AJUSTAR O TIPO DAS COLUNAS', e)
    exit()

###################################################################################
#----------------- EXPORTAR DADOS DO BOLSA FAMILIA COMO PARQUET --------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()
    
    # Diretório onde os arquivos serão salvos
    ENDERECO_DADOS = r'./Dados/camada_bronze/'
    
    # Carrega na camada Bronze (Arquitetura Medallion) os arquivos da camada raw já processados 
    df_bf.write_parquet(ENDERECO_DADOS + 'dados_bolsa_familia_abr_mai.parquet')
    

    print(f'DADOS DO BOLSA FAMILIA EXPORTADOS COMO PARQUET! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO EXPORTAR DADOS DO BOLSA FAMILIA COMO PARQUET', e)
    exit()

# Fim da contagem de tempo
end_time_external = time.time()

# Calcular a diferença (tempo de execução)
execution_time_external = end_time_external - start_time_external
print(f"Tempo total de execução: {execution_time_external:.2f} segundos")