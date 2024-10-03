'''Exemplo 09: A imprensa'''


import time
# Início da contagem de tempo
start_time_external = time.time()

###################################################################################
#---------------------------- IMPORTAR AS BIBLIOTECAS  ----------------------------
###################################################################################

try:

    # Início da contagem de tempo
    start_time = time.time()

    import pandas as pd
    from auxiliar.conexoes import obter_dados_pl#, obter_dados_mysql
    from auxiliar.estatistica import estatistica_descritiva
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
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
    ENDERECO_DADOS = r'./Dados/camada_bronze/'

    # Ler o parquet com o POLARS (muito MENOS tempo de processamento)
    df_bf_pl = pl.read_parquet(ENDERECO_DADOS + 'dados_bolsa_familia.parquet')  

    ENDERECO_DADOS = r'./Dados/camada_raw/'
    df_votacao_pl = pl.read_csv(
                ENDERECO_DADOS + 'votacao_secao_2022_BR.csv',
                separator=';', 
                encoding="latin-1")

    # Converter o df polars para pandas vai aumentar o tempo de processamento
    #df_bf_pandas = df_bf.to_pandas()
    print(df_votacao_pl.head(2))
       
    #print(df_bf.head(2))
    #print(df_bf['MÊS COMPETÊNCIA'].unique())
    #df_bf['UF'].value_counts().sort_values(ascending=True)
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
#---------------------- AJUSTAR DF VOTACAO E DO DF BOLSA FAM. ---------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()


    # Filtrar segundo turno NR_TURNO = 2 e filtrar somente  (LULA 13, BOZO 22) NR_VOTAVEL in [13,22]
    df_votacao_pl = df_votacao_pl.filter(
        (pl.col('NR_TURNO') == 2) &
        (pl.col('NR_VOTAVEL')).is_in([13,22])
    )



    # Memória CACHE é a memória de acesso rápido
    # Ativar o método StringCache, do polars
    # Isso é muito útil para filtos, cálculos em dados de larga escala
    with pl.StringCache():
        
        ''' VOTACAO '''

        # Selecionar nome do candidato (NM_VOTAVEL), a unidade federativa, que é a chave com o df bolsa familia (SG_UF) e quantidade de votos (QT_VOTOS)
        df_votacao_pl_lazy = df_votacao_pl.lazy().select(['SG_UF', 'NM_VOTAVEL', 'QT_VOTOS'])

        # Os tipos de dados categóricos são mais eficientes do que as string, 
        # pq o Polars cria um dicionario de indices numéricos quando o tipo de 
        # dado é categórico e processar um numero é bem mais facil do que um nome, o que otimiza o consumo de memória
        # CONVERTER DE STRING PARA CATEGORICO
        df_votacao_pl_lazy = df_votacao_pl_lazy.with_columns([
            pl.col('SG_UF').cast(pl.Categorical),
            pl.col('NM_VOTAVEL').cast(pl.Categorical)
        ])

        # AGRUPAR utilizando o plano de execução gerado pelo Lazy
        df_votacao_pl_uf = df_votacao_pl_lazy.group_by(['SG_UF', 'NM_VOTAVEL']).\
            agg(pl.col('QT_VOTOS').sum())

        # COLETAR DADOS DO LAZY
        df_votacao_pl_uf = df_votacao_pl_uf.collect()
  
        ''' BOLSA FAMILIA '''

        # Selecionar
        df_bf_pl_lazy = df_bf_pl.lazy().select(['UF', 'VALOR PARCELA'])

        # CONVERTER DE STRING PARA CATEGORICO
        df_bf_pl_lazy = df_bf_pl_lazy.with_columns([
            pl.col('UF').cast(pl.Categorical)
        ])

        # AGRUPAR utilizando o plano de execução gerado pelo Lazy
        df_bf_pl_uf = df_bf_pl_lazy.group_by('UF').\
            agg(pl.col('VALOR PARCELA').sum())

        # COLETAR DADOS DO LAZY
        df_bf_pl_uf = df_bf_pl_uf.collect()

        ''' JOIN DOS DATAFRAMES '''
        df_votos_bf = df_votacao_pl_uf.join(df_bf_pl_uf, left_on='SG_UF', right_on='UF')


    # Muda a identação para sair do StringCache
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_decimal_separator(',')
    pl.Config.set_thousands_separator('.')


    print(df_votacao_pl_uf)
    print(df_bf_pl_uf)
    print(df_votos_bf)

    #Fim da contagem de tempo
    end_time = time.time()

    #Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'DF VOTACAO E DO DF BOLSA FAM. AJUSTADOS. Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO AJUSTAR DF VOTACAO E DO DF BOLSA FAM.', e)
    exit()

###################################################################################
#----------------------------- CALCULAR CORRELAÇÃO --------------------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()

    # Dicionario candidato correlação
    dict_correlacoes = {}

    # Estrutura de repetição para separar os candidatos
    for candidato in df_votos_bf['NM_VOTAVEL'].unique():

        # Filtra dados pelo candidato
        df_candidato = df_votos_bf.filter(pl.col('NM_VOTAVEL') == candidato)

        # Arrays das variáveis quantitativa do candidato filtrado
        array_votos = np.array(df_candidato['QT_VOTOS'])
        array_valor_parcela = np.array(df_candidato['VALOR PARCELA'])

        # Calcular o coeficiente de correalação de Pearson (r)
        correlacao = np.corrcoef(array_votos, array_valor_parcela)[0,1]

        print(f'Correlação para {candidato}: {correlacao}')
        dict_correlacoes[candidato] = correlacao

    #Fim da contagem de tempo
    end_time = time.time()

    #Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'CORRELAÇÃO CALCULADA. Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO CALCULAR CORRELACAO', e)
    exit()





###################################################################################
#------------------------------ VISUALIZAR DADOS ----------------------------------
###################################################################################
    
try:

    # Início da contagem de tempo
    start_time = time.time()

    plt.subplots(2,2, figsize=(17,7))
    plt.suptitle('Votação x Bolsa Família', fontsize=16)

    # Posicao 1
    plt.subplot(2,2,1)
    plt.title('Lula')

    df_lula = df_votos_bf.filter(pl.col('NM_VOTAVEL') == 'LUIZ INÁCIO LULA DA SILVA')
    
    df_lula = df_lula.sort('QT_VOTOS', descending=True)

    plt.bar(df_lula['SG_UF'], df_lula['QT_VOTOS'])

    # Posicao 2
    plt.subplot(2,2,2)
    plt.title('Bozo')

    df_bozo = df_votos_bf.filter(pl.col('NM_VOTAVEL') == 'JAIR MESSIAS BOLSONARO')
    
    df_bozo = df_bozo.sort('QT_VOTOS', descending=True)

    plt.bar(df_bozo['SG_UF'], df_lula['QT_VOTOS'])

    # Posicao 3
    plt.subplot(2,2,3)
    plt.title('Valor Parcela')

    df_bf_pl_uf = df_bf_pl_uf.sort('VALOR PARCELA', descending=True)

    plt.bar(df_bf_pl_uf['UF'], df_bf_pl_uf['VALOR PARCELA'])

    # Posicao 4
    plt.subplot(2,2,4)
    plt.title('Correlações')

    # Coordenadas do plt.text
    x = 0.2
    y = 0.6

    for candidato, correlacao in dict_correlacoes.items():
        plt.text(x,y,f'{candidato}:{correlacao}', fontsize=12)

        # Reduzi 0.1 do eixo y
        #y = y - 0.1 se for somar y+=0.1
        y-=0.1
    plt.axis('off')
    plt.tight_layout()
        


    # Fim da contagem de tempo
    end_time = time.time()

    # Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'DADOS VISUALIZADOS. Tempo de execução {execution_time:.2f} segundos')
    
    
    plt.show()

except Exception as e:
    print('ERRO AO VISUALIZAR DADOS', e)
    exit()

# Fim da contagem de tempo
end_time_external = time.time()

# Calcular a diferença (tempo de execução)
execution_time_external = end_time_external - start_time_external
print(f"Tempo total de execução: {round(execution_time_external,2)} segundos")