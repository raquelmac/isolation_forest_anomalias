'''**Exercício 11:**
Faça um estudo acerca dos valores de parcelas do bolsa família, 
para identificar possíveis fraudes, somente na região Sudeste
Apresente informações que possam auxiliar a investigação, como a 
distribuição dos valores de possíveis fraudes e dados dessas 
pessoas.
Seja o mais rigoroso e criterioso possível na detecção dessas 
possíveis frudes.
Dúvidas? Fale comigo
'''

import time


# Início da contagem de tempo
start_time_external = time.time()

###################################################################################
#---------------------------- IMPORTAR AS BIBLIOTECAS  ----------------------------
###################################################################################

try:

    # Início da contagem de tempo
    start_time = time.time()

    from auxiliar.conexoes import obter_dados_pl#, obter_dados_mysql
    #importa a classe de detecção de anomalias: isolation forest
    from sklearn.ensemble import IsolationForest
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt

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
    df_bf_pl = pl.read_parquet(ENDERECO_DADOS + 'dados_bolsa_familia_abr_mai.parquet')  

    # Fim da contagem de tempo
    end_time = time.time()

    # Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'DADOS OBTIDOS! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO OBTER DADOS', e)
    exit()

###################################################################################
#-------------------- DETECTAR ANOMALIAS COM ISOLATION FOREST ---------------------
###################################################################################
    
try:

    with pl.StringCache(): # Acelerando os trabalhos Trabalhar com processamento em larga escala

        array_valor_parcela = np.array(df_bf_pl.select('VALOR PARCELA'))

        # definir o modelo do isolation forest (em portugues, é floresta de isolamento)
        # É um algoritmo de detecção de anomalias (fraude)
        # a detecção dessas anomalias é realizada a partir de uma floresta de arvores (subconjunto de dados)
        # cada árvore é treinada com um subconjunto, na qual cada dado é comparado com os demais.
        # se um dado é muito diferente dos demais, ou seja, fora do padrão do seu subconjunto
        # ele é considerado uma anomalua ou, popularmente, uma possível fraude
        # Cuidado para nao inferir fraude nos dados anomalos. Estar fora do padrao
        # quer dizer que precisa ser investigado! Antes dessa investigação, NÃO
        # é possível afirmar nada
        modelo = IsolationForest(
                                contamination=0.005, # qto menor o percentual, mais criterioso é a detecção
                                random_state=42,
                                n_estimators=50, #número de árvores (subconjuntos) qnto mais árvores mais precisa é a detecção, porém leva mais tempo de processamento
                                n_jobs=-1 #quantidade de núcleos do processador a serem utilizados (-1 - roda com todos os núcleos)
                                # Botão direito na barra de tarefas >> gerenciador de tarefas >> desempenho >> cpu >> nucleos 
                                #,shuffle=False (escolha fixa e nao precisa usar o random_state)
                                )

        # Aplicar o isolation forest no array valor parcela
        anomalias = modelo.fit_predict(array_valor_parcela.reshape(-1,1)) # reshape(todas as linhas, uma coluna) lembrando que pra ser tudo no python é sempre -1

        # lazy evaluation - plano de execução, para otimizar a performance da execução
        df_bf_pl_lazy = df_bf_pl.lazy()

        # adicionar as anomalias no df_bf_pl
        df_bf_pl = df_bf_pl_lazy.with_columns(
            [pl.Series(name='anomalia', values=anomalias)]
        )

        # Coletar o dataframe executado no lazy evaluation
        df_bf_pl = df_bf_pl.collect()

        # lazy evaluation - plano de execução, para otimizar a performance da execução
        df_bf_pl_lazy = df_bf_pl.lazy()

        # Filtra somente as anomalias. O modelo retorna -1 para os anômalos e 1 para os normais
        df_bf_pl = df_bf_pl_lazy.filter(
            pl.col('anomalia') == -1,
            ((pl.col('UF') == 'RJ') | (pl.col('UF') == 'ES') | (pl.col('UF') == 'SP') | (pl.col('UF') == 'MG'))
            ) 

        # Coletar o dataframe executado no lazy evaluation
        df_bf_pl_anomalias = df_bf_pl.collect()

    print('\nANOMALIAS IDENTIFICADAS:')
    print(len(df_bf_pl_anomalias))
    print(df_bf_pl_anomalias.head(20))


except Exception as e:
    print('ERRO AO DETECTAR ANOMALIAS', e)
    exit()


###################################################################################
#----------------------------- VISUALIZAR DADOS -----------------------------------
##############################################################################b####btry:
try:
    plt.subplots(2,2, figsize=(17,7))
    plt.suptitle('Detecção de anomalias no valor parcela - Sudeste')

    array_valor_parcela_anomalias = np.array(df_bf_pl_anomalias.select('VALOR PARCELA'))

    # Posição 1:
    plt.subplot(2,2,1)
    plt.boxplot(array_valor_parcela_anomalias, vert=False, showmeans=True)
    plt.title('Distribuição das parcelas anômalas')

    
    # Posição 2:
    plt.subplot(2,2,2)
    plt.hist(array_valor_parcela_anomalias, bins=100, edgecolor='black')
    plt.title('Histograma das parcelas anômalas')

    # Posição 3: TOP 10 maiores
    plt.subplot(2,2,3)
    df_anomalias_detectadas_maiores = df_bf_pl_anomalias.sort('VALOR PARCELA', descending=True).head(10)

    colunas = ['NOME FAVORECIDO', 'MÊS COMPETÊNCIA', 'MÊS REFERÊNCIA', 'UF', 'NOME MUNICÍPIO', 'VALOR PARCELA']

    x = 0.1
    y = 0.9

    for nome, mes_competencia, mes_referencia, uf, municipio, valor_parcela \
        in df_anomalias_detectadas_maiores[colunas].to_pandas().values:
        plt.text(x,y,f'{nome}-{mes_competencia}-{mes_referencia}-{uf}-{municipio}-R$ {valor_parcela:.2f}', fontsize=8)
        y-=0.1 #Diminuindo o valor de y para descer a linha no print
        
    plt.axis('off')

    # Posição 4: TOP 10 menores
    plt.subplot(2,2,4)
    df_anomalias_detectadas_menores = df_bf_pl_anomalias.sort('VALOR PARCELA', descending=False).head(10)

    x = 0.1
    y = 0.9

    for nome, mes_competencia, mes_referencia, uf, municipio, valor_parcela \
        in df_anomalias_detectadas_menores[colunas].to_pandas().values:
        plt.text(x,y,f'{nome}-{mes_competencia}-{mes_referencia}-{uf}-{municipio}-R$ {valor_parcela:.2f}', fontsize=8)
        y-=0.1 #Diminuindo o valor de y para descer a linha no print
        
    plt.axis('off')

    plt.tight_layout()

    plt.show()

except Exception as e:
    print('ERRO AO VISUALIZAR DADOS', e)
    exit()

###################################################################################
#------------------------------- EXPORTAR DADOS -----------------------------------
###################################################################################
try:

   
    print(df_bf_pl_anomalias.head(5))
    # Diretório onde os arquivos serão salvos
    ENDERECO_DADOS = r'./Dados/camada_bronze/'
    
    # Carrega na camada Bronze (Arquitetura Medallion) os arquivos da camada raw já processados 
    df_bf_pl_anomalias.write_parquet(ENDERECO_DADOS + 'dados_bolsa_familia_anomalias_sudeste.parquet')
    
    print('aqui')

except Exception as e:
    print('ERRO AO EXPORTAR DADOS', e)
    exit()

# Fim da contagem de tempo
end_time_external = time.time()


# Calcular a diferença (tempo de execução)
execution_time_external = end_time_external - start_time_external
print(f"Tempo total de execução: {round(execution_time_external,2)} segundos")