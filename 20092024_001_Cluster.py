''' Dentro das possíveis fraudes que você identificou na região sudeste, 
a média dessas parcelas está vinculada a cidades com mais ou menos eleitores
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
    df_bf_anomalias_sudeste_pl = pl.read_parquet(ENDERECO_DADOS + 'dados_bolsa_familia_anomalias_sudeste.parquet')  

    ENDERECO_DADOS = r'./Dados/camada_raw/'

    df_votos_pl = pl.read_csv(ENDERECO_DADOS + 'votacao_secao_2022_BR.csv', separator=';', encoding='LATIN-1')  
    
    # Fim da contagem de tempo
    end_time = time.time()

    # Calcular a diferença (tempo de execução)
    execution_time = end_time - start_time

    print(f'DADOS OBTIDOS! Tempo de execução {execution_time:.2f} segundos')

except Exception as e:
    print('ERRO AO OBTER DADOS', e)
    exit()

###################################################################################
#----------------------------- TRATAR E AGRUPAR DFs -------------------------------
###################################################################################
    
try:


    df_bf_anomalias_sudeste_pl_lazy = df_bf_anomalias_sudeste_pl.lazy().select(['NOME MUNICÍPIO', 'VALOR PARCELA'])

    df_bf_anomalias_sudeste_pl = df_bf_anomalias_sudeste_pl_lazy.group_by(['NOME MUNICÍPIO']).agg(pl.col(['VALOR PARCELA']).mean())

    df_bf_anomalias_sudeste_pl = df_bf_anomalias_sudeste_pl.collect()

    df_votos_pl_lazy = df_votos_pl.lazy().select(['SG_UF', 'NM_MUNICIPIO', 'QT_VOTOS'])\
        .filter(pl.col('SG_UF').is_in(['RJ', 'ES', 'SP', 'MG']))

    df_votos_uf_pl = df_votos_pl_lazy.group_by(['NM_MUNICIPIO']).agg(pl.col('QT_VOTOS').sum())

    df_votos_uf_pl = df_votos_uf_pl.collect()


    df_anomalias_votos_pl = df_bf_anomalias_sudeste_pl.join(df_votos_uf_pl, left_on='NOME MUNICÍPIO', right_on='NM_MUNICIPIO')
    
    print(df_anomalias_votos_pl.head())

except Exception as e:
    print('ERRO AO TRATAR E AGRUPAR DFs', e)
    exit()

###################################################################################
#------------------------ RODAR MÉTODO DE COTOVELO (ELBOW) ------------------------
###################################################################################
    
try:
    
    from sklearn.preprocessing import StandardScaler

    #normalizar
    scaler = StandardScaler()

    array_valor_parcela = np.array(df_anomalias_votos_pl.select(['VALOR PARCELA']))
    array_qt_votos = np.array(df_anomalias_votos_pl.select(['QT_VOTOS']))

    # Transforma os arrays em dados bidimensionais
    X = np.column_stack([array_valor_parcela, array_qt_votos])

    # Normalizar z
    array_normalizado = scaler.fit_transform(X)

except Exception as e:
    print('ERRO AO RODAR MÉTODO DE COTOVELO (ELBOW)', e)
    exit()

###################################################################################
#------- RODAR MÉTODO DE CLUSTER (K-MEANS) E VISUALIZAR O GRÁFICO DO COTOVELO------
###################################################################################
    
try:
    
    from sklearn.cluster import KMeans

    # Inércia
    # Soma das distâncias quadradas de cada ponto
    # para o centróide mais próximo
    inercia = []

    #qtde cluster
    #inicializar
    valores_k = range(1,10)

    #aplicar o método cotovelo
    for k in valores_k:
        # iniciar o modelo de k-means com k clusteres
        kmeans = KMeans(n_clusters=k, random_state=42)

        # Treina o modelo com os dados normalizados
        kmeans.fit(array_normalizado)

        # Adicionar a inérciaà lista de inercias
        inercia.append(kmeans.inertia_)

    plt.plot(valores_k,inercia) 
    plt.title("Método de cotovelo (elbow)")   
    plt.xlabel("Número de clusters (K)")
    plt.ylabel('Inércia')
    #plt.show()


except Exception as e:
    print('ERRO AO RODAR MÉTODO DE CLUSTER (K-MEANS) E VISUALIZAR O GRÁFICO DO COTOVELO', e)
    exit()

###################################################################################
#----------------------------- CLUSTERIZAR DADOS DO DF ----------------------------
###################################################################################
    
try:
    
    # Instanciar o modelo de clusters, com 4 grupos
    # após observar o método de cotovelo
    kmeans = KMeans(n_clusters=4, random_state=42)

    # Trenar o modelo
    kmeans.fit(array_normalizado)

    # O modelo já definiu os clusters
    # adicionar ao DF
    df_anomalias_votos_pl = df_anomalias_votos_pl.with_columns([pl.Series(name='cluster', values=kmeans.labels_)])

    print(df_anomalias_votos_pl['cluster'].n_unique())

except Exception as e:
    print('ERRO AO CLUSTERIZAR DADOS DO DF', e)
    exit()

###################################################################################
#----------------------------- VISUALIZAR CLUSTERS --------------------------------
###################################################################################
    
try:
    
    plt.subplots(1,1, figsize=(15,5))
    plt.suptitle('Clusterização de valor parcela BF x Qtd de votos, por municípios do sudeste', fontsize=14)

    plt.subplot(1,1,1)
    array_valor_parcela = np.array(df_anomalias_votos_pl.select('VALOR PARCELA'))
    array_qt_votos = np.array(df_anomalias_votos_pl.select('QT_VOTOS'))
    array_cluster = np.array(df_anomalias_votos_pl.select('cluster'))

    scatter = plt.scatter(array_valor_parcela,array_qt_votos, c=array_cluster, cmap='rainbow') #cmap = color map que são as cores que serão atribuídas aos clusters
    
    plt.title('Clusters de valor parcela x qtd de votos')
    plt.ylabel('qtd votos')
    plt.xlabel('valor parcela')

    # adicionar barra de cores
    cbar = plt.colorbar(scatter)
    #cbar.set_ticks(array_cluster.reshape(-1))
    cbar.set_ticks(np.unique(array_cluster))
    
    cbar.set_label('Cluster')
    plt.ticklabel_format(style='plain', axis='both')
    plt.tight_layout()
    plt.show()


except Exception as e:
    print('ERRO AO VISUALIZAR CLUSTERS', e)
    exit()

# Fim da contagem de tempo
end_time_external = time.time()


# Calcular a diferença (tempo de execução)
execution_time_external = end_time_external - start_time_external
print(f"Tempo total de execução: {round(execution_time_external,2)} segundos")