import math
from statistics import variance

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import datetime as dt

class kMeansIndexer:
    def calculate(T):
        N = len(T)
        s = 1
        M = (int)(s * math.sqrt(N))

        # PIOR DESVIO
        # FABRICA O DENOMINADOR, SENDO O PIOR CASO COMO DENOMIINADOR
        # exemplo array pior caso arcene
        # variance_pior = variance([900, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0])

        array_pior = [0] * M # array de quantidade de instancias por grupo do pior caso
        array_pior[0] = N

        # calcula variancia do array pior
        variance_pior = variance(array_pior)
        print("PIOR VARIANCIA")
        print(variance_pior)
        # desvio padrao do pior caso
        dp_pior = math.sqrt(variance_pior)

        #

        kmeans = KMeans(n_clusters=M, init='k-means++', max_iter=500, n_init=10).fit(T)
        # kmeans = KMeans(n_clusters=M, init='random', max_iter=500, n_init=10).fit(T)
        cluster_labels = kmeans.labels_  # ARMAZENA O INDICE DO AGRUPAMENTO PARA CADA INDICE DE INSTANCIAS
        cluster_centers = kmeans.cluster_centers_  # ARMAZENA CADA UM DOS CENTROS (X, Y, ..) PARA CADA agrupamento
        print(cluster_labels)
        cj = cluster_centers

        clusters_counter = []
        total_instances = 0
        for i in range(M):
            instances = np.where(cluster_labels == i)
            instance_counter = 0
            for p in instances:
                instance_counter = instance_counter + len(p)
                total_instances = total_instances + len(p)

            clusters_counter.append(instance_counter)



        print("PIOR ARRAY")
        print(array_pior)
        print("PIOR VARIANCIA")
        print(variance_pior)
        print("PIOR DP")
        print(dp_pior)
        # variance calc
        variance_calc = variance(clusters_counter)
        print("VARIANCE")
        print(variance_calc)
        dp = math.sqrt(variance_calc)
        print("DESV. PADRAO")
        print(dp)
        index = dp / dp_pior
        print("INDICE")
        print(index)