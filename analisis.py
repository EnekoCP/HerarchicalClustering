import sys
from time import time
import numpy as np
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import pickle


def pause():
    programPause = input("Press the <ENTER> key to continue...")


def menu():
    correcto = False
    num = 0
    while (not correcto):
        try:
            num = int(input("Introduce un numero entero: "))
            correcto = True
        except ValueError:
            print('Error, introduce un numero entero')

    return num


def display_topics(H, W, feature_names, documents, no_topics, no_top_words, no_top_documents):
    lista = []

    for topic_idx, topic in enumerate(H):
        lista.append("Topic %d:" % (topic_idx))
        ##print("Topic %d:" % (topic_idx))    
        ##print(" ".join([feature_names[i]
        ##for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:len(documents)]
        count = 0
        for doc_index in top_doc_indices:
            if ((W[doc_index][0]) > 0.8):
                ##print (" ")
                ##print("** Probabilities: **")  
                ##print (W[doc_index][0])
                lista.append((W[doc_index][0], doc_index))
                ##print (" ")
                ##print(documents[doc_index])
                count = count + 1
        ##print(count)
        lista.append("\n")

    with open("topics.csv", 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lista)

    return lista

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def topic_modeling(df, topicosToF):
    ## TOPIC MODELING Representacion 4
   

    no_topics = 10
    no_top_words = 30
    no_top_documents = 10
    documents = df
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.50, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print(tf)
    print("-------------------")
    print(tf_feature_names)
    # Run LDA
    lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=100, learning_method='online',
                                          learning_offset=50., random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_
    


    pickle.dump(tf_vectorizer,open('vectorizer.pk', 'wb'))
    with open('lda_model.pk', 'wb') as lda_model0:
        pickle.dump(lda_model, lda_model0)

    print("MODELO GUARDADO")
    print("----------------")
    print("LDA Topics")

    if topicosToF == True:
        topicModeling = []
        topicModeling = display_topics(lda_H, lda_W, tf_feature_names, documents, no_topics, no_top_words,
                                       no_top_documents)
        print()
        print("-------------")
        print(topicModeling)

    print("-------------")

    return lda_W


def preprocessData(path):
    dfMergedfMeta = []

    dfMergedfMeta = pd.read_csv(path, low_memory=False)

    print("DATASET CARGADO")
    pause()

    # SACAMOS LOS 10 PRIMEROS PARA COMPROBAR QUE FUNCIONA
    print(dfMergedfMeta[:10])

    print("SACAMOS LOS 10 PRIMEROS PARA COMPROBAR QUE FUNCIONA")
    pause()
    # INDICES
    print(dfMergedfMeta.keys())

    print("INDICES")
    pause()

    # ANALISIS DEL CONJUNTO DE DATOS
    print("ANALISIS DE LOS DATOS")
    print(dfMergedfMeta.info())  # Numero y Tipo de atributos
    print("---------------------------------------------------")
    print("Numero y Tipo de atributos")
    pause()

    print(dfMergedfMeta.describe())  # Min,Max,Media,Desviacion ...

    print("---------------------------------------------------")
    print("Min,Max,Media,Desviacion ...")
    pause()

    print(dfMergedfMeta.duplicated().describe())  # Instacias repetidas
    dfMergedfMeta.drop_duplicates()
    print("---------------------------------------------------")
    print("Instancias repetidas")
    pause()

    print("MISSING VALUES")
    print(dfMergedfMeta.isnull().sum())  # Missing values
    dfMergedfMeta.dropna().reset_index(drop=True)
    print(dfMergedfMeta.isnull().sum())

    print("Missing values")
    pause()

    # FILTROS ENCODING/TOKENIZATION/RE-CASING

    from nltk import word_tokenize
    from nltk.stem.porter import PorterStemmer
    import re
    from nltk.corpus import stopwords
    import csv
    import numpy

    df = dfMergedfMeta['reviewText']
    df = df[:600]

    vocabComplete = []
    sentences = []

    i = 0
    cachedStopWords = stopwords.words('english')

    for reg in df:
        ##print("---------------------------------------------------")
        min_length = 4
        words = word_tokenize(str(reg));

        words = [word for word in words
                 if word not in cachedStopWords]
        words = [word for word in words if word.isalpha()]
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)));

        p = re.compile("[a-z]");

        filtered_tokens = list(filter(lambda token:
                                      p.match(token) and len(token) >= min_length, tokens));

        vocabComplete = vocabComplete + filtered_tokens

        sentences.append(filtered_tokens)  ##metemos las instancias limpiadas en una lista para el BoW

    print(vocabComplete)
    print("VOCABULARIO generado")
    pause()
    print(sentences)
    print("INSTANCIAS LIMPIAS")
    pause()
    ##Palabaras mas frecuentes

    print("PALABRAS MAS FRECUENTES")
    word_freq = defaultdict(int)
    for word in vocabComplete:
        word_freq[word] += 1
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])  ## PALABRAS MAS REPETIDAS
    pause()

    print("-------PREPROCESO  ACABADO -------")
    print(" ")
    print(" ")
    ###############################################################
    print("-------REPRESENTACION -------")

    salir = False
    opcion = 0

    while not salir:

        print("1. Representacion BoW")
        print("2. Vocabulario con Indices")
        print("3. Word2Vec Word Embedding")
        print("4. Topic Modeling")
        print("5. Continuar Clustering")
        print(" ")

        print("Elige una opcion")

        opcion = menu()

        if opcion == 1:
            vector2 = []
            ##print(sentences)
            sen = 0
            # Se recorren las intsancias y por cada palabra de la instancia (limpiada anteriormente) se mira cuntas veces aparece
            # en el vocabulario
            cont = 0
            aux = False
            while (sen < len(sentences)) and (aux == False):

                if cont < 3:
                    pause()
                if cont == 3:
                    aux = True

                words = sentences[sen]
                bag_vector = numpy.zeros(len(vocabComplete))
                for w in words:
                    for i, word in enumerate(vocabComplete):
                        if word == w:
                            bag_vector[i] += 1
                sen = sen + 1
                print("{0}\n{1}\n".format(words, numpy.array(bag_vector)))
                vector2.append(numpy.array(bag_vector))
                ##print(vector2)
                cont = cont + 1

        elif opcion == 2:
            vocab, index = {}, 1  # start indexing from 1
            vocab['<pad>'] = 0  # add a padding token
            for token in vocabComplete:
                if token not in vocab:
                    vocab[token] = index
                    index += 1
            vocab_size = len(vocab)
            print(vocab)
            print("---------")
            print(vocab_size)
            print("---------")
            inverse_vocab = {index: token for token, index in vocab.items()}
            print(inverse_vocab)

            vector = []
            for sentence in sentences:
                vectorAux = []
                for word in sentence:
                    vectorAux.append(vocab[word])
                vector.append(vectorAux)

            print("-------------------")
            print(vector)
            print(len(vector))

        elif opcion == 3:
            from gensim.models import Word2Vec
            import multiprocessing

            cores = multiprocessing.cpu_count()
            w2v_model = Word2Vec(min_count=50,
                                 window=2,
                                 vector_size=300,
                                 sample=6e-5,
                                 alpha=0.03,
                                 min_alpha=0.0007,
                                 negative=20,
                                 workers=cores - 1)  # Creando el modelo
            print("MODELO W2V CREADO")
            pause()

            w2v_model.build_vocab(sentences, progress_per=10000)  ##Creando el vocabulario

            print("vocabulario del modelo CREADO")
            pause()

            print()
            print(w2v_model)

            print("Entrenamos el modelo")
            pause()

            t = time()
            w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)  ##Training
            print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
            pause()

            print("---------------------------------------------------------------------")
            print(w2v_model.wv.most_similar(positive=["intuit"]))  ## Palabras relacionados con ...

            print("Palabras relacionadas con INTUIT")
            pause()

            print("---------------------------------------------------------------------")

            print(w2v_model.wv.similarity('intuit', 'program'))  ##Porcentaje de similarity

            print("Porcentaje de similitud entre intuit y program")
            pause()

            print("---------------------------------------------------------------------")

            print(w2v_model.wv.doesnt_match(['intuit', 'tell', 'data']))  # Cual sobra

            print("Palabra con menos relacion entre INTUIT, TELL , DATA")
            pause()
            print("---------------------------------------------------------------------")

            ##arrays = np.empty((0, 300), dtype='f')
            # adds the vector of the query word
            ##arrays = np.append(arrays,w2v_model.wv.__getitem__("tax"), axis=0)

            print("------ARRAY WORD EMBEDDING-------")
            ##print(arrays)

            from sklearn.decomposition import PCA
            from matplotlib import pyplot
            ##REPRESENTACION 1

            words = list(w2v_model.wv.index_to_key)
            ##print(words)
            X = w2v_model.wv.__getitem__(w2v_model.wv.index_to_key)  ##VERSION DESACTUALIZADA
            print(X)
            pca = PCA(n_components=2)
            result = pca.fit_transform(X)
            # create a scatter plot of the projection
            pyplot.scatter(result[:, 0], result[:, 1])

            for i, word in enumerate(words):
                pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

            pyplot.show()

            print("REPRESENTACION 1 - PCA ESPACIO DE PALABRAS")
            pause()

            ##REPRESENTACION 2

            import numpy as np

            import seaborn as sns

            sns.set_style("darkgrid")

            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            def tsnescatterplot(model, word, list_names):
                """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
                its list of most similar words, and a list of words.
                """
                arrays = np.empty((0, 300), dtype='f')
                word_labels = [word]
                color_list = ['red']

                # adds the vector of the query word
                arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

                # gets list of most similar words
                close_words = model.wv.most_similar([word])

                # adds the vector for each of the closest words to the array
                for wrd_score in close_words:
                    wrd_vector = model.wv.__getitem__([wrd_score[0]])
                    word_labels.append(wrd_score[0])
                    color_list.append('blue')
                    arrays = np.append(arrays, wrd_vector, axis=0)

                # adds the vector for each of the words from list_names to the array
                for wrd in list_names:
                    wrd_vector = model.wv.__getitem__([wrd])
                    word_labels.append(wrd)
                    color_list.append('green')
                    arrays = np.append(arrays, wrd_vector, axis=0)

                # Reduces the dimensionality from 300 to 50 dimensions with PCA
                reduc = PCA(n_components=2).fit_transform(arrays)

                # Finds t-SNE coordinates for 2 dimensions
                np.set_printoptions(suppress=True)

                Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

                # Sets everything up to plot
                df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                                   'y': [y for y in Y[:, 1]],
                                   'words': word_labels,
                                   'color': color_list})

                fig, _ = plt.subplots()
                fig.set_size_inches(9, 9)

                # Basic plot
                p1 = sns.regplot(data=df,
                                 x="x",
                                 y="y",
                                 fit_reg=False,
                                 marker="o",
                                 scatter_kws={'s': 40,
                                              'facecolors': df['color']
                                              }
                                 )

                # Adds annotations one by one with a loop
                for line in range(0, df.shape[0]):
                    p1.text(df["x"][line],
                            df['y'][line],
                            '  ' + df["words"][line].title(),
                            horizontalalignment='left',
                            verticalalignment='bottom', size='medium',
                            color=df['color'][line],
                            weight='normal'
                            ).set_size(15)

                plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
                plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

                plt.title('t-SNE visualization for {}'.format(word.title()))

            ##To make the visualizations more relevant, we will look at the relationships between a query word (in **red**),
            ##its most similar words in the model (in **blue**), and other words from the vocabulary (in **green**).

            # 10 Most similar words vs. 8 Random words

            ##print(tsnescatterplot(w2v_model, 'intuit', ['option', 'offer', 'help']))

            # 10 Most similar words vs. 10 Most dissimilar

            ##print(tsnescatterplot(w2v_model, 'intuit', [i[0] for i in w2v_model.wv.most_similar(negative=["intuit"])]))

            ##REPRESENTACION 3
            analizeWord = 'intuit'
            test = [i[0] for i in w2v_model.wv.most_similar(positive=[analizeWord])]
            test.append(analizeWord)
            words = list(test)
            print("-----------")
            print(words)
            X = w2v_model.wv.__getitem__(test)
            print(X)
            ##print(len(X))
            pca = PCA(n_components=2)
            result = pca.fit_transform(X)
            # create a scatter plot of the projection
            pyplot.scatter(result[:, 0], result[:, 1])
            for i, word in enumerate(words):
                pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

            pyplot.show()

            print("REPRESENTACION 2 - PALABRAS CON MAS SILIMITUD CON 'intuit' GRAFICAMENTE")
            pause()

            # 10 Most similar words vs. 8 Random words

            ##print(tsnescatterplot(w2v_model, 'tax', ['option', 'offer', 'help']))

            # 10 Most similar words vs. 10 Most dissimilar

            ##print(tsnescatterplot(w2v_model, 'tax', [i[0] for i in w2v_model.wv.most_similar(negative=["tax"])]))

        elif opcion == 4:
            topic_modeling(df, True)
        elif opcion == 5:
            salir = True
        else:
            print("Introduce un numero entre 1 y 5")
    ############################################################

    topicModeling = topic_modeling(df, False)
    return [topicModeling, df]


##CLUSTERING


class Instance():
    def __init__(self, id, attributes):
        self.id = id
        self.attributes = attributes

    def getAttrVector(self):
        return self.attributes

    def getId(self):
        return self.id


class Cluster():

    def __init__(self, longV):
        self.averageVector = [0] * longV
        self.instaces = []

    def addIntances(self, instance):

        self.instaces.append(instance)
        iArr = instance.getAttrVector()
        for i in range(0, len(self.averageVector)):
            self.averageVector[i] += iArr[i]

    def getCentroid(self):
        numIns = len(self.instaces)
        centroid = self.averageVector
        cent = [0] * len(centroid)
        for j in range(0, len(self.averageVector)):
            cent[j] = centroid[j] / numIns
        return cent

    def getInstances(self):
        return self.instaces


def mergeClusters(c1, c2):
    c1i = c1.getInstances()
    c2i = c2.getInstances()
    c3 = Cluster(len(c1i[0].getAttrVector()))

    for i in range(0, len(c1i)):
        c3.addIntances(c1i[i])
    for j in range(0, len(c2i)):
        c3.addIntances(c2i[j])

    return c3


# Clase instancia que guarda el numero de instancia y el vector de atributos


def clusterDistance(c1, c2):
    "Average-Link"
    cc1 = c1.getCentroid()
    cc2 = c2.getCentroid()
    return distance.minkowski(cc1, cc2, 1)


def flatVector(vec):
    flat_list = []
    original_list = vec
    for l in original_list:
        for item in l:
            flat_list.append(item)

    return flat_list


def findMin(distanceMatrix):
    min = distanceMatrix[0, 1]
    posI = 0
    posJ = 1
    for i in range(0, len(distanceMatrix)):
        for j in range(i + 1, len(distanceMatrix)):
            if min > distanceMatrix[i, j]:
                min = distanceMatrix[i, j]
                posI = i
                posJ = j

    return [posI, posJ]


def createDistanceMatrix(clustersArray):
    # Creamos una matrix vacia de NxN
    dMatrix = np.zeros([len(clustersArray), len(clustersArray)])
    # Solo vamos a calcular el triangulo derecho superior para no duplicar calculos
    for i in range(0, len(clustersArray)):
        for j in range(i + 1, len(clustersArray)):
            dMatrix[i, j] = clusterDistance(clustersArray[i], clustersArray[j])

    return dMatrix


def deleteClusters(distanceMatrix, i, j):
    if i > j:
        distanceMatrix = np.delete(distanceMatrix, i, 0)
        distanceMatrix = np.delete(distanceMatrix, i, 1)
        distanceMatrix = np.delete(distanceMatrix, j, 0)
        distanceMatrix = np.delete(distanceMatrix, j, 1)
    else:
        distanceMatrix = np.delete(distanceMatrix, j, 0)
        distanceMatrix = np.delete(distanceMatrix, j, 1)
        distanceMatrix = np.delete(distanceMatrix, i, 0)
        distanceMatrix = np.delete(distanceMatrix, i, 1)

    return distanceMatrix


def printableCluster(clustersArray):
    vec = " ;"
    for i in range(0, len(clustersArray)):
        ins = clustersArray[i].getInstances()
        vec += "{"
        for j in range(0, len(ins)):
            if j != 0: vec += ","
            vec += str(ins[j].getId())
        vec += "};"

    return vec


def clustersToArray(nInstances, clustersArray):
    # Array con el nº del cluster al que pertenece i instancia
    rst = np.zeros(nInstances, dtype=int)
    for i in range(0, len(clustersArray)):
        instArr = clustersArray[i].getInstances()

        for j in range(0, len(instArr)):
            rst[instArr[j].getId()] = i

    return rst


def HierarchicalClustering(n_clusters, instances):
    path = "Info" + str(n_clusters) + ".txt"
    f = open("Info.txt", "w")
    clusterArray = []
    nInstances = len(instances)
    # Añadimos las instancias a un cluster cada una
    objs = list()
    for x in range(0, nInstances):
        # f.write("Añadiendo Instancia: "+str(x))
        inst = Instance(x, instances[x])
        c = Cluster(len(instances[x]))
        c.addIntances(inst)

        clusterArray.append(c)

    # Calculamos las matriz de distancia entre clusters
    distMatrix = createDistanceMatrix(clusterArray)

    itr = 0
    # Empieza a iterar hasta que obtengamos el numero de clusters deseados
    while len(clusterArray) > n_clusters:

        itr += 1

        # Usamos la matriz de distancias para calcular los 2 clusters mas proximos
        minPos = findMin(distMatrix)
        dMik = distMatrix[minPos[0], minPos[1]]
        c1 = clusterArray[minPos[0]]
        c2 = clusterArray[minPos[1]]
        # Creamos el cluster c1 U c2
        c3 = mergeClusters(c1, c2)
        # Ahora tenemos que eliminar c1 y c2 de clusterArray y de la distanceMatrix

        del clusterArray[minPos[1]]
        del clusterArray[minPos[0]]
        distMatrix = deleteClusters(distMatrix, minPos[0], minPos[1])

        # Añadimos el cluster a clusterArray
        clusterArray.append(c3)

        # Por ultimo añadimos el nuevo cluster a distanceMatrix y calculamos las distancias de ese nuevo cluster
        # Insertar fila
        distMatrix = np.insert(distMatrix, distMatrix.shape[0], np.zeros(len(distMatrix)), 0)
        # Insertar columna
        distMatrix = np.insert(distMatrix, distMatrix.shape[1], np.zeros(len(distMatrix)), 1)

        # Calcular las nuevas distancias y las añadimos

        for i in range(0, len(distMatrix)):
            distMatrix[i, len(distMatrix) - 1] = clusterDistance(clusterArray[i], c3)
        # Escribimo en el fichero de ClusteringInfo
        evaluation = evaluate(clusterArray)
        f.write("Iteration= " + str(itr) + ", NumClusters: " + str(
            len(clusterArray)) + ", Silhouette: " + str(evaluation[0]) + ", Davies-Bouldin index: " + str(evaluation[1]) + ", DisMikwoski used in Iteration: " + str(
            dMik) + printableCluster(clusterArray))
        f.write("\n")

    f.close()
    rst = clusterArray

    return rst

def plotClustering(X,centroids,lda_w):
    labels = X
    # Getting the cluster centers
    C = centroids
    import matplotlib.pyplot as plt

    #configuro el tamaño del grafico final
    plt.figure(figsize=(10,7))

    #scatter del primer cluster
    plt.scatter(
        lda_w[X == 0, 0], lda_w[X == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )
    #scatter del segundo cluster
    plt.scatter(
        lda_w[X == 1, 0], lda_w[X == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )
    #scatter del tercer cluster
    plt.scatter(
        lda_w[X == 2, 0], lda_w[X == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )
    #scatter del los centroides
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroides'
    )
    #le pongo la leyenda
    plt.legend(scatterpoints=1)
    #hace una grilla en el grafico
    plt.grid()
    #lo imprime en pantalla
    plt.show()


    ##plt.figure(figsize=(10, 7))
    ##plt.plot(instances[:, 0], instances[:, 1], c=clusterArray, cmap='rainbow')
    ##sb.pairplot(dataframe.dropna(), hue='categoria', size=4, vars=["op", "ex", "ag"], kind='scatter')
    ##plt.plot(X[clusterArray == 0, 0], X[clusterArray == 0, 1], 'r.', label='cluster 1')
    ##plt.plot(X[clusterArray == 1, 0], X[clusterArray == 1, 1], 'b.', label='cluster 2')
    ##plt.plot(X[clusterArray == 2, 0], X[clusterArray == 2, 1], 'g.', label='cluster 3')
    ##plt.plot(centroids[:, 0], centroids[:, 1], 'mo', markersize=8, label='centroides')
    ##plt.show()


def saveModel(textsArray, attrArray, clusterLabels, nc):
    path = "Model" + str(nc) + ".csv"
    f = open(path, "w")
    f.write("Text;;;;Attributes;;;;Cluster")
    f.write("\n")
    for i in range(0, len(textsArray)):
        a = ""
        a += textsArray[i].replace("\n"," ")
        a += ";;;;["
        for j in range(0, len(attrArray[i])):
            a += str(attrArray[i][j])
            if j < len(attrArray[i]) - 1:
                a += ","

        a += "];;;;"
        a += str(clusterLabels[i])
        f.write(a)
        f.write("\n")
    f.close()
    return True


def loadModel(modelPath):
    filename = modelPath
    data = pd.read_csv(filename, header=0, delimiter=";;;;", engine='python')
    text = data['Text']
    attributesS = data['Attributes']
    cluster = data['Cluster']
    attributes = []

    # Attributes to float array
    for p in attributesS:
        p = p.replace("[", "")
        p = p.replace("]", "")
        p = list(map(float, p.split(",")))
        attributes.append(p)

    # Crear clusters
    distinct = []
    for x in cluster:
        if x not in distinct:
            distinct.append(x)

    clusterArray = []

    for m in range(0, len(distinct)):
        c = Cluster(len(attributes[m]))
        clusterArray.append(c)

    for i in range(0, len(attributes)):
        inst = Instance(i, attributes[i])
        clusterArray[cluster[i]].addIntances(inst)

    return [clusterArray, text, attributes]


def getModelCentroids(model):
    # Le pasamos el modelo/array de clusters calculado con el HierarchicalClustering
    centroids = []
    for i in model: centroids.append(i.getCentroid())
    return centroids


def calculateInstanceCluster(model, instance):
    # Dada una Instance() calcula a que cluster pertenece y cual es la instancia mas cercana
    # Parametros: Un modelo o array de Cluster() y una Intance()
    c = Cluster(len(instance.getAttrVector()))
    c.addIntances(instance)

    dm = clusterDistance(model[0], c)
    clus = 0
    for i in range(1, len(model)):
        if dm > clusterDistance(model[i], c):
            clus = i
            dm = clusterDistance(model[i], c)

    print("En el modelo con Nº de clusters = " + str(len(model)) +
          " la nueva instancia (" + str(instance.getId()) +
          ") pertenece al cluster " + str(clus))
    return clus


def getNearestIntance(cluster, instance):
    # Dado un cluster y una instancia devuelve la instancia mas cercana perteneciente a ese cluster
    # Parametros: Un Cluster() y una Intance()
    l = len(instance.getAttrVector())
    insArray = cluster.getInstances()
    c = Cluster(l)
    c.addIntances(instance)
    dm = np.array([])
    for i in range(0, len(insArray)):
        c1 = Cluster(l)
        c1.addIntances(insArray[i])
        dm = np.append(dm, clusterDistance(c, c1))

    index = np.where(dm == np.amin(dm))[0][0]
    result = insArray[index].getId()
    print("La instancia mas cercana es la Nº " + str(result))

    return result


def printModel(model, text):
    path = "ClusteringResult" + str(len(model)) + ".txt"
    f = open(path, "w")
    for i in range(0, len(model)):
        f.write("CLUSTER " + str(i))
        f.write("\n")
        lis = model[i].getInstances()
        for j in range(0, len(lis)):
            id = lis[j].getId()
            f.write("Instance " + str(id) + " : " + str(text[id]))
            f.write("\n")
        f.write("\n")
    f.close()
    return True


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate(clustered_instances):
    n_clusters = len(clustered_instances)
    cluster_labels = []
    instances = []
    a = 0
    #Se da formato a los datos de los clusters proporcionados
    #Todas las instancias en el mismo array
    #Un array gemelo con la etiqueta correspondiente a la instancia en cada posicion
    for i in clustered_instances:
        cluster = i.getInstances()
        for j in cluster:
            att = j.getAttrVector()
            instances.append(att)
            cluster_labels.append(a)
        a += 1

    #Se calculan las metricas Silhouette e indice de Davies-Bouldin, ambas internas
    silhouette_avg = silhouette_score(instances, cluster_labels)
    davis_bouldin_idx = davies_bouldin_score(instances, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette score is :", silhouette_avg, "(The higher the better)")
    print("For n_clusters =", n_clusters,
          "the Davies-Bouldin index is :", davis_bouldin_idx, "(The lower the better)")


    return [silhouette_avg, davis_bouldin_idx]

# ESTRUCTURA RECIBIDA--> Array con ["Texto","Vector con texto convertido en numeros"


if __name__ == "__main__":
    "Llamada ej: python -c Intuit.csv 3 "
    "Llamada ej: python -l Intuit.csv -i text"
    "-c datasetPath numClusters--> preprocesa el data set, hace el analisis de los datos y crea un nuevo modelo"
    "-l modelPath --> carga un modelo previamente creado"

    "-e && -i solo se pueden usar con la opcion cargar modelo -l"
    "-e --> devuelve los centroides del cluster "
    "-i  text --> devuelve el cluster al que pertenece y instancia mas cercana"
    

    arg = sys.argv
    if len(arg) >= 3:
        if arg[1] == "-c":
            print("Crear modelo")
            path = arg[2]
            nc = arg[3]
            values = preprocessData(path)
            text = values[1]
            attrArray = values[0]
            print(len(attrArray))
            print(len(text))
            print("CALCULANDO CLUSTERING ...")

            model = HierarchicalClustering(int(nc), attrArray)
            clusters = clustersToArray(len(text), model)
            saveModel(text, attrArray, clusters, len(model))
            print("MODELO GUARDADO")
            printModel(model, text)
            pause()
            print("Calculando centroides")
            centroides = getModelCentroids(model)
            print("Centroides del modelo:")
            centroides = getModelCentroids(model)
            print(centroides)
            pause()
            print("PLOTEO DEL CLUSTERING")
            from sklearn.preprocessing import scale

            dfMerged0 = []

            dfMerged0 = pd.read_csv("intuit.csv", low_memory=False)

            dfMerged = dfMerged0['reviewText']

            topicMode = topic_modeling(dfMerged[:100], False)
            scaled_1 = scale(topicMode)


            ##plotClustering(clusters,np.array(centroides),scaled_1)
            ##pause()

            print(clusters)
            i=0
            print("NUM INSTANCIAS POR CLUSTER")
            for cluster in model:
                print("CLUSTER " ,i)
                print(len(cluster.getInstances()))
                i=i+1
            attr = np.array(clusters)
            ##plotClustering(attr, centroides)
            print("----EVALUACION---")
            pause()
            evaluate(model)

        elif arg[1] == "-l":
            print("Cargar modelo CLUSTERING")
            lm = loadModel(sys.argv[2])
            print("----------")
            model = lm[0]
            text = lm[1]
            attrArray = lm[2]
            clusters = clustersToArray(len(text), model)
            printModel(model, text)
            if len(arg) > 3:
                if arg[3] == "-e":
                    print("Calculando centroides")
                    centroides = getModelCentroids(model)
                    print("Centroides del modelo:")
                    print(centroides)
                    pause()
                    print("PLOT")
                    plotClustering(attrArray, centroides)
                    pause()

                elif arg[3] == "-i":
                  

                    print("Cargamos Modelo LDA")
                    with open('lda_model.pk', 'rb') as pickle_file0:
                        lda_model = pickle.load(pickle_file0)


                    print("-----------------")
                    print("Calculando cluster e instancia mas cercana")
                    print(text)
                    pause()
                    print("-------------------------------------")
                    print("NUEVA INSTANCIA INTRODUCIDA")
                    print("-------------------------------------")
                    f = open(arg[4], "r")
                    new_data = f.readline()

                    nt = pd.Series(new_data)
                    text = pd.concat([text, nt], ignore_index=True)
                    print(text)
                    pause()
                    print("Conseguir vector numerico:")
                    tf_vectorizer = CountVectorizer(max_df=0.50, min_df=2, stop_words='english')
                    tf_vectorizer.fit_transform(text)
                    tt=tf_vectorizer.transform([new_data])
                    vectorizado=lda_model.transform(tt)
                    print(vectorizado)
                    pause()
                    print("A que CLUSTER pertenece la nueva INSTANCIA :")
                    indice = len(attrArray)+1
                    print("Instancia numero ",indice)
                    instance = Instance((len(attrArray)+1), vectorizado)
                    calculateInstanceCluster(model, instance)
                    pause()
                    print("INSTACIA mas cercana a la nueva INSTANCIA :")
                    getNearestIntance(model[1], instance)

    else:
        print("Faltan argumentos")
