Se deben tener previamente instaladas las siguientes librerias:
-numpy
-scipy
-pandas
-pickle
-sklearn
-nltk
-gensin
-matplotlib
-seaborn



El programa analisisDatos.py puede crear o cargar un de clustering jerárquico:
* -c dataPath numClusters → para crear un nuevo modelo. Se creará un fichero ModelX.csv, un InfoX.txt y ClusteringResultX.txt. 
   * ModelX.csv: un csv con el modelo HierarchicalClustering guardado
   * InfoX.txt: Información de cada interacción del HierarchicalClustering
   * ClusteringResultX.txt: Muestra en texto la repartición del texto y los clusters
   * Ej:python analisis.py -c Intuit.csv 3
* -l modelPath → para guardar un modelo previamente calculado
   * python analisis.py -l Model1.csv


Además, la opción de cargar modelo tiene una opciones adicionales (sólo se puede una opción por llamada):
* -e → devuelve por pantalla los centroides del modelo cargado
   * Ej: python analisis.py -l ModelPath -e
* -i textPath → calcula el cluster y la instancia más cercana del texto dado y lo imprime por pantalla
   * Ej: python analisis.py -l ModelPath -i textPath