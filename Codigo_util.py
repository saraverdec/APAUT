#En este docuento escribimos funciones útiles a la hora de la elaboración del trabajo, extraídas de los notebooks

#pandas es una librería que permite "high-performance, easy-to-use data structures and data analysis tools"
#Usaremos pandas para leer los csv y poder representarlos
import pandas as pd

df = pd.read_csv('NOMBRE.csv',sep=';')
df.head(LINE_Nº)
df.shape
df.describe()
df.drop('NOMBRE_COL', axis=1)
df.types

#.mean(axis=0 [will give you the calculated value per column]) - returns the statistical mean
#.median(axis=0 [will give you the calculated value per column]) - returns the statistical median
#.mode(axis=0 [will give you the calculated value per column]) - returns the statistical mode
#.count() - gives number of total values in column
#.unique() - returns array of all unique values in that column
#.value_counts() - returns object containing counts of unique values

df.NOMBRE_COL.FUNCION()

#Histogramas
import matplotlib.pyplot as plt
#Density cambia los valores de absolutos a relativos
hist = plt.hist(df.NOMBRE_COL, bins = Nº_RECTANGULOS, density = True)

#PCA
plt.scatter(X1, X2, alpha=(0-1, transparencia), color="letra", label="string")
plt.legend()
plt.title("string")

from sklearn.decomposition import PCA

pca=PCA(n_components=n)
pca.fit(X).transform(X)
pca.inverse_transform(X_transform)

pca.components_ #Comoponentes de los vectores de PCA, sobre los que se produce la reducción dimensional
pca.expalined_variance_ #Cuanta varianza explica cada PCA, longitud del vector
pca.explained_variance_ratio_ #Lo mismo sobre 1

# read the data, 30 features for each record
data = pd.read_csv('./data/wdbc.csv')
X = data.values[ :, 2:].astype(float) #Todas las filas, las columnas de la 3 en adelante
y = (data.values[ :, 1 ] == 'B').astype(int) #Hacemos una parte de un facto, el vector [0/1] codificando para Malign/Benign
target_names = np.array([('benign'), ('malign')], dtype=np.dtype('U10')) #Hacemos la otra parte del factor
X5 = X[ :, 0:5].astype(float) #Se quedan solo con las 5 primeras columnas
df_cancer = pd.DataFrame(X5) #Hacen el DF
df_cancer['target'] = target_names[y] #Añaden una columna que construye el factor Benign/Malign

sns.pairplot(DF, hue=NOMBRE_COL)


pca=PCA().fit(X)
np.cumsum(pca.explained_variance_ratio))

kmeans=KMeans(n_clusters=N)
kmeans.fit(X)
kmeans.predict(X)
kmeans.fit_predict(X)


from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=4).fit(X)

gmm.predict(X)


#--------------------------------------------------------------------------------


pca = PCA().fit(X)

var_list=pca.explained_variance_ratio_.cumsum()

for value in range(len(var_list)):
    if var_list[value] >= 0.99:
        print(value+1, sum(var_list[:value]))
        break
