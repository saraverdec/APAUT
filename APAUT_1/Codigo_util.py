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


##### EMPIEZA AQUÍ
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# read the data, 30 features for each record
df = pd.read_csv('./data/wdbc.csv', header = None) # La primera fila no es el cabecero

#vemos los datos
df.shape

df.head()

df.describe

df.dtypes # todos son float 64 menos la primera y la segunda filas

y = (data.values[ :, 1 ] == 'B').astype(int) #transformamos la segunda columna en 0 y 1 en función de si son M o B, respectivamente 
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


clean_data = data.values[:, 2:].astype(float)

y = (data.values[:,1] == 'B').astype(int)

target_names = np.array([('benign'), ('malign')], dtype = np.dtype('U10'))

clean_data= pd.DataFrame(clean_data)

df = pd.DataFrame(clean_data)

df['target'] = target_names[y]

df


clean_data = data.values[:, 2:].astype(float)

clean_data= pd.DataFrame(clean_data)

desc = clean_data.describe()
means = desc.loc['mean']
stds = desc.loc['std']
for column in range (clean_data.shape[1]):
    clean_data.iloc[:,column]=(clean_data.iloc[:,column]-means[column])/stds[column]

clean_data


y = (data.values[:,1] == 'B').astype(int)

target_names = np.array([('benign'), ('malign')], dtype = np.dtype('U10'))

df = pd.DataFrame(clean_data)

df['target'] = target_names[y]

df


pca = PCA().fit(clean_data)

var_list=pca.explained_variance_ratio_.cumsum()

for value in range(len(var_list)):
    if var_list[value] >= 0.99:
        print(value+1, var_list[value])
        break



pca=PCA(n_components=17).fit(clean_data).transform(clean_data)

plt.figure()

comp1=0; #first component to visualize, you can modify it
comp2=1; #second component to visualize, you can modify it

# plot the two components selected above for both malign and benign tumors
plt.scatter(pca[y == 0, comp1], pca[y == 0, comp2], color='r', alpha=.7, lw=1,
                label='malign')

plt.scatter(pca[y == 1, comp1], pca [y == 1, comp2], color='g', alpha=.1, lw=1,
                label='beign')

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA WDBC dataset')

plt.show()



kmeans = KMeans(n_clusters=2)

kmeans.fit(clean_data)

y_kmeans = kmeans.predict(clean_data)

cluster = y & y_kmeans


plt.scatter(pca[cluster == 1, comp1], pca[cluster == 1, comp2], color='b', alpha=1, lw=1,
                label='malign')

plt.scatter(pca[cluster == 1, comp1], pca [cluster == 1, comp2], color='b', alpha=.1, lw=1,
                label='beign')

plt.scatter(pca[y == 0, comp1], pca[y == 0, comp2], color='r', alpha=.1, lw=1,
                label='malign')

plt.scatter(pca[y == 1, comp1], pca [y == 1, comp2], color='g', alpha=.1, lw=1,
                label='beign')

target_names_2 = np.array([('M'), ('B')], dtype = np.dtype('U10'))

df['cluster'] = target_names_2[y_kmeans]
