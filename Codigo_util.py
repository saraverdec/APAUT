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
hist = plt.hist(df.NOMBRE_COL, bins = Nº_ RECTANGULOS, density = True)
