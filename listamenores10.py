from math import *
import random

n = 10
lista = [random.randint(1,30) for _ in range(n)]
print ("la lista aleatoria creada es :", lista)

for i in range(0,len(lista)):
    if lista[i] < 10:
        print (lista[i])











