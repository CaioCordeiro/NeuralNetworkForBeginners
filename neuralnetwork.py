import numpy as np
import random as ran
import math
#Script para gerar um numero bem aleatório
def randomizer(seed):
	while True:
		x = seed
		i = 0
		lista =[]
		while(i!= x):
			value = ran.randrange(x+1)
			if value in lista:
				continue
			if value == 0 :
				continue
			lista.append(value)
			i = i+1
		posi = ran.randrange(x)
		return((lista[posi])/100)


# Sigmoid
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1 / (1 + math.exp(-x))
    
# Input
X = int(input("Insira o valor que quer treinar(1-100): "))
    
# Output           
y = int(input("Insira o valor que voce quer chegar(1-100): "))

X = X/100 
y = y/100

#Definindo o numero de propagações
n = int(input("Insira o numero de propagações(quanto maior o numero,mais demora!): "))

# Criando numeros aleatórios
seed = 34

# Inicializando pesos aleatórios
syn0 = 2*(randomizer(seed)) - 1

for iter in range(n):

    # Forward propagation
    l0 = X
    l1 = nonlin((l0*syn0))

    # Quanto a gente errou?
    l1_error = y - l1

    # Multiplicamos quanto erramos  pela 
    # angulação da função sigmoid no valor de l1
    l1_delta = l1_error * nonlin(l1,True)

    # Atualizar os pesos
    syn0 += (l0*l1_delta)

print ("Output Depois de Treinar:")
print (l1*100)
