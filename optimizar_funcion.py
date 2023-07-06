# A Roger Arnau
# 2023 Enero

# f: funcion en R2 en el intervalo [-I,I]^2
# Reinforcement Learning (con NN tipo Q-learning)
# Basado en: Mnih, Playing atari with deep RL
#      	https://doi.org/10.48550/arXiv.1312.5602

import numpy as np
import random
from keras import Sequential, models
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam


def f(x,y):
    return min( 50 + 2*(x-2)**2 + (y+2)**2, 3*(x-7)**2 + (y-7)**2 )

# Hiperparameters

ESPACIO_ESTADOS = 2
ESPACIO_ACCIONES = 5
PASOS_POR_EPISODIO = 50
EPISODIOS = 20

# Con EPISODIOS 30, EPSILON 0.2, GAMMA 0.95  va bien (7,7)
EPSILON = 0.2
GAMMA = 0.9

RN_CAPAS = [32, 48, 32] # Capas ocultas
FUN_ACT = "relu"
TASA_APRENDIZAJE = 0.002

ENTRENA_CADA = 1

# Definicion del entorno

class Nuevo_Entorno():
    
    def __init__(self):
        self.pasos_total = 1
    
    def Paso(self, action):
        
        if action == 1:
            self.x += 1
        if action == 2:
            self.y += 1
        if action == 3:
            self.x -= 1
        if action == 4:
            self.y -= 1
        
        self.x = np.clip(self.x, -10, 10)
        self.y = np.clip(self.y, -10, 10)
        self.recompensa = - f(self.x, self.y)
        self.pasos_total += 1
        
        estado = [self.x, self.y]
        return estado, self.recompensa
    
    def Reinicia(self):
        self.x = 0
        self.y = 0
        return [self.x, self.y]


# Definición del agente

class Nuevo_Agente():
    
    def __init__(self, rn_capas):
        self.memoria = deque(maxlen = 100000)
        self.modelo = self.Crear_Modelo(rn_capas)
        
    def Crear_Modelo(self, rn_capas):
        modelo = Sequential()
        modelo.add( Dense(rn_capas[0],
                          input_shape = (ESPACIO_ESTADOS,),
                          activation = FUN_ACT) )
        for i in range(1,len(rn_capas)):
            modelo.add( Dense(rn_capas[i],
                              activation = FUN_ACT) )
        modelo.add( Dense(ESPACIO_ACCIONES,
                          activation = 'linear') )
        modelo.compile( loss = 'mse',
                        optimizer = Adam(learning_rate = TASA_APRENDIZAJE) )
        modelo.summary()
        
        return modelo
    
    def Guarda(self, estado, accion, recompensa, estado_siguiente):
        self.memoria.append( (estado, accion, recompensa, estado_siguiente) )
    
    def Actua(self, estado, deterministico = False) :
        
        Accion_recompensa = self.modelo(estado)
        
        # Caso deterministico: siempre lo mejor
        if deterministico:
            return np.argmax( Accion_recompensa )
        
        # A veces, aleatorio
        if np.random.rand() <= EPSILON:
            return random.choice( range(ESPACIO_ACCIONES) )
        
        # Elige el mejor
        return np.argmax( Accion_recompensa )
    
    def Muestra_Resultado(self, x_total, y_total, x_act, y_act,
               e, x_ultima, y_ultima, recompensa, puntuacion, acciones,
               dibuja = False, imprime = False) :
    
        if dibuja :    
            fig = plt.figure()
            plt.plot(0, 0, 'ro')
            plt.plot(x_act, y_act)
            plt.plot(x_act[-1], y_act[-1], 'bo')
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.show()
            if e in [1, 5, 10, 15, 20, 1000, EPISODIOS] :
                fig.savefig('images/train_e' + str(e) + '.png')
                with open('data_results/train.e' + str(e) + '.txt', 'w') as f:
                    for i in range(PASOS_POR_EPISODIO) :
                        a = acciones[i+1]
                        if a == 0:
                            f.write( "0" )
                        elif a == 1:
                            f.write( "\\rightarrow" )
                        elif a == 2:
                            f.write( "\\uparrow" )
                        elif a == 3:
                            f.write( "\\leftarrow" )
                        elif a == 4:
                            f.write( "\\downarrow" )
                        f.write(", ")
                                    
        if imprime :
            print("Episodio: {}/{}, x = {:.2f}, y = {:.2f}, recompesa final = {:.2f}".
                  format(e, EPISODIOS, x_ultima, y_ultima, recompensa) )
    
    def Entrena(self, pasos_totales):
        
        if pasos_totales <= 2:
            return
        if ENTRENA_CADA > 1:
            if pasos_totales % ENTRENA_CADA != 0:
                return
        
        # Entrenamos con toda la memoria, pero se podria hacer sobre un subconjunto
        longituda_batch = len(self.memoria)
        batch = random.sample(self.memoria, longituda_batch)
        estados = np.squeeze( np.array([i[0] for i in batch]) )
        acciones = np.squeeze( np.array([i[1] for i in batch]) )
        recompensas = np.squeeze( np.array([i[2] for i in batch]) )
        estados_siguientes = np.squeeze( np.array([i[3] for i in batch]) )
        
        # trabajamos con puntuacion = recompesa actual + futuras
        valor = recompensas + GAMMA * np.amax(
            self.modelo.predict(estados_siguientes, verbose = 0), axis=1)
        valor_RN = self.modelo.predict(estados, verbose = 0)
        valor_RN[[range(longituda_batch)], [acciones]] = valor
        
        self.modelo.fit(x = estados,
                        y = valor_RN,
                        epochs = 1,
                        verbose = 0)
        

if __name__ == "__main__" :
    
    np.random.seed(0)
    entorno = Nuevo_Entorno()
    agente = Nuevo_Agente(RN_CAPAS)
    
    
    ### ENTRENAMIENTO DEL MODELO
    
    x_total, y_total, r_total = [], [], []
    
    for e in range(EPISODIOS) :
        
        x_act, y_act, a_act = [0], [0], [-1]
        estado = entorno.Reinicia()
        estado = np.reshape(estado, (1, ESPACIO_ESTADOS))
        puntuacion = 0
        
        for p in range(PASOS_POR_EPISODIO) :
            accion = agente.Actua(estado)
            estado_siguiente, recompensa = entorno.Paso(accion)
            estado_siguiente = np.reshape(estado_siguiente, (1, ESPACIO_ESTADOS))
            
            agente.Guarda(estado, accion, recompensa, estado_siguiente)
            agente.Entrena(entorno.pasos_total)
            
            x_act.append(entorno.x)
            y_act.append(entorno.y)
            a_act.append(accion)
            estado = estado_siguiente
        
        x_total.append(entorno.x)
        y_total.append(entorno.y)
        r_total.append(entorno.recompensa)
        
        if True:
            agente.Muestra_Resultado(
                x_total, y_total, x_act, y_act,
                e+1, entorno.x, entorno.y, recompensa,
                puntuacion, a_act, True, True)
    
    
    ### DETERMINISTICO    
    
    x_act, y_act, a_act = [0], [0], [-1]
    estado = entorno.Reinicia()
    estado = np.reshape(estado, (1, ESPACIO_ESTADOS))
    
    for p in range(PASOS_POR_EPISODIO) :
        accion = agente.Actua(estado, deterministico = True)
        estado_siguiente, recompensa = entorno.Paso(accion)
        estado_siguiente = np.reshape(estado_siguiente, (1, ESPACIO_ESTADOS))
        
        x_act.append(entorno.x)
        y_act.append(entorno.y)
        a_act.append(accion)
        estado = estado_siguiente
    
    x_total.append(entorno.x)
    y_total.append(entorno.y)
    
    if True:
        agente.Muestra_Resultado(
            x_total, y_total, x_act, y_act,
            1000, entorno.x, entorno.y, recompensa,
            puntuacion, a_act, True, True)
    
    fig = plt.figure()
    plt.plot([i+1 for i in range(EPISODIOS)], r_total)
    plt.show()
    fig.savefig('images/recompensas.png')
        
        
        














