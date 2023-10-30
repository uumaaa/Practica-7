import numpy as np
def morfologia(imagenOriginal:np.array,kernel:np.array,operacion:int):
    #Creacion de la nueva matriz con las posiciones aumentadas
    #Operacion        0 ->> Erosion     1->> Dilatacion
    filasM = imagenOriginal.shape[0]
    columnasM = imagenOriginal.shape[1]
    filasK =  kernel.shape[0]
    columnasK = kernel.shape[1]
    nuevasFilas = filasM + (filasK - 1)
    nuevasColumnas = columnasM + (columnasK - 1)
    diferenciaColumnas = int((nuevasColumnas - columnasM)/2)
    diferenciaFilas = int((nuevasFilas - filasM) / 2)
    matrizExtendida = np.zeros((nuevasFilas,nuevasColumnas))
    matrizResultado = np.zeros((filasM,columnasM))
    #Se insertan los valores de la martriz original en la matriz extendida
    for x in range(diferenciaFilas,nuevasFilas-diferenciaFilas):
        for y in range(diferenciaColumnas,nuevasColumnas-diferenciaColumnas):
            matrizExtendida[x][y] = imagenOriginal[x-diferenciaFilas][y-diferenciaColumnas]
    
    #Se aplica convolucion en la matriz aumentada
    for x in range(diferenciaFilas,nuevasFilas-diferenciaFilas):
        for y in range(diferenciaColumnas,nuevasColumnas-diferenciaColumnas):
            simplificacionBooleana = False
            for xK in range(filasK):
                for yK in range(columnasK):
                    if(operacion == 1):
                        if(kernel[xK][yK] == 1 and matrizExtendida[x-diferenciaFilas+xK][y-diferenciaColumnas+yK] == 255):
                            matrizResultado[x-diferenciaFilas][y-diferenciaColumnas] = 255
                            simplificacionBooleana = True
                            break
                    if(operacion == 0):
                        if(kernel[xK][yK] == 1 and matrizExtendida[x-diferenciaFilas+xK][y-diferenciaColumnas+yK] == 0):
                            matrizResultado[x-diferenciaFilas][y-diferenciaColumnas]= 0 
                            simplificacionBooleana = True
                            break
                if(simplificacionBooleana):
                    break
            if(not simplificacionBooleana):
               if(operacion == 0):
                   matrizResultado[x-diferenciaFilas][y-diferenciaColumnas] = 255
               else:
                   matrizResultado[x-diferenciaFilas][y-diferenciaColumnas] = 0
    return matrizResultado

def cerradura(imagenOriginal:np.array,tamanioKernel:int,iteraciones:int):
    #Cerradura
    kernel = np.ones([tamanioKernel,tamanioKernel])
    imgRes = np.copy(imagenOriginal)
    for _ in range(iteraciones):
        imgRes = morfologia(imgRes,kernel,1)
        imgRes = morfologia(imgRes,kernel,0)
    return imgRes

def apertura(imagenOriginal:np.array,tamanioKernel:int,iteraciones:int):
    kernel = np.ones([tamanioKernel,tamanioKernel])
    imgRes = np.copy(imagenOriginal)
    for _ in range(iteraciones):
        imgRes = morfologia(imgRes,kernel,0)
        imgRes = morfologia(imgRes,kernel,1)
    return imgRes

def dilatacion(imagenOriginal:np.array,tamanioKernel:int,iteraciones:int):
    kernel = np.ones([tamanioKernel,tamanioKernel])
    imgRes = np.copy(imagenOriginal)
    for _ in range(iteraciones):
        imgRes = morfologia(imgRes,kernel,1)
    return imgRes

def erosion(imagenOriginal:np.array,tamanioKernel:int,iteraciones:int):
    kernel = np.ones([tamanioKernel,tamanioKernel])
    imgRes = np.copy(imagenOriginal)
    for _ in range(iteraciones):
        imgRes = morfologia(imgRes,kernel,0)
    return imgRes