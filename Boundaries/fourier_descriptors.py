import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Aplicar un umbral a los coeficientes de la transformada de Fourier
def filter_frequencies(descriptors, porcentaje):
    """
    Filter the Fourier descriptors to remove noise.
    :param descriptors: list of descriptors
    :param n_descriptors: number of descriptors to use
    :return: filtered descriptors
    """
    #Se calcula el número de muestras que se desean
    n_descriptors = int(len(descriptors)*porcentaje)

    values = np.zeros(len(descriptors))

    #Se calcula el valor absoluto de los coeficientes
    for i in range(len(descriptors)):
        values[i] = abs(descriptors[i])

    #Se ordenan los valores de mayor a menor
    values = np.sort(values)[::-1]

    #Se toman los valores que se desean
    values = values[:n_descriptors]

    #Se crea una lista de ceros
    finales = np.zeros(len(descriptors), dtype=complex)

    #Se recorre la lista de valores y se ponen en la lista de coeficientes
    for i in range(len(descriptors)):
        if abs(descriptors[i]) in values:
            finales[i] = descriptors[i]
    
    if np.mod(len(finales), 2) != 0:
        #Se le quita el último valor
        finales = finales[:-1]
    

    return finales




def fourier_descriptors(contour):
    """
    Compute the Fourier descriptors of a contour.
    :param contourImg: contour to compute the descriptors (image)
    :param n_descriptors: number of descriptors to compute
    :return: list of descriptors
    """


    contourComplex = np.zeros(len(contour), dtype=complex)

    

    for i in range(len(contour)):
        contourComplex[i] = complex(contour[i][0], contour[i][1])
    

    #Se calcula la transformada de fourier
    dft = np.fft.fft(contourComplex)




    descriptors = dft.copy()

    


    return descriptors

def plot_IDFT(descriptors, porcentaje, width = 28, height=28):
    """
    Reconstruct a contour from its Fourier descriptors.
    :param descriptors: list of descriptors
    :param n_descriptors: number of descriptors to use
    :return: reconstructed contour
    """
    #Se toma el número de muestras que se desean
    descriptors = filter_frequencies(descriptors, porcentaje)


    #Se calcula la transformada inversa de fourier
    idft = np.fft.ifft(descriptors)



    #Se crea una imagen de zeros
    img = np.zeros(( width, height), dtype=np.uint8)

    #Se recorre la imagen y se pone en 255 los puntos que se encuentran en la transformada inversa
    for i in range(len(idft)):
        img[int(idft[i].real)][int(idft[i].imag)] = 255


    plt.imshow(img, cmap='gray')
    plt.show()

    

    return img



def calculate_rotation_invariance(descriptors):
    #Se calcula el valor absoluto de los coeficientes
    for i in range(len(descriptors)):
        descriptors[i] = abs(descriptors[i])

    descriptors = np.real(descriptors)

    return descriptors

def calculate_scale_invariance(descriptors):
    #Se dividen todos los puntos por a[1]
    constante = abs(descriptors[1])

    print(constante)
    return descriptors/constante
