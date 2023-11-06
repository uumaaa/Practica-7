import cv2
import numpy as np


def calculate_chain_code(image):
    # Define changes in x and y for 8 possible directions
    change_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    change_y = [-1, -1, -1, 0, 1, 1, 1, 0]

    # Find the first pixel on the boundary of the object.
    start_y = None
    start_x = None
    height = len(image)
    width = len(image[0])
    for r in range(height):
        for c in range(width):
            if (
                image[r][c] == 255
            ):  # Assuming 0 represents the background, change to your background value if necessary
                start_y = r
                start_x = c
                break  # Starting point found
        if start_y:
            break

    # Initialize current pixel coordinates
    r = start_y
    c = start_x

    # Initialize the chain code
    chain_code = []
    if(r is None or c is None):
        return []
    direction = 3

    while True:
        # Se calcula la dirección
        b_direction = (direction + 5) % 8
        # Se busca el siguiente pixel en la dirección, se itera cada dirección hasta encontrar un pixel blanco
        for direction in range(b_direction, 8):
            # Posición del nuevo pixel
            new_r = r + change_y[direction]
            new_c = c + change_x[direction]

            # Si el pixel está dentro de la imagen y es blanco, se agrega a la cadena y se actualiza la posición
            if 0 <= new_r < height and 0 <= new_c < width and image[new_r][new_c] != 0:
                chain_code.append(direction)
                r = new_r
                c = new_c
                break
        # Si no se encontró un pixel blanco en la dirección, se busca en las direcciones restantes
        else:
            for direction in range(0, b_direction):
                new_r = r + change_y[direction]
                new_c = c + change_x[direction]
                if (
                    0 <= new_r < height
                    and 0 <= new_c < width
                    and image[new_r][new_c] != 0
                ):
                    chain_code.append(direction)
                    r = new_r
                    c = new_c
                    break
        # Si se regresa al pixel inicial, se termina la cadena
        if (r, c) == (start_y, start_x) and len(chain_code) > 0:
            break

    return chain_code


# Dunción que dibuja el borde usando la salida de la función de arriba
def dibujar_borde(image, chain_code, only_border=False):
    # Color imagen
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

   

    # Se definen las direcciones
    change_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    change_y = [-1, -1, -1, 0, 1, 1, 1, 0]

    # Se encuentra el primer pixel en el borde del objeto
    start_y = None
    start_x = None
    height = len(image)
    width = len(image[0])
    for r in range(height):
        for c in range(width):
            if image[r][c] == 255:
                start_y = r
                start_x = c
                print("Start point:", (start_y, start_x))
                break
        if start_y:
            break

    # Según el chain code se van coloreando los pixeles
    r = start_y
    c = start_x

    if only_border:
        #Se dibuja sobre una imagen de zeros
        color_image = np.zeros((height, width, 3), np.uint8)
    for i in chain_code:
        new_r = r + change_y[i]
        new_c = c + change_x[i]
        color_image[new_r][new_c] = (0, 0, 255)
        r = new_r
        c = new_c

    return color_image


def find_minimum_magnitude(chain_code):
    menor_magnitud = float(
        "inf"
    )  # Inicializa la menor magnitud como infinito para asegurarse de encontrar un valor menor.
    indice_menor_magnitud = 0  # Inicializa el índice de la menor magnitud como 0.
    for i in range(len(chain_code)):
        subcadena = chain_code[i:] + chain_code[:i]  # Forma la subcadena circular
        subcadena_str = "".join(
            map(str, subcadena)
        )  # Convierte la subcadena en una cadena de dígitos
        magnitud = int(subcadena_str)  # Convierte la cadena en un número entero

        if magnitud < menor_magnitud:
            menor_magnitud = magnitud
            indice_menor_magnitud = i

    return chain_code[indice_menor_magnitud:] + chain_code[:indice_menor_magnitud]


def print_chains(chain_code):
    # Se normalizan las direcciones, el 3 pasa a ser el 0, el 4 el 1, ...
    for i in range(len(chain_code)):
         chain_code[i] = (chain_code[i] + 5) % 8

    # Se imprime la cadena
    #print("Chain Code:")
    #print(chain_code)

    # Se calcula la primera diferencia
    """
    R(cki)={mod{cki+1−cki,k},mod{ck0−cki,k}, for 0≤i<N−1 for i=N−1
    """
    first_difference = []
    for i in range(len(chain_code) - 1):
        first_difference.append((chain_code[i + 1] - chain_code[i]) % 8)
    first_difference.append((chain_code[0] - chain_code[len(chain_code) - 1]) % 8)

    # Se imprime la primera diferencia
    #print("First Difference:")
    #print(first_difference)

    # Se encuentra la menor magnitud que se puede formar con la cadena, es decir, el menor número que se puede formar con los elementos de la cadena
    minimum_magnitude = find_minimum_magnitude(chain_code)
    #print("minimum magnitude:", menor_magnitud)

    return chain_code, first_difference, minimum_magnitude

def normalize_chain(chain_code):
    # Se normalizan las direcciones, el 3 pasa a ser el 0, el 4 el 1, ...
    for i in range(len(chain_code)):
        chain_code[i] = (chain_code[i] + 5) % 8
    first_difference = []
    for i in range(len(chain_code) - 1):
        first_difference.append((chain_code[i + 1] - chain_code[i]) % 8)
        first_difference.append((chain_code[0] - chain_code[len(chain_code) - 1]) % 8)
    minimum_magnitude = find_minimum_magnitude(first_difference)
    normalized_chain = minimum_magnitude
    return normalized_chain

if __name__ == "__main__":
    # Carga la imagen binaria de bordes
    image_path = "Results/daltonismo.png"
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    chain_code = calculate_chain_code(binary_image)
    chain1, first_difference1, menor_magnitud1 = print_chains(chain_code.copy())
    result_image = dibujar_borde(binary_image, chain_code)
    border_image = dibujar_borde(binary_image, chain_code, only_border=True)

    cv2.imshow("Image with Freeman Chain Code", result_image)
    cv2.imshow("Chain Code", border_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Se rota la imagen 90 grados para comparar
    rotated_image = cv2.rotate(binary_image, cv2.ROTATE_90_CLOCKWISE)
    chain_code = calculate_chain_code(rotated_image)
    chain2, first_difference2, menor_magnitud2 = print_chains(chain_code.copy())
    result_image = dibujar_borde(rotated_image, chain_code)
    cv2.imshow("Image with Freeman Chain Code", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Se comparan los resultados
    print("Comparación de resultados:")
    print("Menor magnitud:", menor_magnitud1 == menor_magnitud2)
