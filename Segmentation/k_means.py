#Script que implementa k-means en una imagen, buscando clusteres en los colores de la imagen

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2  


class PixelList:
    def __init__(self, image):
        self.image = image
        self.postions = []
        self.pixels = []
        self.get_pixels()

    def get_pixels(self):
        # Obtener una lista de todos los píxeles de la imagen
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.pixels.append(self.image[i, j])
                self.postions.append((i, j))

    def get_data(self):
        # Obtener una lista de todos los píxeles de la imagen
        return np.array(self.pixels)
    
    def get_positions(self):
        # Obtener una lista de todas las posiciones de los píxeles de la imagen
        return np.array(self.postions)

class Kmeans_classifier:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self):
        # Inicializar los centroides de manera aleatoria con puntos generados
        # aleatoriamente dentro del rango de valores de cada característica
        self.centroids = []
        for i in range(self.k):
            centroid = []
            for j in range(len(self.data[0])):
                centroid.append(random.uniform(min(self.data[:, j]), max(self.data[:, j])))
            self.centroids.append(centroid)

        #Si el núemero de clusters es 7, se inicializan los centroides con los colores de los circulos
        if self.k == 7:
            self.centroids[0] = [200,59,57] #Rojo
            self.centroids[1] = [107,210,80] #Verde
            self.centroids[2] = [99, 201, 225] #Azul Claro
            self.centroids[3] = [63, 115, 211] #Azul Fuerte
            self.centroids[4] = [186,61,191] #Morado
            self.centroids[5] = [232,216,93] #Amarillo
            self.centroids[6] = [33, 37, 63] #Fondo de la imagen
        

    def assign_to_clusters(self):
        # Asignar cada punto de datos al clúster más cercano
        self.clusters = [[] for _ in range(self.k)]
        
        for data_point in self.data:
            min_distance = float('inf')
            closest_cluster = None
            
            for i, centroid in enumerate(self.centroids):
                distance = self.euclidean_distance(data_point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i
            
            self.clusters[closest_cluster].append(data_point)

    def update_centroids(self):
        # Calcular nuevos centroides basados en los puntos asignados a cada clúster
        new_centroids = []
        for cluster in self.clusters:
            if cluster:
                new_centroid = [sum(x) / len(cluster) for x in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                # Si un clúster está vacío, el centroide permanece igual
                new_centroids.append(self.centroids[len(new_centroids)])
        
        self.centroids = new_centroids

    def calculate_sswc(self):
        sswc = 0.0
        for i in range(self.k):
            cluster_points = np.array(self.clusters[i])
            if len(cluster_points) > 0:
                centroid = self.centroids[i]
                sswc += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        return sswc

    def fit(self, max_iterations=100):
        self.initialize_centroids()

        for _ in range(max_iterations):
            self.assign_to_clusters()
            old_sswc = self.calculate_sswc()
            self.update_centroids()
            new_sswc = self.calculate_sswc()

            # Comprobar si el SSWC ha convergido (cambia poco entre iteraciones)
            if abs(old_sswc - new_sswc) < 1e-6:
                break

    def predict(self, new_data):
        # Predecir el clúster al que pertenecen nuevos datos
        predictions = []
        for data_point in new_data:
            min_distance = float('inf')
            closest_cluster = None
            
            for i, centroid in enumerate(self.centroids):
                distance = self.euclidean_distance(data_point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i
            
            predictions.append(closest_cluster)
        
        return predictions
    
    

    def visualize_clusters(self):
        # Crea un objeto de ejes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Asigna etiquetas a los ejes
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')

        # Inicializa una lista de colores para cada cluster
        cluster_colors = []

        # Calcula el color promedio de cada cluster y normaliza los valores al rango 0-1
        for cluster in self.clusters:
            if cluster:
                cluster_data = np.array(cluster)
                cluster_color = np.mean(cluster_data, axis=0) / 255.0
                cluster_colors.append(cluster_color)
            else:
                cluster_colors.append([0, 0, 0])  # Si el cluster está vacío, se usa negro

        cluster_colors = np.array(cluster_colors)

        # Se plotean los puntos de cada cluster con su respectivo color promedio
        for i, cluster in enumerate(self.clusters):
            if cluster:
                cluster_data = np.array(cluster)
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], s=0.1, c=[cluster_colors[i]])

        # Se plotean los centroides, los centroides deben ser grandes para que se vean bien
        centroid_data = np.array(self.centroids)
        ax.scatter(centroid_data[:, 0], centroid_data[:, 1], centroid_data[:, 2], s=100, c='black')

        #Se asignan etiquetas R, G y B al gráfico


        plt.show()


    def calculate_wcss(data, k):
        wcss_values = []
        for i in range(1, k + 1):
            kmeans = Kmeans_classifier(data, i)  # Elimina el argumento None
            kmeans.fit()
            wcss = 0
            for j in range(len(data)):
                cluster_index = kmeans.predict([data[j]])[0]
                centroid = kmeans.centroids[cluster_index]
                distance = Kmeans_classifier.euclidean_distance(data[j], centroid)  # Usa el método estático
                wcss += distance ** 2
            wcss_values.append(wcss)
        return wcss_values


    @staticmethod
    def euclidean_distance(point1, point2):
        # Calcular la distancia euclidiana entre dos puntos
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))
    
    def generate_cluster_images(self, image):
        cluster_images = []  # Lista para almacenar las imágenes de los clústeres

        # Obtén los clústeres después de ajustar el modelo K-means
        clusters = self.clusters

        # Asigna un color específico a cada clúster
        cluster_colors = []
        for cluster in clusters:
            if cluster:
                cluster_data = np.array(cluster)
                cluster_color = np.mean(cluster_data, axis=0).astype(int)
                cluster_colors.append(cluster_color)
            else:
                # Si el clúster está vacío, usa un color negro
                cluster_colors.append([0, 0, 0])

        # Crea una imagen vacía con las mismas dimensiones que la imagen original
        height, width, _ = image.shape
        segmented_images = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(cluster_colors))]

        # Recorre la imagen original y asigna el color correspondiente a cada píxel según su clúster
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                cluster_index = self.predict([pixel])[0]
                segmented_images[cluster_index][i, j] = cluster_colors[cluster_index]

        # Agrega las imágenes de los clústeres a la lista
        cluster_images = segmented_images

        return cluster_images
    

def main():
    # Leer imagen
    image = cv2.imread('images/lena.png')
