# -*- coding: utf-8 -*-
"""
@author: Mariana Varela Gómez 
         Cesar David Vargas Cano
"""

import os
import cv2
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pathlib

t = time.time()

# os.environ["OMP_NUM_THREADS"] = "1"
# mpl.rc('figure', max_open_warning = 0)

c1 = 0  # 2 Blue
c2 = 1  # 0 Green
c3 = 2  # 1 Red

# To define whether images are saved
IS_SAVED = False;

# PATHS
PATH_LOCAL = str(pathlib.Path(__file__).parent.absolute())

PATH_TRAINING_IMG = PATH_LOCAL + "/1. ENTRENAMIENTO"
PATH_TRAINING_RESULTS = PATH_LOCAL + "/R_ENTRENAMIENTO"

PATH_TESTING_IMG = PATH_LOCAL + "/2. PRUEBAS"
PATH_TESTING_RESULTS = PATH_LOCAL + "/R_PRUEBAS"

PATH_VERIFICATION_IMG = PATH_LOCAL + "/3. COMPROBACION"
PATH_VERIFICATION_RESULTS = PATH_LOCAL + "/R_COMPROBACION"

# %% # ---------------------------- Get desviations ----------------------------
def save_image(nombre, archivo, path, isSaved):
    if (isSaved):
        path = path + "/" + nombre + ".JPG"
        cv2.imwrite(path, archivo)
        
def save_histogram(path_dest, nombre_img, name_hist, hist, isSaved):
    if (isSaved):
        plt.figure(figsize=(11, 7), dpi=600)
        plt.subplot(111), plt.plot(hist)
        plt.tick_params(labelsize = 20)
        # plt.title(nombre_img + ' ' + name_hist, fontsize=19)
        plt.xlabel('Gray-Scale Value', fontsize=20)
        plt.ylabel('Number of Pixels', fontsize=20)
        plt.savefig(path_dest + '/HISTOGRAMS/' + nombre_img + name_hist + '.jpg')
        plt.close()

def get_img_names(path_source):
    """
    Get the name of all images in the path, because there are 5 images per name, 
        e.g. img1_NIR, img1_RGB, etc. Just get the different names, for example 
        "img1" from the given example.

    Parameters
    ----------
    path_source : string
        Path where the images are located.

    Returns
    -------
    imgs_names : array of strings
        Unique image names.

    """
    contents = os.listdir(path_source)
    contents = contents[0:(len(contents))]
    contents_list = []
    # Delete names with _RGB.JPG to avoid naming issues
    for name in contents:
        if name[(len(name)-8):] == '_RGB.JPG':
            contents.remove(name)
    # Convert to a list
    for name in contents:
        contents_list.append(name[:(len(name)-8)])
    # Delete similar names
    imgs_names = []
    for item in contents_list:
        if item not in imgs_names:
            imgs_names.append(item)

    return imgs_names


def get_image_path(img_name, path_source):
    """
    Concatenate the name of the images with the path where they are located.

    Parameters
    ----------
    img_name : string
        Image name.
    path_source : string
        Source path.

    Returns
    -------
    img_path : array of string
        Full image path

    """
    img_path = [glob.glob(path_source + "/*" + img_name + "_RGB.JPG"),
                glob.glob(path_source + "/*" + img_name + "_GRE.TIF"),
                glob.glob(path_source + "/*" + img_name + "_NIR.TIF"),
                glob.glob(path_source + "/*" + img_name + "_RED.TIF"),
                glob.glob(path_source + "/*" + img_name + "_REG.TIF")]

    return img_path


def get_image_file(img_path):
    """
    Get the image file 

    Parameters
    ----------
    img_path : string
        Full image path

    Returns
    -------
    img_NIR : numpy.ndarray
        Image file.
    img_RED : numpy.ndarray
        Image file.
    """
    end = (len(str(img_path[1]))-2)
    # img_RGB = np.array(cv2.imread(str(img_path[0])[2:end], 0), dtype=float)
    # img_GRE = np.array(cv2.imread(str(img_path[1])[2:end], 0), dtype=float)
    img_NIR = np.array(cv2.imread(str(img_path[2])[2:end], 0), dtype=float)
    img_RED = np.array(cv2.imread(str(img_path[3])[2:end], 0), dtype=float)
    # img_REG = np.array(cv2.imread(str(img_path[4])[2:end], 0), dtype=float)

    return img_NIR, img_RED


def get_NDVI(NIR, RED):
    ndvi = np.where((NIR + RED) == 0, 0, (NIR - RED) / (NIR + RED))

    if ndvi.min() < 0:
        ndvi = ndvi + (ndvi.min() * -1)

    ndvi = (ndvi * 255) / ndvi.max()
    ndvi = ndvi.round()

    ndvi_image = np.array(ndvi, dtype=np.uint8)
    ndviCalculado = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_RAINBOW)

    return ndviCalculado


def get_EVI2(NIR, RED):

    evi2 = np.where((NIR + RED) == 0, 0, 2.5 * (NIR - RED) / (NIR+2.4*RED+1))
    vmine, vmaxe = np.nanpercentile(evi2, (1, 99))

    if evi2.min() < 0:
        evi2 = evi2 + (evi2.min() * -1)

    evi2 = (evi2 * 255) / evi2.max()
    evi2 = evi2.round()

    evi2_image = np.array(evi2, dtype=np.uint8)
    evi2Calculado = cv2.applyColorMap(evi2_image, cv2.COLORMAP_RAINBOW)

    return evi2Calculado


def get_vegetation_indices(img_NIR, img_RED, img_name, path_dest):
    # NDVI
    img_NDVI = get_NDVI(img_NIR, img_RED)
    save_image(img_name + '_indice_NDVI', img_NDVI, path_dest, IS_SAVED)
    
    img_NDVI_gray = cv2.cvtColor(img_NDVI, cv2.COLOR_BGR2GRAY)
    save_image(img_name + '_indice_NDVI_gray', img_NDVI_gray, path_dest, IS_SAVED)
    
    # EVI2
    img_EVI2 = get_EVI2(img_NIR, img_RED)
    save_image(img_name + '_indice_EVI2', img_EVI2, path_dest, IS_SAVED)
    
    img_EVI2_gray = cv2.cvtColor(img_EVI2, cv2.COLOR_BGR2GRAY)
    save_image(img_name + '_indice_EVI2_gray', img_EVI2_gray, path_dest, IS_SAVED)

    return img_NDVI, img_EVI2_gray, img_NDVI_gray


def get_mask(vegIndex):
    complemento = 255 - vegIndex
    img = cv2.cvtColor(complemento, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(
        img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


def mask_image(NDVI_mask, vegIndex):
    img_copy = vegIndex.copy()
    img_masked = cv2.bitwise_and(img_copy, img_copy, mask=NDVI_mask)

    return img_masked


def get_an_histogram(img, mask, imgName, name_hist, path_dest):
    histogram = cv2.calcHist([img], [0], mask, [256], [0, 256])
    save_histogram(path_dest, imgName, name_hist, histogram, IS_SAVED)

    return histogram


def get_histograms(img_EVI2_gray, img_NDVI_gray, mask, imgName, path_dest):
    # EVI2
    hist_EVI2_gray = get_an_histogram(img_EVI2_gray, mask, imgName,
                                      'hist_EVI2_gray', path_dest)
    # NDVI
    hist_NDVI_gray = get_an_histogram(img_NDVI_gray, mask, imgName,
                                      'hist_NDVI_gray', path_dest)

    histograms = [hist_EVI2_gray, hist_NDVI_gray]  # Return

    return histograms


def get_size_interest_area(mascara):
    sizeInterestArea = np.size(np.where(mascara == 255)[1])

    return sizeInterestArea


def convert_histogram(histograma):
    hist = []
    for h in histograma:
        # hist.append(h[0])
        hist.append(h)

    return hist


def process_by_image(img_name, path_source, path_dest):
    # Get the arrays of each monochrome image
    img_path = get_image_path(img_name, path_source)
    img_NIR, img_RED = get_image_file(img_path)

    # Get vegetation indices
    img_NDVI, img_EVI2_gray, img_NDVI_gray = get_vegetation_indices(img_NIR,
                                                                    img_RED, img_name, path_dest)

    # Get mask
    NDVI_mask = get_mask(img_NDVI)
    save_image(img_name + '_NDVI_mask', NDVI_mask, path_dest, IS_SAVED)

    # Mask to remove zone of null interest
    img_EVI2_masked = mask_image(NDVI_mask, img_EVI2_gray)  # Return
    save_image(img_name + '_masked_EVI2', img_EVI2_masked, path_dest, IS_SAVED)

    img_NDVI_masked = mask_image(NDVI_mask, img_NDVI_gray)  # Return
    save_image(img_name + '_masked_NDVI', img_EVI2_masked, path_dest, IS_SAVED)

    # Get histograms
    histograms = get_histograms(img_EVI2_masked, img_NDVI_masked,
                                NDVI_mask, img_name, path_dest)

    # Get the size of the interest area
    sizeInterestArea = get_size_interest_area(NDVI_mask)

    # Normalize
    h_EVI2 = convert_histogram(np.array(histograms[0][1:])/sizeInterestArea)
    h_NDVI = convert_histogram(np.array(histograms[1][1:])/sizeInterestArea)

    # Get Standard deviation
    des_EVI2 = np.std(h_EVI2)*1e4
    des_NDVI = np.std(h_NDVI)*1e4

    return des_EVI2, des_NDVI


def main_img_loop(imgs_names, path_source, path_dest):
    print('\t Processing images')
    desvi_EVI2 = []
    desvi_NDVI = []
    for name in imgs_names:
        des_EVI2, des_NDVI = process_by_image(name, path_source, path_dest)
        desvi_EVI2.append(des_EVI2)
        desvi_NDVI.append(des_NDVI)

    return (desvi_EVI2, desvi_NDVI)


def get_desviations(path_source, path_dest):
    print('\t Getting desviations')
    # Get images names
    imgs_names = get_img_names(path_source)
    # Get characteristics
    desvi_EVI2, desvi_NDVI = main_img_loop(imgs_names, path_source, path_dest)
    deviations = np.transpose(np.array([desvi_NDVI, desvi_EVI2]))

    return deviations, desvi_EVI2, desvi_NDVI

# -------------------------- End of get desviations --------------------------

# %% # -------------------------------- Kmeans ---------------------------------


def kmeans_method(data_deviations):
    print('\t Kmeans method')
    kmeans = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300,
                    random_state=128, algorithm='full').fit(data_deviations)

    return kmeans


def classify_data(desvi_EVI2, desvi_NDVI, labels):
    print('\t Classify data')
    groupC1 = []
    groupC2 = []
    groupC3 = []
    data_deviations = np.transpose(np.array([desvi_NDVI, desvi_EVI2, labels]))

    for dato in data_deviations:
        if dato[2] == c1:
            groupC1.append(dato[0:2])
        if dato[2] == c2:
            groupC2.append(dato[0:2])
        if dato[2] == c3:
            groupC3.append(dato[0:2])

    return data_deviations, groupC1, groupC2, groupC3


def get_refence_data(groupC3):
    minimo = min(np.transpose(groupC3)[0])
    for i in range(len(groupC3)):
        if groupC3[i:][0][0] == minimo:
            reference_data = groupC3[i]

    return reference_data


def get_distances(data_deviations, reference_data):
    distances = []
    for b in data_deviations:
        distancia = distance.euclidean(reference_data, b[0:2])
        distances.append(distancia)

    return distances


def get_distance_max_min(grupo):
    # Get the maximum of each group
    maximo = max(grupo)
    minimo = min(grupo)

    return maximo, minimo

# --------------------------- End of Kmeans method ---------------------------

# %% # ----------------------------- Damege Index ------------------------------


def damage_index_formula(m_sick, d_max, d_min, d_sample):
    di = (m_sick * (d_max - d_sample))/(d_max - d_min)

    return di


def get_DI_per_sample(distances, data_deviations, distanceMax_C1, distanceMin_C1,
                      distanceMax_C2, distanceMin_C2, distanceMax_C3,
                      distanceMin_C3):
    print("\t Get DIs")
    '''
    Change ifs according to kmeans 
    '''
    all_DI = []
    for i in range(len(distances)):
        # To groupC1 = c1
        if data_deviations[i, 2] == c1:
            m_sick = 9  # %
            di = damage_index_formula(m_sick, distanceMax_C1,
                            distanceMin_C1, distances[i])

        # To groupC2 = c2
        if data_deviations[i, 2] == c2:
            m_sick = 89  # %
            di = damage_index_formula(m_sick, distanceMax_C1,
                            distanceMin_C1, distances[i])

            if di == 0:
                di = 10

        # To sick group = c3
        if data_deviations[i, 2] == c3:
            m_sick = 100  # %
            di = damage_index_formula(m_sick, distanceMax_C1,
                            distanceMin_C1, distances[i])

        all_DI.append(di)

    return all_DI

# %% # ------------------------------- Training --------------------------------


def training_system(path_source, path_dest):
    print("\t --- TRAINING ---")
    deviations, desvi_EVI2, desvi_NDVI = get_desviations(path_source, path_dest)
    
    kmeans = kmeans_method(deviations)

    data_deviations, groupC1, groupC2, groupC3 = classify_data(desvi_EVI2,
                                                               desvi_NDVI,
                                                               kmeans.labels_)

    reference_data = get_refence_data(groupC3)

    disgroupC1 = get_distances(groupC1, reference_data)
    disgroupC2 = get_distances(groupC2, reference_data)
    disgroupC3 = get_distances(groupC3, reference_data)

    distanceMax_C1, distanceMin_C1 = get_distance_max_min(disgroupC1)
    distanceMax_C2, distanceMin_C2 = get_distance_max_min(disgroupC2)
    distanceMax_C3, distanceMin_C3 = get_distance_max_min(disgroupC3)

    distances = get_distances(data_deviations, reference_data)
    all_DI = get_DI_per_sample(distances, data_deviations,
                               distanceMax_C1, distanceMin_C1,
                               distanceMax_C2, distanceMin_C2,
                               distanceMax_C3, distanceMin_C3)

    return (all_DI, kmeans, reference_data, distanceMax_C1, distanceMin_C1,
            distanceMax_C2, distanceMin_C2, distanceMax_C3, distanceMin_C3,
            desvi_NDVI, desvi_EVI2)


# %% # ------------------------------- New Data --------------------------------
def upgrade_reference_distance(distanceMax_C1_p, distanceMin_C1_p, distanceMax_C2_p,
                               distanceMin_C2_p, distanceMax_C3_p, distanceMin_C3_p,
                               _distanceMax_C1, _distanceMin_C1, _distanceMax_C2,
                               _distanceMin_C2, _distanceMax_C3, _distanceMin_C3):

    distanceMax_C1 = distanceMax_C1_p if distanceMax_C1_p > _distanceMax_C1 else _distanceMax_C1
    distanceMin_C1 = distanceMin_C1_p if distanceMin_C1_p < _distanceMin_C1 else _distanceMin_C1
    distanceMax_C2 = distanceMax_C2_p if distanceMax_C2_p > _distanceMax_C2 else _distanceMax_C2
    distanceMin_C2 = distanceMin_C2_p if distanceMin_C2_p < _distanceMin_C2 else _distanceMin_C2
    distanceMax_C3 = distanceMax_C3_p if distanceMax_C3_p > _distanceMax_C3 else _distanceMax_C3
    distanceMin_C3 = distanceMin_C3_p if distanceMin_C3_p < _distanceMin_C3 else _distanceMin_C3

    return (distanceMax_C1, distanceMin_C1, distanceMax_C2, distanceMin_C2,
            distanceMax_C3, distanceMin_C3)


def new_data_system(path_source, path_dest, kmeans, reference_data_p,
                    distanceMax_C1_p, distanceMin_C1_p, distanceMax_C2_p,
                    distanceMin_C2_p, distanceMax_C3_p, distanceMin_C3_p,):
    print("\t --- NEW IMAGES ---")
    deviations, desvi_EVI2, desvi_NDVI = get_desviations(
        path_source, path_dest)
    _labels = kmeans.predict(deviations)

    data_deviations, groupC1, groupC2, groupC3 = classify_data(desvi_EVI2,
                                                               desvi_NDVI,
                                                               _labels)
    # Get new reference data
    _reference_data = get_refence_data(groupC3)
    reference_data = reference_data_p if reference_data_p[0] < _reference_data[0] else _reference_data

    # Get distances per group
    disgroupC1 = get_distances(groupC1, reference_data)
    disgroupC2 = get_distances(groupC2, reference_data)
    disgroupC3 = get_distances(groupC3, reference_data)

    # Upgrade distances
    _distanceMax_C1, _distanceMin_C1 = get_distance_max_min(disgroupC1)
    _distanceMax_C2, _distanceMin_C2 = get_distance_max_min(disgroupC2)
    _distanceMax_C3, _distanceMin_C3 = get_distance_max_min(disgroupC3)

    (distanceMax_C1, distanceMin_C1, distanceMax_C2, distanceMin_C2,
        distanceMax_C3, distanceMin_C3) = upgrade_reference_distance(
                                   distanceMax_C1_p, distanceMin_C1_p,
                                   distanceMax_C2_p, distanceMin_C2_p,
                                   distanceMax_C3_p, distanceMin_C3_p,
                                   _distanceMax_C1, _distanceMin_C1,
                                   _distanceMax_C2, _distanceMin_C2,
                                   _distanceMax_C3, _distanceMin_C3)

    # Get all distances
    distances = get_distances(data_deviations, reference_data)

    all_DI = get_DI_per_sample(distances, data_deviations,
                               distanceMax_C1, distanceMin_C1,
                               distanceMax_C2, distanceMin_C2,
                               distanceMax_C3, distanceMin_C3)

    return (all_DI, _labels, reference_data, distanceMax_C1,
            distanceMin_C1, distanceMax_C2, distanceMin_C2, distanceMax_C3,
            distanceMin_C3, desvi_NDVI, desvi_EVI2)

# %% # ------------------------------- Main menu -------------------------------


def menu():
    run = 0
    while run != 'n':
        string_menu = '''
        Menú
                 
        1. Entrenamiento
        2. Pruebas
        3. Comprobación
        
        Opción: '''

        option = input(string_menu)

        if option == '1':
            (training_DI, kmeans, reference_data, distanceMax_C1,
             distanceMin_C1, distanceMax_C2, distanceMin_C2, distanceMax_C3,
             distanceMin_C3, desvi_NDVI_training, desvi_EVI2_training) = training_system(PATH_TRAINING_IMG, PATH_TRAINING_RESULTS)

        elif option == '2':
            (test_DI, test_labels, reference_data, distanceMax_C1_test,
             distanceMin_C1_test, distanceMax_C2_test, distanceMin_C2_test,
             distanceMax_C3_test, distanceMin_C3_test, desvi_NDVI_test,
             desvi_EVI2_test) = new_data_system(PATH_TESTING_IMG, PATH_TESTING_RESULTS,
                                                kmeans, reference_data,
                                                distanceMax_C1, distanceMin_C1,
                                                distanceMax_C2, distanceMin_C2,
                                                distanceMax_C3, distanceMin_C3)

        elif option == '3':
            (verification_DI, verification_labels, reference_data_ver,
             distanceMax_C1_ver, distanceMin_C1_ver, distanceMax_C2_ver,
             distanceMin_C2_ver, distanceMax_C3_ver, distanceMin_C3_ver,
             desvi_NDVI_ver, desvi_EVI2_ver) = new_data_system(PATH_VERIFICATION_IMG, PATH_VERIFICATION_RESULTS,
                                                               kmeans, reference_data,
                                                               distanceMax_C1_test, distanceMin_C1_test,
                                                               distanceMax_C2_test, distanceMin_C2_test,
                                                               distanceMax_C3_test, distanceMin_C3_test)

        else:
            print("Opción incorrecta")

        plot = input(''' -> ¿Otra opcion? [y/n]: ''')

        if plot == 'n':
            mx = 15
            my = 6
            '''
            Cambiar colores segun etiquetas
            '''
            colores = ['green', 'red', 'blue']

            ######################## GRAFICA ENTRENAMIENTO ########################

            color_training = []
            C = kmeans.cluster_centers_

            plt.figure(figsize=(mx, my))

            for row in kmeans.labels_: color_training.append(colores[row])
            plt.scatter(desvi_NDVI_training, desvi_EVI2_training, c=color_training, s=11)

            plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)

            plt.scatter([], [], c='blue',  s=11, label='C1')
            plt.scatter([], [], c='green', s=11, label='C2')
            plt.scatter([], [], c='red',   s=11, label='C3')
            plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
            plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
            plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
            plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
            plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
            plt.title("Training: Standard deviation NDVI vs EVI2", fontsize=15)
            plt.legend()
            plt.grid()
            plt.tick_params(labelsize=15)
            plt.show()
            plt.savefig('kmeansEntrenamiento_eng')

            plt.figure(figsize=(mx, my))
            plt.scatter(desvi_NDVI_training[0:30],  desvi_EVI2_training[0:30],  c="blue",  s=11)
            plt.scatter(desvi_NDVI_training[30:39], desvi_EVI2_training[30:39], c="green", s=11)
            plt.scatter(desvi_NDVI_training[39:81], desvi_EVI2_training[39:81], c="red",   s=11)

            plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)

            plt.scatter([], [], c='blue',  s=11, label='C1')
            plt.scatter([], [], c='green', s=11, label='C2')
            plt.scatter([], [], c='red',   s=11, label='C3')
            plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
            plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
            plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
            plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
            plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
            plt.title("Before training: Standard deviation NDVI vs EVI2", fontsize=15)
            plt.tick_params(labelsize=15)
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig('Entrenamiento_antes_eng')

            # ---------------------------------------------------------------------
            plt.figure(figsize=(mx, my))
            plt.plot(training_DI, 'g')
            plt.plot(training_DI, 'ro')
            plt.xlabel('Sample', fontsize=15)
            plt.ylabel('Damage Index [%]', fontsize=15)
            plt.title("Training: Damage Index by sample", fontsize=15)
            plt.tick_params(labelsize=15)
            plt.show()
            plt.savefig('danioEntrenamiento_eng')

            if ('test_DI' and 'test_labels' and 'reference_data' and 'distanceMax_C1_test' and
                'distanceMin_C1_test' and 'distanceMax_C2_test' and 'distanceMin_C2_test' and
                'distanceMax_C3_test' and 'distanceMin_C3_test' and 'desvi_NDVI_test' and
                    'desvi_EVI2_test') in locals():

                ########################### GRAFICA PRUEBAS ###########################

                plt.figure(figsize=(mx, my))
                plt.scatter(desvi_NDVI_training, desvi_EVI2_training, c=color_training, s=11)
                plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)

                color_test = []
                for row in test_labels: color_test.append(colores[row])
                plt.scatter(desvi_NDVI_test, desvi_EVI2_test, c=color_test, s=40, marker='^')

                plt.scatter([], [], c='blue',  s=11, label='C1')
                plt.scatter([], [], c='green', s=11, label='C2')
                plt.scatter([], [], c='red',   s=11, label='C3')
                plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
                plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
                plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
                plt.scatter([], [], marker='^', c='blue',  s=40,  label='C1 Test data')
                plt.scatter([], [], marker='^', c='green', s=40,  label='C2 Test data')
                plt.scatter([], [], marker='^', c='red',   s=40,  label='C3 Test data')
                plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
                plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
                plt.title("Testing: Standard deviation NDVI vs EVI2", fontsize=15)
                plt.legend()
                plt.tick_params(labelsize=15)
                plt.grid()
                plt.show()
                plt.savefig('kmeansPruebas_eng')
                # ---------------------------------------------------------------------
                plt.figure(figsize=(mx, my))
                plt.plot(test_DI, 'g')
                plt.plot(test_DI, 'ro')
                plt.xlabel('Sample', fontsize=15)
                plt.ylabel('Damage Index [%]', fontsize=15)
                plt.title("Testing: Damage Index by sample", fontsize=15)
                plt.tick_params(labelsize=15)
                plt.show()
                plt.savefig('danioPruebas_eng')

                ####################### GRAFICA ANTES DE KMEANS #######################

                plt.figure(figsize=(mx, my))
                plt.scatter(
                    desvi_NDVI_training[0:30], desvi_EVI2_training[0:30], c="blue", s=11)
                plt.scatter(
                    desvi_NDVI_training[30:39], desvi_EVI2_training[30:39], c="green", s=11)
                plt.scatter(
                    desvi_NDVI_training[39:81], desvi_EVI2_training[39:81], c="red", s=11)

                plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)

                plt.scatter(desvi_NDVI_test[0:8],   desvi_EVI2_test[0:8],   c="blue",  s=40, marker='^')
                plt.scatter(desvi_NDVI_test[8:11],  desvi_EVI2_test[8:11],  c="green", s=40, marker='^')
                plt.scatter(desvi_NDVI_test[11:23], desvi_EVI2_test[11:23], c="red",   s=40, marker='^')

                plt.scatter([], [], c='blue',  s=11, label='C1')
                plt.scatter([], [], c='green', s=11, label='C2')
                plt.scatter([], [], c='red',   s=11, label='C3')
                plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
                plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
                plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
                plt.scatter([], [], marker='^', c='blue',  s=40, label='C1 Test data')
                plt.scatter([], [], marker='^', c='green', s=40, label='C2 Test data')
                plt.scatter([], [], marker='^', c='red',   s=40, label='C3 Test data')
                plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
                plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
                plt.title("Before kmeans (training + testing): Standard deviation NDVI vs EVI2", fontsize=15)
                plt.legend()
                plt.tick_params(labelsize=15)
                plt.grid()
                plt.show()
                plt.savefig('Entrenamiento_y_pruebas_antes_eng')

                ###################### GRAFICA DESPUES DE KMEANS ######################

                plt.figure(figsize=(mx, my))
                plt.scatter(desvi_NDVI_training, desvi_EVI2_training, c=color_training, s=11)

                plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)

                plt.scatter(desvi_NDVI_test, desvi_EVI2_test, c=color_test, s=40, marker='^')

                plt.scatter([], [], c='blue',  s=11, label='C1')
                plt.scatter([], [], c='green', s=11, label='C2')
                plt.scatter([], [], c='red',   s=11, label='C3')
                plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
                plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
                plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
                plt.scatter([], [], marker='^', c='blue',  s=40, label='C1 Test data')
                plt.scatter([], [], marker='^', c='green', s=40, label='C2 Test data')
                plt.scatter([], [], marker='^', c='red',   s=40, label='C3 Test data')
                plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
                plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
                plt.title("After kmeans (training + testing): Standard deviation NDVI vs EVI2", fontsize=15)
                plt.legend()
                plt.tick_params(labelsize=15)
                plt.grid()
                plt.show()
                plt.savefig('Entrenamiento_y_pruebas_despues_eng')

            if ('verification_DI' and 'verification_labels' and 'reference_data_ver' and
                'distanceMax_C1_ver' and 'distanceMin_C1_ver' and 'distanceMax_C2_ver' and
                'distanceMin_C2_ver' and 'distanceMax_C3_ver' and 'distanceMin_C3_ver' and
                    'desvi_NDVI_ver', 'desvi_EVI2_ver') in locals():
                ######################## GRAFICA VERIFICACIÓN ########################

                plt.figure(figsize=(mx, my))
                plt.scatter(desvi_NDVI_training, desvi_EVI2_training, c=color_training, s=11)
                plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=100)
                plt.scatter(desvi_NDVI_ver, desvi_EVI2_ver, c=color_test, s=11)

                color_verification = []
                for row in verification_labels: color_verification.append(colores[row])
                plt.scatter(desvi_NDVI_ver, desvi_EVI2_ver, c=color_verification, s=40, marker='^')

                plt.scatter([], [], c='blue',  s=11, label='C1')
                plt.scatter([], [], c='green', s=11, label='C2')
                plt.scatter([], [], c='red',   s=11, label='C3')
                plt.scatter([], [], marker='*', c='blue',  s=100, label='C1 Center')
                plt.scatter([], [], marker='*', c='green', s=100, label='C2 Center')
                plt.scatter([], [], marker='*', c='red',   s=100, label='C3 Center')
                plt.scatter([], [], marker='^', c='blue',  s=40, label='C1 Verification data')
                plt.scatter([], [], marker='^', c='green', s=40, label='C2 Verification data')
                plt.scatter([], [], marker='^', c='red',   s=40, label='C3 Verification data')
                plt.xlabel("Standard deviations in the NDVI histograms ", fontsize=15)
                plt.ylabel("Standard deviations in the EVI2 histograms", fontsize=15)
                plt.title("Verificatio: Standard deviation NDVI vs EVI2", fontsize=15)
                plt.legend()
                plt.tick_params(labelsize=15)
                plt.grid()
                plt.show()
                plt.savefig('kmeansComprobacion_eng')
                # ---------------------------------------------------------------------
                plt.figure(figsize=(mx, my))
                plt.plot(verification_DI, 'g')
                plt.plot(verification_DI, 'ro')
                plt.xlabel('Sample', fontsize=15)
                plt.ylabel('Damage Index [%]', fontsize=15)
                plt.title("Verification: Damage Index by sample", fontsize=15)
                plt.tick_params(labelsize=15)
                plt.show()
                plt.savefig('danioComprobacion_eng')

        run = plot

# %%


menu()

elapsed = time.time() - t
print()
print()
print(elapsed)
