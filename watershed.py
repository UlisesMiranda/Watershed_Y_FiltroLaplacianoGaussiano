
import math
import cv2
import numpy as np
import libreriaFiltros as lf
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt


def algoritmoWaterShed(image):
    
    
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    
    return labels, -distance

imgOriginal = cv2.imread('globos.png')
imgOriginalBRG = imgOriginal[:, :, [2,1,0]]
imgEscalaGrises = lf.convertirEscalaGrisesNTSC(imgOriginal)

mascSize = int(input("Ingrese el tamaño de su mascara: "))
sigma = float(input("Ingrese el sigma: "))

mascaraGauss = lf.mascaraGaussiana(mascSize, sigma)
matrizRelleno = lf.crearMatrizRelleno(imgEscalaGrises, mascSize)
imgFiltroGaussiano = lf.aplicarFiltro(imgEscalaGrises, matrizRelleno, mascaraGauss, mascSize)

imgFiltroGaussiano = imgFiltroGaussiano.astype(np.uint8)

histograma = lf.obtenerHistograma(imgFiltroGaussiano)

umbralOtsu = lf.umbralAlgoritmoOTSU(histograma)
print("Umbral obtnido de Otsu: ", umbralOtsu)

imgUmbralizada = lf.umbralizarImagen(imgFiltroGaussiano, umbralOtsu)

imgLabels, imgDistance = algoritmoWaterShed(imgUmbralizada)

""" imgDistancias = imgDistancias.astype(np.uint8)
imgSeparacion = imgSeparacion.astype(np.uint8) """

cv2.imshow("Imagen Original", imgOriginal)
cv2.imshow("Escala de grises", imgEscalaGrises)
cv2.imshow("Bordes expandidos", matrizRelleno)
cv2.imshow("Filtro gaussiano", imgFiltroGaussiano)
cv2.imshow("Imagen Umbralizada", imgUmbralizada) 

cv2.waitKey()
cv2.destroyAllWindows()