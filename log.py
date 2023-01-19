import math
import cv2
import numpy as np
import libreriaFiltros as lf
import matplotlib.pyplot as plt

def mascaraLaplacianoGaussiano(mascSize, sigma):
    limite = int((mascSize - 1) / 2)
    logResultado= 0.0
    mascara = np.zeros((mascSize, mascSize), dtype=np.float64)
    sum = 0.0
    
    for x in range(-limite, limite + 1):
        for y in range(-limite, limite + 1):
            a = 1 / (2 * (3.1416) * sigma**4)
            b = 2 - ((x**2 + y**2) / sigma**2)
            c = - ((x**2 + y**2) / (2 * sigma**2))
            d = math.exp(c)
            
            logResultado = a * b * d
            
            mascara[x + limite][y + limite] = logResultado
            sum += logResultado
              
    return mascara

def umbraladoLog(imagen, delta):
    
    largo, ancho = imagen.shape
    
    matrizCambioSigno = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    matrizDifDelta = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    imagenUmbralada = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    
    for i in range(1, largo - 1):
        for j in range(1, ancho - 1):
            vecinos = [(imagen[i-1][j-1], (i-1, j-1)), (imagen[i][j-1], (i, j-1)), (imagen[i+1][j-1], (i+1, j-1)), 
                       (imagen[i-1][j], (i-1, j)), (imagen[i+1][j], (i+1, j)),
                       (imagen[i-1][j+1], (i-1, j+1)), (imagen[i][j+1], (i, j+1)), (imagen[i+1][j+1], (i+1, j+1))] 
            
            val_actual = imagen[i][j]  
            signo_actual = signoNumero(val_actual)       
            
            for vecino in vecinos:
                val_vecino = vecino[0]
                signo_vecino = signoNumero(val_vecino)
                
                coords = vecino[1]
                coordX = coords[0]
                coordY = coords[1]
                
                matrizCambioSigno[coordX][coordY] = cambioSigno(signo_actual, signo_vecino)
                
                difDelta = val_vecino - val_actual
                matrizDifDelta[coordX][coordY] = cumplioDelta(delta, difDelta)
                
    for i in range(largo):
        for j in range(ancho):
            
            if (matrizCambioSigno[i][j] == 255 and matrizDifDelta[i][j] == 255):
                imagenUmbralada[i][j] = 255
            else:
                imagenUmbralada[i][j] = 0
             
                
    return imagenUmbralada, matrizCambioSigno, matrizDifDelta

def signoNumero(numero):
    signo = ""
    if numero < 0:
        signo = "-"
    else:
        signo = "+"  
    
    return signo   

def cambioSigno(signo_actual, signo_vecino )  :
    if signo_vecino == signo_actual:
        return 0
    else:
        return 255
    
def cumplioDelta(delta, dif):
    if abs(dif) > delta:
        return 255
    else:
        return 0

imgOriginal = cv2.imread('crayones.jpg')
imgEscalaGrises = lf.convertirEscalaGrisesNTSC(imgOriginal)

mascSize = int(input("Ingrese el tamaño de su mascara: "))
sigma = float(input("Ingrese el sigma: "))
delta = float(input("Ingrese su delta: "))

mascaraLog = mascaraLaplacianoGaussiano(mascSize, sigma)
print(mascaraLog)

matrizRelleno = lf.crearMatrizRelleno(imgEscalaGrises, mascSize) 

imgLog = lf.aplicarFiltro(imgEscalaGrises, matrizRelleno, mascaraLog, mascSize)

imgUmbralada, matrizCambioSigno, matrizDifDelta = umbraladoLog(imgLog, delta)


cv2.imshow("Imagen Original", imgOriginal)
cv2.imshow("Escala de grises", imgEscalaGrises)
cv2.imshow("Bordes expandidos", matrizRelleno)
#cv2.imshow("Imagen Log", imgLog)
cv2.imshow("Imagen de cambio de signo", matrizCambioSigno)
cv2.imshow("Imagen de diferencia delta", matrizDifDelta)
cv2.imshow("Imagen umbralada Log AND", imgUmbralada)
plt.imshow(imgLog, cmap=plt.cm.gray)
#print("fila: ", imgLog[100])
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

