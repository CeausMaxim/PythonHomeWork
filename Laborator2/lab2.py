import numpy as np
import matplotlib.pyplot as plt

#Ex a
#Citiți imaginile din aceste fișiere și salvați-le într-un np.array (va avea dimensiunea 9x400x600).
img = np.array([np.load(r"images/car_{idx}.npy".format(idx=i)) for i in range(9)])
print(img)

#Ex b
#Calculați suma valorilor pixelilor tuturor imaginilor.
suma = np.sum(img)
print("Suma valorilor pixelilor tuturor imaginilor: ",suma)

#Ex c
#Calculați suma valorilor pixelilor pentru fiecare imagine în parte.
suma_part = np.sum(img, axis=(1,2))
print("Calculați suma valorilor pixelilor pentru fiecare imagine în parte:",suma_part)

#Ex d
#Afișați indexul imaginii cu suma maximă.
print("Indexul imaginii cu suma maximă:", np.argmax(np.sum(img, axis=(1,2))))

#Ex e
#Calculați imaginea medie și afișati-o.
from skimage import io
avg_img= np.mean(img, axis=0)
io.imshow(avg_img.astype(np.uint8))
io.show()

#Ex f
#Cu ajutorul funcției np.std(images_array), calculați deviația standard a imaginilor.
deviatia_standard = np.std(img)
print("Deviația standard a imaginilor:",deviatia_standard)

#Ex g
#Normalizați imaginile. (se scade imaginea medie și se împarte rezultatul la deviația standard)
norma = (img - avg_img) / np.std(img)
print("Normalizația:", norma)

#Ex h
#Decupați fiecare imagine, afișând numai liniile cuprinse între 200 și 300, respectiv coloanele cuprinse între 280 și 400.
cutting = img[:, 200:300, 280:400]
plt.imshow(img[7].astype(np.uint), cmap='gray')
plt.imshow(cutting[7].astype(np.uint), cmap='gray')