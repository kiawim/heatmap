# Importation des modules
    
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import timeit
from PIL import Image
import numpy as np
from random import randint
from math import floor
import joblib
import os
from sewar.full_ref import vifp

# ignorer le FutureWarning de sklearn
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# commencer à compter le temps pris par l'algorithme
start = timeit.default_timer()


def learnform(ilots,carte,N,k):

    #X sera la liste contenant les arrays désignant les photos de cartes et Y contiendra les arrays de cartes avec ilots
    X=[[]]*N
    Y=[[]]*N

    image = Image.open(ilots)
    image2= Image.open(carte)
    # on prend les dimensions minimales pour rester dans la partie accessible de chaque image
    l1, h1 = image.size
    l2, h2 = image2.size
    l, h = (min(l1,l2), min(h1,h2))

    arr_ilots = np.array(image)
    arr_carte = np.array(image2)

    for i in range(N):
        # on prend un pixel au hasard en prenant une hauteur et une largeur au hasard (en évitant le dernier pixel de la rangée/colonne)
        rand_l = randint(0,l-2)
        rand_h = randint(0,h-2)
        # on prend des dimensions (hauteur et largeur) au hasard
        size_l = randint(1,l-rand_l-1)
        size_h = randint(1,h-rand_h-1)
        rectangle_ilots=arr_ilots[rand_h:rand_h+size_h, rand_l:rand_l+size_l]
        rectangle_carte=arr_carte[rand_h:rand_h+size_h, rand_l:rand_l+size_l]

        X[i]=rectangle_carte
        Y[i]=rectangle_ilots
    
    # chargement de la première image d'entrainement (entrée)
    
    image_1=X[0]
    #plt.imshow(image_1)
    #image_1.shape
    
    # traitement de l'image entrée
    image_1_1=image_1[:,:,0].ravel()
    #print(image_1_1.shape)
    image_1_2=image_1[:,:,1].ravel()
    image_1_3=image_1[:,:,2].ravel()
    image_1t=np.vstack((image_1_1,image_1_2,image_1_3))
    image_1f=image_1t.transpose()
    #image_1t.shape
    
    
    # chargement de la première image d'entrainement (resultat)
    
    image_2=Y[0]
    #plt.imshow(image_2)
    #image_2.shape
    
    # traitement de l'image résultat
    
    image_2_1=image_2[:,:,0].ravel()
    #print(image_2_1.shape)
    image_2_2=image_2[:,:,1].ravel()
    #print(image_2_2.shape)
    image_2_3=image_2[:,:,2].ravel()
    #print(image_2_3.shape)
    image_2t=np.vstack((image_2_1,image_2_2,image_2_3))
    image_2f=image_2t.transpose()
    #image_2t.shape

    
    # récupérer toutes les images d'entraînement 
    for i in range (1,N):
    
        # chargement de l'image d'entrainement (entrée)
        
        image_1=X[i]
        #plt.imshow(image_1)
        #image_1.shape
        
        # traitement de l'image entrée
        
        image_1_1=image_1[:,:,0].ravel()
        #print(image_1_1.shape)
        image_1_2=image_1[:,:,1].ravel()
        image_1_3=image_1[:,:,2].ravel()
        image_1t=np.vstack((image_1_1,image_1_2,image_1_3))
        image_1t=image_1t.transpose()
        #image_1t.shape
        image_1f=np.vstack((image_1f,image_1t))
        
        
        # chargement de l'image d'entrainement (resultat)
        
        image_2=Y[i]
        #plt.imshow(image_2)
        #image_2.shape
        
        
        # traitement de l'image résultat
        
        image_2_1=image_2[:,:,0].ravel()
        #print(image_2_1.shape)
        image_2_2=image_2[:,:,1].ravel()
        #print(image_2_2.shape)
        image_2_3=image_2[:,:,2].ravel()
        #print(image_2_3.shape)
        image_2t=np.vstack((image_2_1,image_2_2,image_2_3))
        image_2t=image_2t.transpose()
        #image_2t.shape
        image_2f=np.vstack((image_2f,image_2t))

        # imprimer les étapes pour voir où l'algorithme en est
        if (i%100==0) or (i==1):
            print(i)
    
    # Entrainement du modèle
    
    model=KNeighborsClassifier(n_neighbors=k)
    model=model.fit(image_1f,image_2f)
    
    # sauvegarder le modèle dans le fichier pour ne pas avoir à le relancer
    filename = "modèles/carteN"+str(N)+"k"+str(k)+".sav"  
    joblib.dump(model, filename)

    return model


def predictfrom(image,model):
    
    # chargement de l'image à traiter
    image_3=plt.imread(image)
    #plt.imshow(image_3)
    #image_3.shape
    
    
    # chargement de l'image à traiter
    image_3_1=image_3[:,:,0].ravel()
    image_3_2=image_3[:,:,1].ravel()
    image_3_3=image_3[:,:,2].ravel()
    image_3f=np.vstack((image_3_1,image_3_2,image_3_3))
    image_3f=image_3f.transpose()
    image_3f.shape
    
    # calcul du potentiel solaire
    print("Avant predict")
    image_r=model.predict(image_3f)
    print("Apres predict")

    image_r=image_r.reshape((image_3.shape[0],image_3.shape[1],image_3.shape[2]))

    # imprimer le runtime en secondes
    stop = timeit.default_timer()
    print('Runtime: ', stop - start)  

    plt.imshow(image_r)
    plt.savefig("résultats/cartesaintetienne_N"+str(N)+"k"+str(k)+".png")

    #image_r.shape

    return image_r #resultat image

ilots="saintetienne_icu.jpeg"
satellite="saintetienne_satellite.jpeg"
carte="saintetienne_carte.jpeg"
N=100
k=5

# vérifer si le modèle a déjà été entraîné, sinon on l'entraîne
entries = os.listdir('modèles/')
f= "carteN"+str(N)+"k"+str(k)+".sav"
pathf="modèles/"+f

# vérifer si un modèle a déjà été entraîné avec ces paramètres, si oui l'importer plutôt que de refaire un entraînement
if f in entries:
    model=joblib.load(f)
    newimage=model.predict(carte)
else:
    model=learnform(ilots,carte,N,k)
    newimage=predictfrom(carte,model)

# calculer la correlation entre l'image espérée et l'image produite
goal=plt.imread(ilots)

print("VIF: ", vifp(newimage,goal))


