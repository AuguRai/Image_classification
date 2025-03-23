import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import arff
from io import StringIO
import pandas as pd
import random

#Nuskaitomi duomenys (pradiniai testavimo ir mokymo duomenys buvo sujungti į vieną)
data =  pd.read_csv('combined_file.csv', sep=",")
duom_dydis = len(data)
#Atsitiktinai atrenkami duomenys testavimui, iš viso bus atrenkama 10% bendrų duomenų
selected_indices1 = random.sample(range(duom_dydis), round(duom_dydis * 0.1))
#Atrinkti duomenys priskiriami testavimo aibei
test_data = data.iloc[selected_indices1]
#Panaikinami testavimo duomenys iš pradinės duomenų aibės
data.drop(selected_indices1, inplace=True)
data.reset_index(drop=True, inplace=True)

#Atsitiktinai atrenkami duomenys validavimui, iš viso bus atrenkama 10% bendrų duomenų 
selected_indices2 = random.sample(range(len(data)), round(duom_dydis * 0.1))

#Atrinkti duomenys priskiriami validavimo aibei
valid_data = data.iloc[selected_indices2]
valid_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
#Panaikinami validavimo duomenys iš pradinės aibės, pradinė aibė bus mokymo duomenys
data.drop(selected_indices2, inplace=True)
data.reset_index(drop=True, inplace=True)

train_file = 'train_data1.csv'   
test_file = 'test_data1.csv'
valid_file = 'validation_data1.csv'

#Gautos aibės įrašomos į failus
data.to_csv(train_file)
test_data.to_csv(test_file)
valid_data.to_csv(valid_file)

#Nuskaitomi duomenys
train_x = pd.read_csv("train_data1.csv")
test_x = pd.read_csv("test_data1.csv")
valid_x = pd.read_csv("validation_data1.csv")

# Paimamos duomenų klasių reikšmės
train_y1 = train_x.iloc[:, 1]
test_y1 = test_x.iloc[:, 1]
valid_y1 = valid_x.iloc[:, 1]

# Panaikinami nereikalingi stulpeliai
train_x = train_x.drop(train_x.columns[1], axis=1)
test_x = test_x.drop(test_x.columns[1], axis=1)
valid_x = valid_x.drop(valid_x.columns[1], axis=1)

train_x = train_x.drop(train_x.columns[0], axis=1)
test_x = test_x.drop(test_x.columns[0], axis=1)
valid_x = valid_x.drop(valid_x.columns[0], axis=1)

# Patikrinama ar nėra tuščių reikšmių
print(train_x.isnull().any().sum())
print(test_x.isnull().any().sum())
print(valid_x.isnull().any().sum())

# Normalizavimas
train_x = train_x / 255.0
test_x = test_x / 255.0
valid_x = valid_x / 255.0


# Duomenys paruošiami neuroniniam tinklui mokyti tinkamu formatu (28x28 matrica)
train_x = train_x.to_numpy().reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.to_numpy().reshape((test_x.shape[0], 28, 28, 1))
valid_x = valid_x.to_numpy().reshape((valid_x.shape[0], 28, 28, 1))

# Tikrosios klasių reikšmės paruošiamos mokymui
train_y = keras.utils.to_categorical(train_y1)
test_y = keras.utils.to_categorical(test_y1)
valid_y = keras.utils.to_categorical(valid_y1)

# Neuroninio tinklo modelis
model2 = tf.keras.Sequential([
    # Konvoliucijos sluoksnis
    tf.keras.layers.Conv2D(filters=128, kernel_size=2, activation="relu", input_shape=(28, 28, 1)),
    # Normalizavimas
    tf.keras.layers.BatchNormalization(),
    # Išmetimo sluoksnis. Mokymo metu kiekvienoje iteracijoje laikinai išmetami dalis neuronų.
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    # Atliekamas pooling'as
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    # Šiame sluoksnyje dvimačiai masyvai paverčiami į vienmatį masyvą
    tf.keras.layers.Flatten(),
    #Pilnai sujungtas sluoksnis
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    # Gaunamos kiekvienos klasės tikimybės
    tf.keras.layers.Dense(10, activation='softmax')
])
# Sukompiliuojamas neuroninio tinklo modelis
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2= model2.fit(train_x, train_y, batch_size=512, epochs=10, verbose=1, validation_data=(valid_x, valid_y))

# Įvertinami testavimo aibės rezultatai
evaluation2 = model2.evaluate(test_x, test_y)
print(f'Test accuracy : {evaluation2[1]:.3f}')
print(f'Test loss : {evaluation2[0]:.3f}')

#Tikrosios klasės
test_y_arg = np.argmax(test_y, axis = 1)
#Nuspėtos klasės
predicted_classes = np.argmax(model2.predict(test_x), axis = 1) 

# Klasifikavimo matrica
cmatrix = confusion_matrix(test_y_arg, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cmatrix, annot=True)
plt.savefig('confusion_matrix.png')

# Vaizduojamos pirmos 30 testavimo duomenų reikšmių
fig = pyplot.figure(figsize=(15, 5))
for i in range(30):
    ax = pyplot.subplot(3, 10, i + 1)
    ax.text(5, -5, f'True: {test_y_arg[i]}\nPredicted: {predicted_classes[i]}',
            color='black', fontsize=8)
    ax.imshow(test_x[i], cmap=pyplot.get_cmap('gray'))
    
pyplot.tight_layout()
pyplot.savefig("fashion_data.png")

# Tikslumo ir paklaidos grafikai
plt.figure(figsize=(12, 8))
plt.plot(history2.history['loss'], label='Loss')
plt.plot(history2.history['val_loss'], label='val_Loss')
plt.xlabel('Epocha')
plt.ylabel('Paklaida')
plt.legend(['Mokymo', 'Validavimo'])
plt.title('Paklaidos kitimas')

# Paskutinės paklaidos vaizdavimas
last_epoch2 = len(history2.history['loss']) - 1
plt.annotate(
    f'{history2.history["loss"][-1]:.3f}',
    xy=(last_epoch2, history2.history['loss'][-1]),
    xytext=(-20, 20),
    textcoords='offset points',
    ha='left',
    va='top'
)
plt.annotate(
    f'{history2.history["val_loss"][-1]:.3f}',
    xy=(last_epoch2, history2.history['val_loss'][-1]),
    xytext=(-20, 20),
    textcoords='offset points',
    ha='left',
    va='top'
)
plt.savefig('paklaida.png')

plt.figure(figsize=(12, 8))
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title("Tikslumo kitimas")
plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.legend(['Mokymo', 'Validavimo'])

# Paskutinio tikslumo vaizdavimas
plt.annotate(
    f'{history2.history["accuracy"][-1]:.3f}',
    xy=(last_epoch2, history2.history['accuracy'][-1]),
    xytext=(-20, -10),
    textcoords='offset points',
    ha='left',
    va='top'
)

plt.annotate(
    f'{history2.history["val_accuracy"][-1]:.3f}',
    xy=(last_epoch2, history2.history['val_accuracy'][-1]),
    xytext=(-20, -10),
    textcoords='offset points',
    ha='left',
    va='top'
)
plt.savefig('tikslumas.png')
