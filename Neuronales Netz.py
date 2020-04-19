#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy
#importieren der Bibliothek scipy.special für die Sigmoid Funktion expi()
import scipy.special
#Bibliothek zum Plotten von Matrizen
import matplotlib
#Sicherstellen, dass Plots im Programm und nicht in einem externen Fenster geöffnet werden
get_ipython().run_line_magic('matplotlib', 'inline')

#Klasse Neuronales Netz definieren
class neuralNetwork:
    
    #Neuronales Netzwerk initialisieren
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Anzahl von Knoten in Input, Hidden und Qutputlayer festlegen
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #Erzeugung der Verkünpfungsgewichte. (Gewichte werden mit wih und who verlinkt)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #Die Sigmoid Funktion wird als Aktivierungsfunktion festeglegt
        self.activation_function = lambda x: scipy.special.expit(x)
                    
        #Lernrate
        self.lr = learningrate
        pass
    
    # trainieren des Neuronalen Netzwerks
    def train(self, inputs_list, targets_list):
        
        #Input Liste in Matrix Konvertieren
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #Berechne die Signale die in die Hidden Layers gehen            
        hidden_inputs = numpy.dot(self.wih, inputs)
        #Berechne die Signale aus den Hidden Layers
        hidden_outputs = self.activation_function(hidden_inputs)
                    
        #Berechne die Signale die in das Ouput Layer gehen
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #Berechne die Signale die aus dem Output Layer 
        final_outputs = self.activation_function(final_inputs)
        
        #Fehler berechnen (target - actual)
        output_errors = targets - final_outputs
        
        #Fehler entsprechend den Verbindungsgewichten aufteilen und für jeden Knoten der Hidden Layern entsprechend zusammenfassen
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #Gewichte zwischen Hidden Layer und Output Layer anpassen
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 -final_outputs)), numpy.transpose(hidden_outputs))
        
        #Gewichte zwischen Input und Hidden Layers anpassen
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass


    
    #Abfragen des Neuronalen Netzwerks
    def query(self, inputs_list):
        #Input Liste in Matrix konvertieren
        inputs = numpy.array(inputs_list, ndmin=2).T
                    
        #Berechne die Signale die in die Hidden Layers gehen            
        hidden_inputs = numpy.dot(self.wih, inputs)
        #Berechne die Signale aus den Hidden Layers
        hidden_outputs = self.activation_function(hidden_inputs)
                    
        #Berechne die Signale die in das Ouput Layer gehen
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #Berechne die Signale die aus dem Output Layer 
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
#28x28
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#festlegen der Lernrate
learning_rate = 0.1

#Erstellen einer Instanz von neuralNetwork
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)    

#Trainingsdatensatz (csv) in eine Liste laden
training_data_file = open("PATH\\mnist_train_100_2.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#Trainieren des neuronalen Netzes

#Durch alle Datensätze des Trainingsdatensatzen gehen
for record in training_data_list:
    all_values = record.split(',')
    #Eingangsdaten auf den Bereich 0,01-1 Skalieren
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #Zielausgabewerte erzeugen (alle 0.01, außer Zielwert =0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    #all_values[0] wird als Zielwert für den Datensatz festgelegt
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass


# In[9]:


test_data_file = open("PATH\\MNIST\\Teilmengen\\mnist_test_10_2.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[20]:


all_values = test_data_list[0].split(',')
print(all_values[0])


# In[21]:


image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')


# In[22]:


n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)






