# NN_SIMPLE
Redes neuronales simple

El archivo `NNsimple-1_1.ipynb` es un cuaderno Jupyter que implementa una red neuronal simple en Python puro. El código se divide en varias secciones, que se describen a continuación.

## Librerías

El código comienza importando las librerías necesarias para el correcto funcionamiento del programa. En este caso, la única librería requerida es NumPy, que se utiliza para las operaciones matemáticas.

```python
import numpy as np
```

## Clase NeuralNetwork

Luego, se define la clase `NeuralNetwork`, que se encarga de la creación, entrenamiento y uso de la red neuronal.

### Inicialización

El método `__init__` inicializa la red neuronal. En este caso, la red neuronal consiste en una sola capa, y los pesos se inicializan aleatoriamente.

```python
def __init__(self, x, y):
    self.input      = x
    self.weights1   = np.random.rand(self.input.shape[1],4) 
    self.weights2   = np.random.rand(4,1)                 
    self.y          = y
    self.output     = np.zeros(self.y.shape)
```

### Función de activación

La función `sigmoid` es la función de activación utilizada en la red neuronal, y `sigmoid_derivative` es su derivada. Estas funciones se utilizan durante el entrenamiento de la red.

```python
def sigmoid(self, x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(self, x):
    return x * (1.0 - x)
```

### Entrenamiento

El método `train` entrena la red neuronal utilizando el algoritmo de retropropagación. Durante cada iteración del entrenamiento, los pesos de la red se actualizan en función del error de la red, que se calcula comparando la salida actual de la red con la salida esperada.

```python
def train(self, X, y):
    self.output = self.forward(X)
    self.backward(X, y, self.output)
```

### Predicción

Finalmente, el método `predict` se utiliza para predecir la salida de la red neuronal para una nueva entrada después de que la red ha sido entrenada.

```python
def predict(self, x):
    self.layer1 = self.sigmoid(np.dot(x, self.weights1))
    return self.sigmoid(np.dot(self.layer1, self.weights2))
```

## Uso de la red neuronal

Después de definir la clase `NeuralNetwork`, el código crea una instancia de esta clase y la entrena con algunos datos de entrada y salida.

```python
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)

for i in range(1500):
    nn.train(X, y)
```

Posteriormente, el programa imprime los pesos de la red neuronal después del entrenamiento y utiliza la red para predecir la salida para una nueva entrada.

```python
print(nn.weights1)
print(nn.weights2)

print(nn.predict(np.array([1, 0, 0])))
```

En resumen, este cuaderno Jupyter proporciona una implementación simple y clara de una red neuronal de una capa en Python puro. Puede ser útil para aprender los fundamentos de las redes neuronales y el algoritmo de retropropagación.
