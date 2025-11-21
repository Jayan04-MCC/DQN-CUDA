# Proyecto CUDA - DQN y Ejemplos

Implementación de Deep Q-Network (DQN) con CUDA para entrenamiento de agentes de reinforcement learning.

## 🎯 Objetivo del Proyecto

El agente DQN se entrena para **resolver el problema CartPole**: equilibrar un péndulo invertido sobre un carrito móvil.

### El Desafío
- **Problema**: Un carrito con un péndulo encima. El agente debe mover el carrito (izquierda/derecha) para mantener el péndulo vertical.
- **Estado**: 4 dimensiones (posición carrito, velocidad, ángulo péndulo, velocidad angular)
- **Acciones**: 2 opciones (mover izquierda o derecha)
- **Recompensa**: +1 por cada paso que el péndulo se mantiene balanceado

### Criterios de Éxito
- **Excelente**: Promedio ≥ 450 puntos en test (péndulo balanceado casi 500 pasos)
- **Buen progreso**: Promedio ≥ 200 puntos
- **Necesita más entrenamiento**: < 200 puntos

El episodio termina cuando:
1. El carrito se sale del área (`|x| > 2.4`)
2. El péndulo se inclina mucho (`|theta| > 12°`)
3. Se alcanzan 500 pasos (máximo)

## Estructura del Proyecto

### 📁 `include/` - Headers

#### `dqn.cuh`
- **Clase `DQN`**: Implementación completa del algoritmo Deep Q-Network
  - Policy network (Q-network) y Target network
  - Epsilon-greedy para exploración
  - Replay buffer para almacenar experiencias
  - Métodos: `select_action()`, `train_step()`, `update_target_network()`

- **Clase `CartPoleEnv`**: Ambiente de simulación CartPole
  - 4 estados: posición, velocidad, ángulo, velocidad angular
  - 2 acciones: mover izquierda o derecha
  - Métodos: `reset()`, `step()`

#### `neural_network.cuh`
- **Clase `NeuralNetwork`**: Red neuronal fully-connected en CUDA
  - Forward propagation en GPU
  - Backward propagation con gradientes
  - Activación ReLU
  - Actualización de pesos con SGD
  - Métodos: `forward()`, `backward()`, `copy_weights_from()`

- **Kernels CUDA**: Operaciones de red neuronal paralelizadas
  - `fully_connected_forward`: Multiplicación matriz-vector
  - `relu_activation`: Función de activación
  - `relu_backward`: Gradiente de ReLU
  - `fully_connected_backward`: Backpropagation
  - `update_weights`: Actualización de parámetros

#### `replay_buffer.cuh`
- **Clase `ReplayBuffer`**: Buffer circular para experiencias
  - Almacena transiciones (state, action, reward, next_state, done)
  - Datos en GPU para acceso rápido
  - Muestreo aleatorio de batches
  - Métodos: `add()`, `sample()`, `can_sample()`

### 📁 `src/` - Implementaciones

#### `main_dqn.cu` (Programa principal DQN)
- Loop de entrenamiento completo de 500 episodios
- Configuración: arquitectura 4→64→64→2
- Métricas de progreso (reward promedio, epsilon, pasos)
- Fase de test sin exploración (10 episodios)
- Objetivo: Lograr 450+ puntos promedio (péndulo balanceado 90% del tiempo)
- Ejecutable: `dqn_train`

#### `dqn.cu`
Implementación de:
- Algoritmo DQN completo (Deep Q-Learning)
- Epsilon-greedy strategy (exploración vs explotación)
- Training step con batch sampling del replay buffer
- Target network update cada N episodios
- CartPole physics simulation (ecuaciones de movimiento del péndulo)
- Función Q(estado, acción) para predecir recompensas futuras

#### `neural_network.cu`
Implementación de:
- Inicialización de capas con Xavier/He
- Forward pass paralelo
- Backward pass con chain rule
- Copia de pesos entre redes (policy → target)

#### `replay_buffer.cu`
Implementación de:
- Buffer circular eficiente
- Gestión de memoria GPU
- Muestreo aleatorio uniforme
- Almacenamiento de experiencias

## Compilación

```bash
# Windows
compile_dqn.bat

# Linux/Mac
./compile_dqn.sh
```

## Ejecutables Generados

- **`cuda_hello`**: Programa de prueba básico de CUDA
- **`dqn_train`**: Entrenamiento de agente DQN en CartPole

## Dependencias

- CUDA Toolkit 12.8
- Compute Capability 5.0+ (MX110 compatible)
- CMake 3.18+
- cuRAND (generación aleatoria en GPU)

## Arquitectura DQN

```
Estado (4) → FC(64) → ReLU → FC(64) → ReLU → FC(2) → Q-values
```

- Learning rate: 0.001
- Gamma: 0.99
- Epsilon decay: 0.995
- Batch size: 32
- Replay buffer: 10,000 experiencias
