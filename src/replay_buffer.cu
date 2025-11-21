#include "replay_buffer.cuh"
#include <iostream>
#include <cstdlib>
#include <ctime>

extern void checkCudaError(cudaError_t error, const char* message);

// Constructor
ReplayBuffer::ReplayBuffer(int capacity, int state_dim) {
    this->capacity = capacity;
    this->state_dim = state_dim;
    this->current_size = 0;
    this->position = 0;

    // Alocar buffers en GPU
    checkCudaError(cudaMalloc(&d_states, capacity * state_dim * sizeof(float)),
                  "Allocating states buffer");
    checkCudaError(cudaMalloc(&d_actions, capacity * sizeof(int)),
                  "Allocating actions buffer");
    checkCudaError(cudaMalloc(&d_rewards, capacity * sizeof(float)),
                  "Allocating rewards buffer");
    checkCudaError(cudaMalloc(&d_next_states, capacity * state_dim * sizeof(float)),
                  "Allocating next_states buffer");
    checkCudaError(cudaMalloc(&d_dones, capacity * sizeof(bool)),
                  "Allocating dones buffer");

    // Inicializar random seed
    srand(time(NULL));
}

// Destructor
ReplayBuffer::~ReplayBuffer() {
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_next_states);
    cudaFree(d_dones);
}

// Agregar experiencia
void ReplayBuffer::add(float* state, int action, float reward, float* next_state, bool done) {
    // Copiar datos a GPU en la posición circular
    int offset = position * state_dim;

    checkCudaError(cudaMemcpy(d_states + offset, state,
                             state_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying state to buffer");

    checkCudaError(cudaMemcpy(d_actions + position, &action,
                             sizeof(int),
                             cudaMemcpyHostToDevice),
                  "Copying action to buffer");

    checkCudaError(cudaMemcpy(d_rewards + position, &reward,
                             sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying reward to buffer");

    checkCudaError(cudaMemcpy(d_next_states + offset, next_state,
                             state_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying next_state to buffer");

    checkCudaError(cudaMemcpy(d_dones + position, &done,
                             sizeof(bool),
                             cudaMemcpyHostToDevice),
                  "Copying done to buffer");

    // Actualizar posición circular
    position = (position + 1) % capacity;
    if (current_size < capacity) {
        current_size++;
    }
}

// Muestrear batch aleatorio
void ReplayBuffer::sample(int batch_size, float* states_out, int* actions_out,
                         float* rewards_out, float* next_states_out, bool* dones_out) {
    // Generar índices aleatorios
    int* indices = new int[batch_size];
    for (int i = 0; i < batch_size; i++) {
        indices[i] = rand() % current_size;
    }

    // Buffers temporales en host
    float* h_states = new float[batch_size * state_dim];
    int* h_actions = new int[batch_size];
    float* h_rewards = new float[batch_size];
    float* h_next_states = new float[batch_size * state_dim];
    bool* h_dones = new bool[batch_size];

    // Copiar experiencias seleccionadas desde GPU a host
    for (int i = 0; i < batch_size; i++) {
        int idx = indices[i];
        int offset = idx * state_dim;

        // Copiar state
        checkCudaError(cudaMemcpy(h_states + i * state_dim,
                                 d_states + offset,
                                 state_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "Copying sampled state");

        // Copiar action
        checkCudaError(cudaMemcpy(h_actions + i,
                                 d_actions + idx,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "Copying sampled action");

        // Copiar reward
        checkCudaError(cudaMemcpy(h_rewards + i,
                                 d_rewards + idx,
                                 sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "Copying sampled reward");

        // Copiar next_state
        checkCudaError(cudaMemcpy(h_next_states + i * state_dim,
                                 d_next_states + offset,
                                 state_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "Copying sampled next_state");

        // Copiar done
        checkCudaError(cudaMemcpy(h_dones + i,
                                 d_dones + idx,
                                 sizeof(bool),
                                 cudaMemcpyDeviceToHost),
                      "Copying sampled done");
    }

    // Copiar batch completo a los buffers de salida (en GPU)
    checkCudaError(cudaMemcpy(states_out, h_states,
                             batch_size * state_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying batch states to output");

    checkCudaError(cudaMemcpy(actions_out, h_actions,
                             batch_size * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "Copying batch actions to output");

    checkCudaError(cudaMemcpy(rewards_out, h_rewards,
                             batch_size * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying batch rewards to output");

    checkCudaError(cudaMemcpy(next_states_out, h_next_states,
                             batch_size * state_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying batch next_states to output");

    checkCudaError(cudaMemcpy(dones_out, h_dones,
                             batch_size * sizeof(bool),
                             cudaMemcpyHostToDevice),
                  "Copying batch dones to output");

    // Limpiar
    delete[] indices;
    delete[] h_states;
    delete[] h_actions;
    delete[] h_rewards;
    delete[] h_next_states;
    delete[] h_dones;
}
