#include "dqn.cuh"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern void checkCudaError(cudaError_t error, const char* message);

// Kernel para calcular Q-values objetivo
__global__ void compute_target_q_values(float* rewards, float* next_q_values,
                                       bool* dones, float* target_q,
                                       float gamma, int batch_size, int action_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Encontrar máximo Q-value para siguiente estado
        float max_next_q = next_q_values[idx * action_dim];
        for (int a = 1; a < action_dim; a++) {
            float q = next_q_values[idx * action_dim + a];
            if (q > max_next_q) {
                max_next_q = q;
            }
        }

        // Calcular target: r + gamma * max_Q(s', a') si no es terminal
        target_q[idx] = rewards[idx] + (dones[idx] ? 0.0f : gamma * max_next_q);
    }
}

// Kernel para calcular loss y gradientes (solo para acciones tomadas)
__global__ void compute_dqn_loss(float* q_values, int* actions, float* target_q,
                                float* loss_grad, int batch_size, int action_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int action = actions[idx];

        // Calcular gradiente solo para la acción tomada
        for (int a = 0; a < action_dim; a++) {
            if (a == action) {
                float q = q_values[idx * action_dim + a];
                loss_grad[idx * action_dim + a] = 2.0f * (q - target_q[idx]) / batch_size;
            } else {
                loss_grad[idx * action_dim + a] = 0.0f;
            }
        }
    }
}

// Constructor DQN
DQN::DQN(int state_dim, int action_dim, int* hidden_dims, int num_hidden_layers,
         float learning_rate, float gamma, float epsilon, float epsilon_min,
         float epsilon_decay, int batch_size, int replay_capacity,
         int target_update_freq) {

    this->state_dim = state_dim;
    this->action_dim = action_dim;
    this->gamma = gamma;
    this->epsilon = epsilon;
    this->epsilon_min = epsilon_min;
    this->epsilon_decay = epsilon_decay;
    this->learning_rate = learning_rate;
    this->batch_size = batch_size;
    this->target_update_freq = target_update_freq;
    this->steps_done = 0;

    // Crear redes neuronales
    policy_net = new NeuralNetwork(state_dim, hidden_dims, num_hidden_layers, action_dim);
    target_net = new NeuralNetwork(state_dim, hidden_dims, num_hidden_layers, action_dim);

    // Copiar pesos iniciales a target network
    target_net->copy_weights_from(policy_net);

    // Crear replay buffer
    replay_buffer = new ReplayBuffer(replay_capacity, state_dim);

    // Crear generador de números aleatorios
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, time(NULL));
}

// Destructor
DQN::~DQN() {
    delete policy_net;
    delete target_net;
    delete replay_buffer;
    curandDestroyGenerator(rng);
}

// Seleccionar acción usando epsilon-greedy
int DQN::select_action(float* state) {
    float random_val = (float)rand() / RAND_MAX;

    if (random_val < epsilon) {
        // Exploración: acción aleatoria
        return rand() % action_dim;
    } else {
        // Explotación: acción greedy
        return select_greedy_action(state);
    }
}

// Seleccionar acción greedy
int DQN::select_greedy_action(float* state) {
    // Copiar estado a GPU
    float* d_state;
    checkCudaError(cudaMalloc(&d_state, state_dim * sizeof(float)),
                  "Allocating state for action selection");
    checkCudaError(cudaMemcpy(d_state, state, state_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying state for action selection");

    // Alocar salida
    float* d_q_values;
    checkCudaError(cudaMalloc(&d_q_values, action_dim * sizeof(float)),
                  "Allocating q_values for action selection");

    // Forward pass
    policy_net->forward(d_state, d_q_values, 1);

    // Copiar Q-values a host
    float* h_q_values = new float[action_dim];
    checkCudaError(cudaMemcpy(h_q_values, d_q_values,
                             action_dim * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Copying q_values from GPU");

    // Encontrar acción con máximo Q-value
    int best_action = 0;
    float max_q = h_q_values[0];
    for (int a = 1; a < action_dim; a++) {
        if (h_q_values[a] > max_q) {
            max_q = h_q_values[a];
            best_action = a;
        }
    }

    // Limpiar
    cudaFree(d_state);
    cudaFree(d_q_values);
    delete[] h_q_values;

    return best_action;
}

// Almacenar experiencia
void DQN::store_experience(float* state, int action, float reward,
                          float* next_state, bool done) {
    replay_buffer->add(state, action, reward, next_state, done);
}

// Entrenar con un batch
void DQN::train_step() {
    if (!replay_buffer->can_sample(batch_size)) {
        return;
    }

    // Alocar buffers en GPU para el batch
    float* d_states;
    int* d_actions;
    float* d_rewards;
    float* d_next_states;
    bool* d_dones;

    checkCudaError(cudaMalloc(&d_states, batch_size * state_dim * sizeof(float)),
                  "Allocating batch states");
    checkCudaError(cudaMalloc(&d_actions, batch_size * sizeof(int)),
                  "Allocating batch actions");
    checkCudaError(cudaMalloc(&d_rewards, batch_size * sizeof(float)),
                  "Allocating batch rewards");
    checkCudaError(cudaMalloc(&d_next_states, batch_size * state_dim * sizeof(float)),
                  "Allocating batch next_states");
    checkCudaError(cudaMalloc(&d_dones, batch_size * sizeof(bool)),
                  "Allocating batch dones");

    // Muestrear batch del replay buffer
    replay_buffer->sample(batch_size, d_states, d_actions, d_rewards,
                         d_next_states, d_dones);

    // Calcular Q-values para next_states usando target network
    float* d_next_q_values;
    checkCudaError(cudaMalloc(&d_next_q_values, batch_size * action_dim * sizeof(float)),
                  "Allocating next_q_values");
    target_net->forward(d_next_states, d_next_q_values, batch_size);

    // Calcular target Q-values
    float* d_target_q;
    checkCudaError(cudaMalloc(&d_target_q, batch_size * sizeof(float)),
                  "Allocating target_q");

    int blocks = (batch_size + 255) / 256;
    compute_target_q_values<<<blocks, 256>>>(d_rewards, d_next_q_values, d_dones,
                                            d_target_q, gamma, batch_size,
                                            action_dim);
    checkCudaError(cudaGetLastError(), "Computing target Q-values");

    // Forward pass en policy network para obtener Q-values actuales
    float* d_q_values;
    checkCudaError(cudaMalloc(&d_q_values, batch_size * action_dim * sizeof(float)),
                  "Allocating q_values");
    policy_net->forward(d_states, d_q_values, batch_size);

    // Calcular gradientes del loss
    float* d_loss_grad;
    checkCudaError(cudaMalloc(&d_loss_grad, batch_size * action_dim * sizeof(float)),
                  "Allocating loss gradient");

    compute_dqn_loss<<<blocks, 256>>>(d_q_values, d_actions, d_target_q,
                                     d_loss_grad, batch_size, action_dim);
    checkCudaError(cudaGetLastError(), "Computing DQN loss");

    // Backward pass para actualizar policy network
    // Nota: En lugar de usar target como en MSE simple, usamos el gradiente calculado
    // Para eso, modificamos temporalmente el backward para aceptar gradientes directos
    // Por simplicidad, usamos el approach de crear un "target" artificial
    float* d_q_targets;
    checkCudaError(cudaMalloc(&d_q_targets, batch_size * action_dim * sizeof(float)),
                  "Allocating q_targets");

    // Copiar q_values actuales como base
    checkCudaError(cudaMemcpy(d_q_targets, d_q_values,
                             batch_size * action_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice),
                  "Copying q_values to targets");

    // Actualizar solo los Q-values de las acciones tomadas
    // (Esto se hace en CPU por simplicidad)
    float* h_q_targets = new float[batch_size * action_dim];
    float* h_target_q = new float[batch_size];
    int* h_actions = new int[batch_size];

    checkCudaError(cudaMemcpy(h_q_targets, d_q_targets,
                             batch_size * action_dim * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Copying q_targets to host");
    checkCudaError(cudaMemcpy(h_target_q, d_target_q,
                             batch_size * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Copying target_q to host");
    checkCudaError(cudaMemcpy(h_actions, d_actions,
                             batch_size * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "Copying actions to host");

    for (int i = 0; i < batch_size; i++) {
        h_q_targets[i * action_dim + h_actions[i]] = h_target_q[i];
    }

    checkCudaError(cudaMemcpy(d_q_targets, h_q_targets,
                             batch_size * action_dim * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "Copying modified q_targets to GPU");

    // Backward pass
    policy_net->backward(d_q_targets, d_q_values, learning_rate, batch_size);

    // Limpiar
    cudaFree(d_states);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_next_states);
    cudaFree(d_dones);
    cudaFree(d_next_q_values);
    cudaFree(d_target_q);
    cudaFree(d_q_values);
    cudaFree(d_loss_grad);
    cudaFree(d_q_targets);
    delete[] h_q_targets;
    delete[] h_target_q;
    delete[] h_actions;

    steps_done++;
}

// Actualizar red objetivo
void DQN::update_target_network() {
    target_net->copy_weights_from(policy_net);
}

// Decay epsilon
void DQN::decay_epsilon() {
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}

// ============================================================================
// CartPole Environment Implementation
// ============================================================================

CartPoleEnv::CartPoleEnv() {
    gravity = 9.8f;
    mass_cart = 1.0f;
    mass_pole = 0.1f;
    total_mass = mass_cart + mass_pole;
    length = 0.5f;  // mitad de la longitud del péndulo
    pole_mass_length = mass_pole * length;
    force_mag = 10.0f;
    tau = 0.02f;  // 20 ms

    theta_threshold = 12.0f * M_PI / 180.0f;  // 12 grados
    x_threshold = 2.4f;

    max_steps = 500;
    steps = 0;

    // Estado inicial aleatorio
    x = 0.0f;
    x_dot = 0.0f;
    theta = 0.0f;
    theta_dot = 0.0f;
}

void CartPoleEnv::reset(float* state) {
    // Reset con valores aleatorios pequeños
    x = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    x_dot = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    theta = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    theta_dot = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    steps = 0;

    get_state(state);
}

float CartPoleEnv::step(int action, float* next_state, bool* done) {
    // Aplicar fuerza (0 = izquierda, 1 = derecha)
    float force = (action == 1) ? force_mag : -force_mag;

    // Calcular aceleraciones usando las ecuaciones de movimiento
    float costheta = cosf(theta);
    float sintheta = sinf(theta);

    float temp = (force + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass;
    float theta_acc = (gravity * sintheta - costheta * temp) /
                     (length * (4.0f/3.0f - mass_pole * costheta * costheta / total_mass));
    float x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass;

    // Integración Euler
    x += tau * x_dot;
    x_dot += tau * x_acc;
    theta += tau * theta_dot;
    theta_dot += tau * theta_acc;

    steps++;

    // Verificar si terminó el episodio
    *done = (fabs(x) > x_threshold) ||
            (fabs(theta) > theta_threshold) ||
            (steps >= max_steps);

    // Recompensa: +1 por cada paso que se mantiene balanceado
    float reward = *done ? 0.0f : 1.0f;

    get_state(next_state);

    return reward;
}

void CartPoleEnv::get_state(float* state) {
    state[0] = x;
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;
}
