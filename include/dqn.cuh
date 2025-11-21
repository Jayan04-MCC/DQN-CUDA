#ifndef DQN_CUH
#define DQN_CUH

#include "neural_network.cuh"
#include "replay_buffer.cuh"
#include <curand.h>

// Clase principal DQN
class DQN {
private:
    NeuralNetwork* policy_net;  // red de política (Q-network)
    NeuralNetwork* target_net;  // red objetivo
    ReplayBuffer* replay_buffer;

    int state_dim;
    int action_dim;
    float gamma;           // factor de descuento
    float epsilon;         // epsilon para epsilon-greedy
    float epsilon_min;
    float epsilon_decay;
    float learning_rate;
    int batch_size;
    int target_update_freq;
    int steps_done;

    curandGenerator_t rng;

public:
    DQN(int state_dim, int action_dim, int* hidden_dims, int num_hidden_layers,
        float learning_rate = 0.001f, float gamma = 0.99f,
        float epsilon = 1.0f, float epsilon_min = 0.01f, float epsilon_decay = 0.995f,
        int batch_size = 32, int replay_capacity = 10000, int target_update_freq = 10);

    ~DQN();

    // Seleccionar acción usando epsilon-greedy
    int select_action(float* state);

    // Seleccionar acción greedy (sin exploración)
    int select_greedy_action(float* state);

    // Almacenar experiencia
    void store_experience(float* state, int action, float reward,
                         float* next_state, bool done);

    // Entrenar con un batch
    void train_step();

    // Actualizar red objetivo
    void update_target_network();

    // Decay epsilon
    void decay_epsilon();

    // Getters
    float get_epsilon() { return epsilon; }
    int get_steps() { return steps_done; }
};

// Ambiente CartPole simple
class CartPoleEnv {
private:
    float x;            // posición del carrito
    float x_dot;        // velocidad del carrito
    float theta;        // ángulo del péndulo
    float theta_dot;    // velocidad angular del péndulo

    float gravity;
    float mass_cart;
    float mass_pole;
    float total_mass;
    float length;       // mitad de la longitud del péndulo
    float pole_mass_length;
    float force_mag;
    float tau;          // paso de tiempo

    int steps;
    int max_steps;

    float theta_threshold;
    float x_threshold;

public:
    CartPoleEnv();

    // Reset el ambiente
    void reset(float* state);

    // Realizar un paso
    float step(int action, float* next_state, bool* done);

    // Obtener estado actual
    void get_state(float* state);

    // Dimensiones
    static const int STATE_DIM = 4;
    static const int ACTION_DIM = 2;
};

#endif // DQN_CUH
