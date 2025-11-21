#ifndef REPLAY_BUFFER_CUH
#define REPLAY_BUFFER_CUH

#include <cuda_runtime.h>

// Estructura para una experiencia (transición)
struct Experience {
    float* state;       // estado actual
    int action;         // acción tomada
    float reward;       // recompensa recibida
    float* next_state;  // siguiente estado
    bool done;          // si el episodio terminó
};

// Clase para el replay buffer
class ReplayBuffer {
private:
    int capacity;
    int state_dim;
    int current_size;
    int position;

    // Buffers en GPU
    float* d_states;
    int* d_actions;
    float* d_rewards;
    float* d_next_states;
    bool* d_dones;

public:
    ReplayBuffer(int capacity, int state_dim);
    ~ReplayBuffer();

    // Agregar experiencia
    void add(float* state, int action, float reward, float* next_state, bool done);

    // Muestrear batch aleatorio
    void sample(int batch_size, float* states_out, int* actions_out,
                float* rewards_out, float* next_states_out, bool* dones_out);

    // Obtener tamaño actual
    int size() { return current_size; }
    bool can_sample(int batch_size) { return current_size >= batch_size; }
};

#endif // REPLAY_BUFFER_CUH
