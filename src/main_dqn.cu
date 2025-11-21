#include <iostream>
#include <cuda_runtime.h>
#include "dqn.cuh"
#include <ctime>
#include <vector>
#include <numeric>

void checkCudaError(cudaError_t error, const char* message);

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No se encontraron dispositivos CUDA!" << std::endl;
        exit(1);
    }

    std::cout << "\n=== Informacion del Dispositivo CUDA ===" << std::endl;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Dispositivo " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memoria Global: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   DQN Training with CUDA" << std::endl;
    std::cout << "   CartPole Environment" << std::endl;
    std::cout << "========================================" << std::endl;

    // Mostrar información del dispositivo
    print_device_info();

    // Inicializar random seed
    srand(time(NULL));

    // Configuración del entrenamiento
    const int NUM_EPISODES = 500;
    const int MAX_STEPS_PER_EPISODE = 500;
    const int TARGET_UPDATE_FREQ = 10;
    const int PRINT_FREQ = 10;

    // Configuración de la red
    int hidden_dims[] = {64, 64};
    int num_hidden = 2;

    std::cout << "Configuracion del entrenamiento:" << std::endl;
    std::cout << "  Episodios: " << NUM_EPISODES << std::endl;
    std::cout << "  Max pasos por episodio: " << MAX_STEPS_PER_EPISODE << std::endl;
    std::cout << "  Arquitectura de red: " << CartPoleEnv::STATE_DIM;
    for (int i = 0; i < num_hidden; i++) {
        std::cout << " -> " << hidden_dims[i];
    }
    std::cout << " -> " << CartPoleEnv::ACTION_DIM << std::endl;
    std::cout << "  Frecuencia de actualizacion de target: " << TARGET_UPDATE_FREQ << " episodios" << std::endl;
    std::cout << std::endl;

    // Crear agente DQN
    std::cout << "Inicializando agente DQN..." << std::endl;
    DQN agent(CartPoleEnv::STATE_DIM, CartPoleEnv::ACTION_DIM,
              hidden_dims, num_hidden,
              0.001f,  // learning_rate
              0.99f,   // gamma
              1.0f,    // epsilon inicial
              0.01f,   // epsilon min
              0.995f,  // epsilon decay
              32,      // batch_size
              10000,   // replay_capacity
              TARGET_UPDATE_FREQ);

    // Crear ambiente
    CartPoleEnv env;

    // Variables para tracking
    std::vector<float> episode_rewards;
    std::vector<int> episode_lengths;
    float best_avg_reward = 0.0f;

    std::cout << "Iniciando entrenamiento...\n" << std::endl;

    // Loop de entrenamiento
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        float state[CartPoleEnv::STATE_DIM];
        env.reset(state);

        float episode_reward = 0.0f;
        int step = 0;

        // Loop del episodio
        for (step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
            // Seleccionar acción
            int action = agent.select_action(state);

            // Ejecutar acción en el ambiente
            float next_state[CartPoleEnv::STATE_DIM];
            bool done;
            float reward = env.step(action, next_state, &done);

            // Almacenar experiencia
            agent.store_experience(state, action, reward, next_state, done);

            // Entrenar
            agent.train_step();

            // Actualizar estado
            for (int i = 0; i < CartPoleEnv::STATE_DIM; i++) {
                state[i] = next_state[i];
            }

            episode_reward += reward;

            if (done) {
                break;
            }
        }

        // Decay epsilon
        agent.decay_epsilon();

        // Actualizar target network cada TARGET_UPDATE_FREQ episodios
        if ((episode + 1) % TARGET_UPDATE_FREQ == 0) {
            agent.update_target_network();
        }

        // Guardar métricas
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(step + 1);

        // Imprimir progreso
        if ((episode + 1) % PRINT_FREQ == 0) {
            // Calcular promedio de últimos episodios
            int window = std::min(PRINT_FREQ, (int)episode_rewards.size());
            float avg_reward = 0.0f;
            float avg_length = 0.0f;

            for (int i = 0; i < window; i++) {
                avg_reward += episode_rewards[episode_rewards.size() - 1 - i];
                avg_length += episode_lengths[episode_lengths.size() - 1 - i];
            }
            avg_reward /= window;
            avg_length /= window;

            if (avg_reward > best_avg_reward) {
                best_avg_reward = avg_reward;
            }

            std::cout << "Episodio " << episode + 1 << "/" << NUM_EPISODES
                     << " | Avg Reward: " << avg_reward
                     << " | Avg Length: " << avg_length
                     << " | Epsilon: " << agent.get_epsilon()
                     << " | Best Avg: " << best_avg_reward << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Entrenamiento completado!" << std::endl;
    std::cout << "========================================" << std::endl;

    // Calcular estadísticas finales
    float total_reward = std::accumulate(episode_rewards.begin(), episode_rewards.end(), 0.0f);
    float avg_total_reward = total_reward / episode_rewards.size();

    // Promedio de últimos 100 episodios
    int last_n = std::min(100, (int)episode_rewards.size());
    float last_100_reward = 0.0f;
    for (int i = 0; i < last_n; i++) {
        last_100_reward += episode_rewards[episode_rewards.size() - 1 - i];
    }
    last_100_reward /= last_n;

    std::cout << "\nEstadisticas finales:" << std::endl;
    std::cout << "  Recompensa promedio total: " << avg_total_reward << std::endl;
    std::cout << "  Recompensa promedio ultimos 100 episodios: " << last_100_reward << std::endl;
    std::cout << "  Mejor recompensa promedio: " << best_avg_reward << std::endl;

    // Test del agente entrenado (sin exploración)
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test del agente entrenado (sin exploracion)" << std::endl;
    std::cout << "========================================" << std::endl;

    const int NUM_TEST_EPISODES = 10;
    float test_rewards_sum = 0.0f;

    for (int episode = 0; episode < NUM_TEST_EPISODES; episode++) {
        float state[CartPoleEnv::STATE_DIM];
        env.reset(state);

        float episode_reward = 0.0f;
        int step = 0;

        for (step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
            // Usar acción greedy (sin exploración)
            int action = agent.select_greedy_action(state);

            float next_state[CartPoleEnv::STATE_DIM];
            bool done;
            float reward = env.step(action, next_state, &done);

            for (int i = 0; i < CartPoleEnv::STATE_DIM; i++) {
                state[i] = next_state[i];
            }

            episode_reward += reward;

            if (done) {
                break;
            }
        }

        test_rewards_sum += episode_reward;
        std::cout << "Episodio de test " << episode + 1 << ": Recompensa = "
                 << episode_reward << ", Pasos = " << step + 1 << std::endl;
    }

    float avg_test_reward = test_rewards_sum / NUM_TEST_EPISODES;
    std::cout << "\nRecompensa promedio en test: " << avg_test_reward << std::endl;

    if (avg_test_reward >= 450.0f) {
        std::cout << "\n¡¡¡EXCELENTE!!! El agente ha aprendido a resolver CartPole!" << std::endl;
    } else if (avg_test_reward >= 200.0f) {
        std::cout << "\n¡Buen trabajo! El agente esta aprendiendo." << std::endl;
    } else {
        std::cout << "\nEl agente necesita mas entrenamiento." << std::endl;
    }

    std::cout << "\n========================================" << std::endl;

    return 0;
}
