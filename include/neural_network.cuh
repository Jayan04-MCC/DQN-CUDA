#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Estructura para una capa fully-connected
struct Layer {
    int input_size;
    int output_size;
    float* weights;      // weights en GPU
    float* biases;       // biases en GPU
    float* weights_grad; // gradientes de weights
    float* biases_grad;  // gradientes de biases
    float* output;       // salida de la capa
    float* input_cache;  // cache de entrada para backprop
};

// Clase para la red neuronal
class NeuralNetwork {
private:
    int num_layers;
    Layer* layers;
    int input_size;
    int output_size;
    curandGenerator_t rng;

public:
    // Constructor y destructor
    NeuralNetwork(int input_dim, int* hidden_dims, int num_hidden, int output_dim);
    ~NeuralNetwork();

    // Forward pass
    void forward(float* input, float* output, int batch_size = 1);

    // Backward pass
    void backward(float* target, float* output, float learning_rate, int batch_size = 1);

    // Copiar pesos de otra red (para target network)
    void copy_weights_from(NeuralNetwork* other);

    // Obtener pesos para guardar/cargar
    void get_weights(float** weights_out, int* total_params);
    void set_weights(float* weights_in);

    // Getter para salida
    float* get_output() { return layers[num_layers - 1].output; }
    int get_output_size() { return output_size; }
    int get_input_size() { return input_size; }
};

// Kernels CUDA para operaciones de red neuronal
__global__ void fully_connected_forward(float* input, float* weights, float* biases,
                                       float* output, int input_size, int output_size,
                                       int batch_size);

__global__ void relu_activation(float* data, int size);

__global__ void relu_backward(float* grad, float* output, int size);

__global__ void compute_output_gradient(float* output, float* target, float* grad,
                                       int size, int batch_size);

__global__ void fully_connected_backward(float* input, float* weights, float* output_grad,
                                        float* input_grad, float* weights_grad,
                                        float* biases_grad, int input_size,
                                        int output_size, int batch_size);

__global__ void update_weights(float* weights, float* weights_grad, float learning_rate,
                              int size, int batch_size);

// Utilidades
void checkCudaError(cudaError_t error, const char* message);

#endif // NEURAL_NETWORK_CUH
