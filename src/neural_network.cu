#include "neural_network.cuh"
#include <iostream>
#include <cmath>

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "Error CUDA: " << message << " - "
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel para forward pass de capa fully connected
__global__ void fully_connected_forward(float* input, float* weights, float* biases,
                                       float* output, int input_size, int output_size,
                                       int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = output_size * batch_size;

    if (idx < total_outputs) {
        int batch_idx = idx / output_size;
        int neuron_idx = idx % output_size;

        float sum = biases[neuron_idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] *
                   weights[neuron_idx * input_size + i];
        }
        output[idx] = sum;
    }
}

// Kernel para activación ReLU
__global__ void relu_activation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel para backward de ReLU
__global__ void relu_backward(float* grad, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (output[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

// Kernel para calcular gradiente de salida (MSE loss)
__global__ void compute_output_gradient(float* output, float* target, float* grad,
                                       int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size * batch_size;

    if (idx < total_size) {
        grad[idx] = 2.0f * (output[idx] - target[idx]) / batch_size;
    }
}

// Kernel para backward pass de capa fully connected
__global__ void fully_connected_backward(float* input, float* weights, float* output_grad,
                                        float* input_grad, float* weights_grad,
                                        float* biases_grad, int input_size,
                                        int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_params = input_size * output_size;

    if (idx < total_params) {
        int out_idx = idx / input_size;
        int in_idx = idx % input_size;

        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float out_g = output_grad[b * output_size + out_idx];
            float in_val = input[b * input_size + in_idx];
            grad_sum += out_g * in_val;
        }
        weights_grad[idx] = grad_sum;
    }

    // Calcular gradiente de biases
    if (idx < output_size) {
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_grad += output_grad[b * output_size + idx];
        }
        biases_grad[idx] = bias_grad;
    }

    // Calcular gradiente de input (para propagar a capa anterior)
    int total_inputs = input_size * batch_size;
    if (idx < total_inputs) {
        int batch_idx = idx / input_size;
        int input_idx = idx % input_size;

        float grad = 0.0f;
        for (int out_idx = 0; out_idx < output_size; out_idx++) {
            grad += output_grad[batch_idx * output_size + out_idx] *
                    weights[out_idx * input_size + input_idx];
        }
        input_grad[idx] = grad;
    }
}

// Kernel para actualizar pesos con gradiente descendente
__global__ void update_weights(float* weights, float* weights_grad, float learning_rate,
                              int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * weights_grad[idx];
    }
}

// Constructor de NeuralNetwork
NeuralNetwork::NeuralNetwork(int input_dim, int* hidden_dims, int num_hidden, int output_dim) {
    this->input_size = input_dim;
    this->output_size = output_dim;
    this->num_layers = num_hidden + 1; // capas ocultas + capa de salida

    // Alocar array de capas
    layers = new Layer[num_layers];

    // Crear generador de números aleatorios
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 1234ULL);

    // Inicializar cada capa
    int current_input = input_dim;
    for (int i = 0; i < num_layers; i++) {
        int current_output = (i < num_hidden) ? hidden_dims[i] : output_dim;

        layers[i].input_size = current_input;
        layers[i].output_size = current_output;

        // Alocar memoria para weights, biases y gradientes
        int weights_size = current_input * current_output;
        checkCudaError(cudaMalloc(&layers[i].weights, weights_size * sizeof(float)),
                      "Allocating weights");
        checkCudaError(cudaMalloc(&layers[i].biases, current_output * sizeof(float)),
                      "Allocating biases");
        checkCudaError(cudaMalloc(&layers[i].weights_grad, weights_size * sizeof(float)),
                      "Allocating weights_grad");
        checkCudaError(cudaMalloc(&layers[i].biases_grad, current_output * sizeof(float)),
                      "Allocating biases_grad");
        checkCudaError(cudaMalloc(&layers[i].output, current_output * sizeof(float)),
                      "Allocating output");
        checkCudaError(cudaMalloc(&layers[i].input_cache, current_input * sizeof(float)),
                      "Allocating input_cache");

        // Inicializar weights con Xavier initialization
        // curandGenerateUniform genera valores en [0, 1]
        // Los escalamos a [-limit, limit] en CPU por simplicidad
        float limit = sqrtf(6.0f / (current_input + current_output));
        float* h_weights = new float[weights_size];
        curandGenerateUniform(rng, layers[i].weights, weights_size);

        // Copiar a host, escalar y copiar de vuelta
        checkCudaError(cudaMemcpy(h_weights, layers[i].weights,
                                 weights_size * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "Copying weights for scaling");

        for (int j = 0; j < weights_size; j++) {
            h_weights[j] = h_weights[j] * 2.0f * limit - limit; // Escalar de [0,1] a [-limit, limit]
        }

        checkCudaError(cudaMemcpy(layers[i].weights, h_weights,
                                 weights_size * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "Copying scaled weights back");

        // Inicializar biases a cero
        checkCudaError(cudaMemset(layers[i].biases, 0, current_output * sizeof(float)),
                      "Initializing biases");

        delete[] h_weights;
        current_input = current_output;
    }
}

// Destructor
NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < num_layers; i++) {
        cudaFree(layers[i].weights);
        cudaFree(layers[i].biases);
        cudaFree(layers[i].weights_grad);
        cudaFree(layers[i].biases_grad);
        cudaFree(layers[i].output);
        cudaFree(layers[i].input_cache);
    }
    delete[] layers;
    curandDestroyGenerator(rng);
}

// Forward pass
void NeuralNetwork::forward(float* input, float* output, int batch_size) {
    float* current_input = input;

    for (int i = 0; i < num_layers; i++) {
        Layer& layer = layers[i];

        // Guardar input para backward pass
        int input_size_total = layer.input_size * batch_size;
        checkCudaError(cudaMemcpy(layer.input_cache, current_input,
                                 input_size_total * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "Caching input");

        // Realizar multiplicación de matrices (fully connected)
        int output_total = layer.output_size * batch_size;
        int blocks = (output_total + 255) / 256;
        fully_connected_forward<<<blocks, 256>>>(current_input, layer.weights,
                                                 layer.biases, layer.output,
                                                 layer.input_size, layer.output_size,
                                                 batch_size);
        checkCudaError(cudaGetLastError(), "Forward pass kernel");

        // Aplicar ReLU excepto en la última capa
        if (i < num_layers - 1) {
            relu_activation<<<blocks, 256>>>(layer.output, output_total);
            checkCudaError(cudaGetLastError(), "ReLU activation");
        }

        current_input = layer.output;
    }

    // Copiar salida final
    int final_output_size = layers[num_layers - 1].output_size * batch_size;
    checkCudaError(cudaMemcpy(output, layers[num_layers - 1].output,
                             final_output_size * sizeof(float),
                             cudaMemcpyDeviceToDevice),
                  "Copying final output");
}

// Backward pass
void NeuralNetwork::backward(float* target, float* output, float learning_rate, int batch_size) {
    // Alocar memoria para gradientes
    float* d_output_grad;
    int final_size = layers[num_layers - 1].output_size * batch_size;
    checkCudaError(cudaMalloc(&d_output_grad, final_size * sizeof(float)),
                  "Allocating output gradient");

    // Calcular gradiente de la salida (MSE loss)
    int blocks = (final_size + 255) / 256;
    compute_output_gradient<<<blocks, 256>>>(output, target, d_output_grad,
                                            layers[num_layers - 1].output_size,
                                            batch_size);
    checkCudaError(cudaGetLastError(), "Computing output gradient");

    float* current_grad = d_output_grad;

    // Backpropagate a través de cada capa
    for (int i = num_layers - 1; i >= 0; i--) {
        Layer& layer = layers[i];

        // Alocar gradiente de entrada
        float* d_input_grad;
        int input_total = layer.input_size * batch_size;
        checkCudaError(cudaMalloc(&d_input_grad, input_total * sizeof(float)),
                      "Allocating input gradient");

        // Backward pass de la capa
        int total_params = layer.input_size * layer.output_size;
        blocks = (total_params + 255) / 256;
        fully_connected_backward<<<blocks, 256>>>(layer.input_cache, layer.weights,
                                                  current_grad, d_input_grad,
                                                  layer.weights_grad, layer.biases_grad,
                                                  layer.input_size, layer.output_size,
                                                  batch_size);
        checkCudaError(cudaGetLastError(), "Backward pass kernel");

        // Actualizar pesos y biases
        update_weights<<<blocks, 256>>>(layer.weights, layer.weights_grad,
                                       learning_rate, total_params, batch_size);
        blocks = (layer.output_size + 255) / 256;
        update_weights<<<blocks, 256>>>(layer.biases, layer.biases_grad,
                                       learning_rate, layer.output_size, batch_size);
        checkCudaError(cudaGetLastError(), "Updating weights");

        // Aplicar backward de ReLU si no es la última capa
        if (i < num_layers - 1) {
            relu_backward<<<(input_total + 255) / 256, 256>>>(d_input_grad,
                                                              layer.input_cache,
                                                              input_total);
            checkCudaError(cudaGetLastError(), "ReLU backward");
        }

        // Liberar gradiente actual y preparar para siguiente capa
        if (i < num_layers - 1) {
            cudaFree(current_grad);
        }
        current_grad = d_input_grad;
    }

    // Limpiar
    cudaFree(current_grad);
    cudaFree(d_output_grad);
}

// Copiar pesos de otra red
void NeuralNetwork::copy_weights_from(NeuralNetwork* other) {
    for (int i = 0; i < num_layers; i++) {
        int weights_size = layers[i].input_size * layers[i].output_size;
        checkCudaError(cudaMemcpy(layers[i].weights, other->layers[i].weights,
                                 weights_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "Copying weights");
        checkCudaError(cudaMemcpy(layers[i].biases, other->layers[i].biases,
                                 layers[i].output_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "Copying biases");
    }
}
