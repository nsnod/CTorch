// MNIST_Neural_Network.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "src/tensor.h"


using namespace std;

// Hyperparameters
const int input_dim = 784;
const int hidden_dim = 128;
const int output_dim = 10;
const int batch_size = 16;
const double learning_rate = 0.001;
const int epochs = 5;
double epoch_loss = 0.0;

// One-hot encoding
vector<double> one_hot(int label) {
    vector<double> vec(output_dim, 0.0);
    vec[label] = 1.0;
    return vec;
}

// Load MNIST data (simplified placeholder)
void load_mnist(
    vector<Tensor<float>>& train_images,
    vector<Tensor<float>>& train_labels,
    vector<Tensor<float>>& val_images,
    vector<Tensor<float>>& val_labels) {
    int total_samples = 1000;
    int val_samples = 200; // 20% for validation

    for (int i = 0; i < total_samples; ++i) {
        Tensor<float> image(input_dim, 0.5); // Dummy image data
        Tensor<float> label = one_hot(i % output_dim); // Dummy label
        
        if (i < total_samples - val_samples) {
            train_images.push_back(image);
            train_labels.push_back(label);
        } else {
            val_images.push_back(image);
            val_labels.push_back(label);
        }
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    // Initialize weights and biases 
    Tensor<float>* W1({input_dim, hidden_dim}); 
    Tensor<float>* W2({hidden_dim, output_dim}); 


    // // Initialize weights and biases
    // vector<vector<double>> W1, W2;
    // vector<double> b1, b2;
    // initialize_weights(W1, hidden_dim, input_dim);
    // initialize_weights(W2, output_dim, hidden_dim);
    // initialize_vector(b1, hidden_dim);
    // initialize_vector(b2, output_dim);

    // Load data
    vector<Tensor<float>> train_images, train_labels;
    vector<Tensor<float>> val_images, val_labels;
    load_mnist(train_images, train_labels, val_images, val_labels);

    int num_batches = train_images.size() / batch_size;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;

        for (int batch = 0; batch < num_batches; ++batch) {
            auto batch_start = batch * batch_size;
            auto batch_end = batch_start + batch_size;

            // Forward and backward pass
            for (int i = batch_start; i < batch_end; ++i) {
                // Forward pass
                Tensor<float>& x = train_images[i];
                Tensor<float>& y = train_labels[i];

                linear_output1 = x * W1;
                hidden_output1 = relu(linear_output1);
                linear_output2 = hidden_output1 * W2;
                hidden_output2 = relu(linear_output2);
                logits = softmax(hidden_output2);

                // Compute loss
                Tensor<float> error = mse_loss(logits, y);
                epoch_loss += error

                // Backward pass 
                backward(&error);
            }

            // Update weights and biases
            for (size_t i = 0; i < W1.size(); ++i) {
                for (size_t j = 0; j < W1[0].size(); ++j) {
                    W1[i][j] -= learning_rate * W1->grad_[i][j] / batch_size;
                }
            }

            for (size_t i = 0; i < W2.size(); ++i) {
                for (size_t j = 0; j < W2[0].size(); ++j) {
                    W2[i][j] -= learning_rate * W2->[i][j] / batch_size;
                }
            }
        }

        epoch_loss /= train_images.size();

        // Validation loss
        float val_loss = 0.0f;
        for (size_t i = 0; i < val_images.size(); ++i) {
            Tensor<float>& x = val_images[i];
            Tensor<float>& y = val_labels[i];

            Tensor<float> error = vec_sub(a2, y);
            double sample_loss = 0.0;
            for (size_t j = 0; j < error.size(); ++j) {
                sample_loss += error[j] * error[j];
            }
            Tensor<float> val_loss = mse_loss(logits, y);
        }
        val_loss /= val_images.size();

        cout << "Epoch " << epoch + 1 << " completed. Training Loss: " << epoch_loss << ", Validation Loss: " << val_loss << endl;
    }
    return 0;
}