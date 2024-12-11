// MNIST_Neural_Network.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// Hyperparameters
const int input_dim = 784;
const int hidden_dim = 128;
const int output_dim = 10;
const int batch_size = 16;
const double learning_rate = 0.001;
const int epochs = 5;
double epoch_loss = 0.0;

// Activation functions
vector<double> sigmoid(const vector<double>& x) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0 / (1.0 + exp(-x[i]));
    }
    return result;
}

vector<double> sigmoid_derivative(const vector<double>& x) {
    vector<double> sig = sigmoid(x);
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sig[i] * (1 - sig[i]);
    }
    return result;
}

// Initialize weights and biases
void initialize_weights(vector<vector<double>>& W, int rows, int cols) {
    W.resize(rows, vector<double>(cols));
    for (auto& row : W) {
        for (auto& val : row) {
            val = ((double) rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
    }
}

void initialize_vector(vector<double>& vec, int size) {
    vec.resize(size);
    for (auto& val : vec) {
        val = ((double) rand() / RAND_MAX) * 2 - 1;
    }
}

// Matrix-vector multiplication
vector<double> mat_vec_mul(const vector<vector<double>>& mat, const vector<double>& vec) {
    vector<double> result(mat.size());
    for (size_t i = 0; i < mat.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < vec.size(); ++j) {
            sum += mat[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

// Vector addition
vector<double> vec_add(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Vector subtraction
vector<double> vec_sub(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Element-wise multiplication
vector<double> vec_mul(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// Transpose of a matrix
vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    vector<vector<double>> result(mat[0].size(), vector<double>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

// Outer product of two vectors
vector<vector<double>> outer_product(const vector<double>& a, const vector<double>& b) {
    vector<vector<double>> result(a.size(), vector<double>(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

// One-hot encoding
vector<double> one_hot(int label) {
    vector<double> vec(output_dim, 0.0);
    vec[label] = 1.0;
    return vec;
}

// Load MNIST data (simplified placeholder)
void load_mnist(
    vector<vector<double>>& train_images,
    vector<vector<double>>& train_labels,
    vector<vector<double>>& val_images,
    vector<vector<double>>& val_labels) {
    int total_samples = 1000;
    int val_samples = 200; // 20% for validation

    for (int i = 0; i < total_samples; ++i) {
        vector<double> image(input_dim, 0.5); // Dummy image data
        vector<double> label = one_hot(i % output_dim); // Dummy label
        
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
    vector<vector<double>> W1, W2;
    vector<double> b1, b2;
    initialize_weights(W1, hidden_dim, input_dim);
    initialize_weights(W2, output_dim, hidden_dim);
    initialize_vector(b1, hidden_dim);
    initialize_vector(b2, output_dim);

    // Load data
    vector<vector<double>> train_images, train_labels;
    vector<vector<double>> val_images, val_labels;
    load_mnist(train_images, train_labels, val_images, val_labels);

    int num_batches = train_images.size() / batch_size;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (int batch = 0; batch < num_batches; ++batch) {
            auto batch_start = batch * batch_size;
            auto batch_end = batch_start + batch_size;

            // Initialize gradients
            vector<vector<double>> dW1(hidden_dim, vector<double>(input_dim, 0.0));
            vector<double> db1(hidden_dim, 0.0);
            vector<vector<double>> dW2(output_dim, vector<double>(hidden_dim, 0.0));
            vector<double> db2(output_dim, 0.0);

            // Forward and backward pass
            for (int i = batch_start; i < batch_end; ++i) {
                // Forward pass
                vector<double>& x = train_images[i];
                vector<double>& y = train_labels[i];

                vector<double> z1 = vec_add(mat_vec_mul(W1, x), b1);
                vector<double> a1 = sigmoid(z1);
                vector<double> z2 = vec_add(mat_vec_mul(W2, a1), b2);
                vector<double> a2 = sigmoid(z2);

                // Compute loss
                vector<double> error = vec_sub(a2, y);
                double sample_loss = 0.0;
                for (size_t j = 0; j < error.size(); ++j) {
                    sample_loss += error[j] * error[j];
                }
                epoch_loss += sample_loss / error.size();

                // Backward pass
                vector<double> delta2 = vec_mul(error, sigmoid_derivative(z2));
                vector<vector<double>> dw2 = outer_product(delta2, a1);
                vector<double> delta1 = vec_mul(mat_vec_mul(transpose(W2), delta2), sigmoid_derivative(z1));
                vector<vector<double>> dw1 = outer_product(delta1, x);

                // Accumulate gradients
                for (size_t j = 0; j < dW2.size(); ++j) {
                    for (size_t k = 0; k < dW2[0].size(); ++k) {
                        dW2[j][k] += dw2[j][k];
                    }
                    db2[j] += delta2[j];
                }
                for (size_t j = 0; j < dW1.size(); ++j) {
                    for (size_t k = 0; k < dW1[0].size(); ++k) {
                        dW1[j][k] += dw1[j][k];
                    }
                    db1[j] += delta1[j];
                }
            }

            // Update weights and biases
            for (size_t i = 0; i < W1.size(); ++i) {
                for (size_t j = 0; j < W1[0].size(); ++j) {
                    W1[i][j] -= learning_rate * dW1[i][j] / batch_size;
                }
                b1[i] -= learning_rate * db1[i] / batch_size;
            }

            for (size_t i = 0; i < W2.size(); ++i) {
                for (size_t j = 0; j < W2[0].size(); ++j) {
                    W2[i][j] -= learning_rate * dW2[i][j] / batch_size;
                }
                b2[i] -= learning_rate * db2[i] / batch_size;
            }
        }

        epoch_loss /= train_images.size();

        // Validation loss
        double val_loss = 0.0;
        for (size_t i = 0; i < val_images.size(); ++i) {
            vector<double>& x = val_images[i];
            vector<double>& y = val_labels[i];

            vector<double> z1 = vec_add(mat_vec_mul(W1, x), b1);
            vector<double> a1 = sigmoid(z1);
            vector<double> z2 = vec_add(mat_vec_mul(W2, a1), b2);
            vector<double> a2 = sigmoid(z2);

            vector<double> error = vec_sub(a2, y);
            double sample_loss = 0.0;
            for (size_t j = 0; j < error.size(); ++j) {
                sample_loss += error[j] * error[j];
            }
            val_loss += sample_loss / error.size();
        }
        val_loss /= val_images.size();

        cout << "Epoch " << epoch + 1 << " completed. Training Loss: " << epoch_loss << ", Validation Loss: " << val_loss << endl;
    }
    return 0;
}