// // driver code for testing
// #include "src/linear.h"
// #include "src/tensor.h"
// #include "src/array.h"
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <random>
// #include <ctime>
// // int main () {
// //     std::vector<int> shape = {2, 2};  // Shape for a 2x2 tensor
// //     Tensor<float> tensor(shape);      // Instantiate Tensor object

// //     tensor.randomize_tensor(0.0f, 1.0f);
    
// //     tensor.resetZeroData();
// //     tensor.resetZeroGrad();  // Randomize contents between 0 and 1
// //     tensor.print_tensor();
    
// //     tensor.print_tensor();                // Print tensor contents

// //     return 0;

// // }

// // main.cpp

// // main.cpp

// // Function to shuffle data and labels in unison
// void shuffle_data(std::vector<std::vector<float>> &data, std::vector<int> &labels) {
//     std::random_device rd;
//     std::mt19937 g(rd());
//     size_t n = data.size();
//     std::vector<size_t> indices(n);
//     for(size_t i = 0; i < n; ++i) indices[i] = i;
//     std::shuffle(indices.begin(), indices.end(), g);
//     std::vector<std::vector<float>> shuffled_data;
//     std::vector<int> shuffled_labels;
//     shuffled_data.reserve(n);
//     shuffled_labels.reserve(n);
//     for(auto idx : indices){
//         shuffled_data.push_back(data[idx]);
//         shuffled_labels.push_back(labels[idx]);
//     }
//     data = shuffled_data;
//     labels = shuffled_labels;
// }

// int main() {
//     srand(static_cast<unsigned int>(time(0)));

//     // Hyperparameters
//     const int input_dim = 784;    // 28x28 images
//     const int hidden_dim = 128;
//     const int output_dim = 10;    // Digits 0-9
//     const int batch_size = 64;     // You can experiment with this
//     const float learning_rate = 0.001f; // You can experiment with this
//     const int epochs = 5;

//     // Load MNIST data
//     std::vector<std::vector<float>> train_images;
//     std::vector<int> train_labels;
//     std::vector<std::vector<float>> test_images;
//     std::vector<int> test_labels;

//     std::cout << "Loading MNIST training data..." << std::endl;
//     bool load_success = load_mnist("path_to_train_images/train-images-idx3-ubyte", 
//                                    "path_to_train_labels/train-labels-idx1-ubyte",
//                                    train_images, train_labels);
//     if (!load_success) {
//         std::cerr << "Failed to load MNIST training data." << std::endl;
//         return EXIT_FAILURE;
//     }

//     std::cout << "Loading MNIST test data..." << std::endl;
//     load_success = load_mnist("path_to_test_images/t10k-images-idx3-ubyte", 
//                               "path_to_test_labels/t10k-labels-idx1-ubyte",
//                               test_images, test_labels);
//     if (!load_success) {
//         std::cerr << "Failed to load MNIST test data." << std::endl;
//         return EXIT_FAILURE;
//     }

//     std::cout << "Training samples: " << train_images.size() << std::endl;
//     std::cout << "Test samples: " << test_images.size() << std::endl;

//     // Initialize network layers
//     Linear<float> linear1(input_dim, hidden_dim);      // Input to Hidden
//     ReLU<float> relu_activation;                        // ReLU Activation
//     Linear<float> linear2(hidden_dim, output_dim);     // Hidden to Output
//     Softmax<float> softmax_activation;                  // Softmax Activation
//     CrossEntropyLoss<float> loss_fn;                    // Loss function

//     // Training loop
//     for (int epoch = 1; epoch <= epochs; ++epoch) {
//         // Shuffle data at the beginning of each epoch
//         shuffle_data(train_images, train_labels);
//         std::cout << "Epoch " << epoch << "/" << epochs << std::endl;

//         float epoch_loss = 0.0f;
//         int correct_predictions = 0;

//         int num_batches = train_images.size() / batch_size;

//         for(int batch = 0; batch < num_batches; ++batch){
//             int batch_start = batch * batch_size;
//             int batch_end = batch_start + batch_size;

//             // Prepare input tensor
//             Tensor<float> input_tensor({batch_size, input_dim});
//             for(int i = 0; i < batch_size; ++i){
//                 for(int j = 0; j < input_dim; ++j){
//                     input_tensor.data_->data_[i * input_dim + j] = train_images[batch_start + i][j];
//                 }
//             }

//             // Prepare labels
//             std::vector<int> batch_labels(train_labels.begin() + batch_start, train_labels.begin() + batch_end);

//             // Forward Pass
//             Tensor<float>* hidden = linear1.forward(input_tensor);           // Linear1
//             Tensor<float>* activated = relu_activation.forward(*hidden);     // ReLU
//             Tensor<float>* output = linear2.forward(*activated);            // Linear2
//             Tensor<float>* probabilities = softmax_activation.forward(*output); // Softmax

//             // Compute Loss
//             float loss = loss_fn.forward(*probabilities, batch_labels);
//             epoch_loss += loss;

//             // Compute Accuracy
//             correct_predictions += loss_fn.compute_accuracy(*probabilities, batch_labels);

//             // Backward Pass
//             Tensor<float>* grad_loss = loss_fn.backward();                     // dL/dprobabilities
//             Tensor<float>* grad_softmax = softmax_activation.backward(grad_loss); // dL/dz2
//             Tensor<float>* grad_linear2 = linear2.backward(grad_softmax);      // dL/da1
//             Tensor<float>* grad_relu = relu_activation.backward(grad_linear2); // dL/dz1
//             Tensor<float>* grad_linear1 = linear1.backward(grad_relu);        // dL/dinput

//             // Update Weights and Biases
//             linear1.update_parameters(learning_rate);
//             linear2.update_parameters(learning_rate);

//             // Clean up dynamically allocated tensors
//             delete hidden;
//             delete activated;
//             delete output;
//             delete probabilities;
//             delete grad_loss;
//             delete grad_softmax;
//             delete grad_linear2;
//             delete grad_relu;
//             delete grad_linear1;
//         }

//         // Handle remaining samples that don't fit into a full batch
//         if(train_images.size() % batch_size != 0){
//             int batch_start = num_batches * batch_size;
//             int current_batch_size = train_images.size() - batch_start;

//             // Prepare input tensor
//             Tensor<float> input_tensor({current_batch_size, input_dim});
//             for(int i = 0; i < current_batch_size; ++i){
//                 for(int j = 0; j < input_dim; ++j){
//                     input_tensor.data_->data_[i * input_dim + j] = train_images[batch_start + i][j];
//                 }
//             }

//             // Prepare labels
//             std::vector<int> batch_labels(train_labels.begin() + batch_start, train_labels.end());

//             // Forward Pass
//             Tensor<float>* hidden = linear1.forward(input_tensor);           // Linear1
//             Tensor<float>* activated = relu_activation.forward(*hidden);     // ReLU
//             Tensor<float>* output = linear2.forward(*activated);            // Linear2
//             Tensor<float>* probabilities = softmax_activation.forward(*output); // Softmax

//             // Compute Loss
//             float loss = loss_fn.forward(*probabilities, batch_labels);
//             epoch_loss += loss;

//             // Compute Accuracy
//             correct_predictions += loss_fn.compute_accuracy(*probabilities, batch_labels);

//             // Backward Pass
//             Tensor<float>* grad_loss = loss_fn.backward();                     // dL/dprobabilities
//             Tensor<float>* grad_softmax = softmax_activation.backward(grad_loss); // dL/dz2
//             Tensor<float>* grad_linear2 = linear2.backward(grad_softmax);      // dL/da1
//             Tensor<float>* grad_relu = relu_activation.backward(grad_linear2); // dL/dz1
//             Tensor<float>* grad_linear1 = linear1.backward(grad_relu);        // dL/dinput

//             // Update Weights and Biases
//             linear1.update_parameters(learning_rate);
//             linear2.update_parameters(learning_rate);

//             // Clean up dynamically allocated tensors
//             delete hidden;
//             delete activated;
//             delete output;
//             delete probabilities;
//             delete grad_loss;
//             delete grad_softmax;
//             delete grad_linear2;
//             delete grad_relu;
//             delete grad_linear1;
//         }

//         // Calculate average loss and accuracy for the epoch
//         float average_loss = epoch_loss / num_batches;
//         float accuracy = (static_cast<float>(correct_predictions) / (num_batches * batch_size)) * 100.0f;

//         std::cout << "Epoch [" << epoch << "/" << epochs << "] - "
//                   << "Loss: " << average_loss << " - "
//                   << "Accuracy: " << accuracy << "%" << std::endl;

//         // Optionally, evaluate on test data after each epoch
//         // Implement similar forward passes on test data and compute accuracy
//     }

//     return 0;
// }
