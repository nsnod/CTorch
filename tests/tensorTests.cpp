#include <mpi.h>
#include <gtest/gtest.h>
#include <chrono>
#include "../src/tensor.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Tensor<float> tensorA({784, 16});
    Tensor<float> tensorB({16, 784});

    //only use ONE tensor across all different processes
    if (rank == 0) {
        tensorA.randomize_tensor(-1.0, 1.0);
        tensorB.randomize_tensor(-1.0, 1.0);
    }

    //ensure all instances have the same base tensor
    MPI_Bcast(tensorA.data_->data_.data(), tensorA.data_->size_, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tensorB.data_->data_.data(), tensorB.data_->size_, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tensorA.grad_->data_.data(), tensorA.grad_->size_, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tensorB.grad_->data_.data(), tensorB.grad_->size_, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // // original tensor check
        // std::cout << "Tensor A:" << std::endl;
        // tensorA.print_tensor();
        // std::cout << "Tensor B:" << std::endl;
        // tensorB.print_tensor();


        auto startNonMPI = std::chrono::high_resolution_clock::now();
        Tensor<float> resultNonMPI = tensorA.non_parallel_tensor_mult_test(tensorB);
        // resultNonMPI.print_tensor();
        auto endNonMPI = std::chrono::high_resolution_clock::now();

        std::cout << "Non-MPI Time: " 
                  << std::chrono::duration<double>(endNonMPI - startNonMPI).count() 
                  << " seconds" << std::endl;

        std::cout << "Non-MPI Result:" << std::endl;
        // resultNonMPI.print_tensor();
    }
    // sync up
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI Test
    auto startMPI = std::chrono::high_resolution_clock::now();
    Tensor<float> resultMPI = tensorA * tensorB;
    auto endMPI = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "MPI Time: " 
                  << std::chrono::duration<double>(endMPI - startMPI).count() 
                  << " seconds" << std::endl;

        std::cout << "MPI Result:" << std::endl;
        // resultMPI.print_tensor();
    }

    MPI_Finalize();
    return 0;
}