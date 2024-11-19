#include <iostream>
#include <vector>
#include <future>
#include <algorithm>
#include <chrono>

using namespace std;

void normalizeSubset(vector<vector<uint8_t>> &images, size_t start, size_t end) {
    auto begin = chrono::high_resolution_clock::now(); //get the time for normalization
    
    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < images[i].size(); ++j) {
            images[i][j] = static_cast<uint8_t>(images[i][j] / 255.0f * 255); // normalize
        }
    }
    auto ending = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration<double>(ending - begin).count() << " seconds" << endl;
}

int main() {
    vector<vector<uint8_t>> images(1000, vector<uint8_t>(784, 128)); // 1000 images, 28x28 pixels
    vector<future<void>> futures;

    size_t numThreads = 4;
    size_t subsetSize = images.size() / numThreads;

    for (size_t i = 0; i < numThreads; ++i) {
        size_t start = i * subsetSize;
        size_t end = (i + 1) * subsetSize;
        futures.emplace_back(async(launch::async, normalizeSubset, ref(images), start, end));
    }

    // Wait for all tasks to complete
    for (auto &f : futures) {
        f.get();
    }

    cout << "Normalization completed using async tasks." << endl;
    return 0;
}
