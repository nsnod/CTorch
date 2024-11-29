#include <iostream>
#include <vector>
#include <future>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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
    //Load the image
    string imagePath = "../src/mnist/cat.jpg";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Could not read the image. Please check the file path." << endl;
        return -1; // Exit with an error code
    }

    // vector<vector<uint8_t>> images(1000, vector<uint8_t>(784, 128)); // 1000 images, 28x28 pixels
    vector<future<void>> futures;
    // Convert the image to a 2D vector of uint8_t
    vector<vector<uint8_t>> imageData(image.rows, vector<uint8_t>(image.cols));

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            imageData[i][j] = image.at<uint8_t>(i, j);
        }
    }

    size_t numThreads = 4;
    size_t subsetSize = imageData.size() / numThreads;

    for (size_t i = 0; i < numThreads; ++i) {
        size_t start = i * subsetSize;
        size_t end = (i + 1) * subsetSize;
        futures.emplace_back(async(launch::async, normalizeSubset, ref(imageData), start, end));
    }

    // Wait for all tasks to complete
    for (auto &f : futures) {
        f.get();
    }

    cout << "Normalization completed using async tasks." << endl;

    // Convert the normalized 2D vector back to a Mat and save/display the result
    Mat normalizedImage(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            normalizedImage.at<uint8_t>(i, j) = imageData[i][j];
        }
    }

    imwrite("normalized_cat.jpg", normalizedImage); // Save the normalized image
    imshow("Normalized Image", normalizedImage); // Display the image
    waitKey(0); // Wait for a key press

    return 0;
}
