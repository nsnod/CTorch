# build.ps1

# Exit immediately if a command exits with a non-zero status
$ErrorActionPreference = "Stop"

try {
    # Check if the "build" directory exists; if yes, clean it to avoid conflicts
    if (Test-Path -Path "build") {
        Write-Output "Cleaning 'build' directory to avoid generator conflicts..."
        Remove-Item -Recurse -Force "build"
    }
    
    # Create a fresh "build" directory
    Write-Output "Creating 'build' directory..."
    New-Item -ItemType Directory -Path "build" | Out-Null

    # Navigate into the "build" directory
    Set-Location "build"

    # Configure the project using CMake with MinGW Makefiles and specific compilers
    Write-Output "Configuring project with CMake..."
    cmake -G "MinGW Makefiles" -DBUILD_TESTING=ON `
          -DCMAKE_C_COMPILER="C:/MinGW/mingw64/bin/gcc.exe" `
          -DCMAKE_CXX_COMPILER="C:/MinGW/mingw64/bin/g++.exe" `
          -DCMAKE_VERBOSE_MAKEFILE=ON ..

    # Build the project using mingw32-make
    Write-Output "Building the project..."
    mingw32-make -j4

    # Rename the executable to mnist.exe
    if (Test-Path "./CTorch.exe") {
        Write-Output "Renaming CTorch.exe to mnist.exe..."
        Rename-Item -Path "./CTorch.exe" -NewName "mnist.exe" -Force
        Write-Output "Executable has been renamed to mnist.exe in the 'build' directory."
    } else {
        Write-Error "Build succeeded, but CTorch.exe was not found in the build directory."
    }

    Write-Output "Build and rename completed successfully."
}
catch {
    Write-Error "An error occurred: $_"
    exit 1
}
