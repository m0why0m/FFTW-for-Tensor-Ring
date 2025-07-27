# FFTW-for-Tensor-Ring
### Visual Studio 2022 Configuration Instructions
This project was completed under windows 10 using C/C ++, aiming to perform polynomial multiplication on a cyclotomic ring using the fftw library. If you are using **Visual Studio 2022**, please follow these steps to configure your project for using the FFTW library:

#### 1. Set Include Directory
Add the following path to the `AdditionalIncludeDirectories` of your project:
\`
fftw-3.3.4-dll64
\`

#### 2. Set Library Directory
Add the same path to the `AdditionalLibraryDirectories`:
\`
fftw-3.3.4-dll64
\`

#### 3. Add Library Dependency
Add the following library file to `AdditionalDependencies`:
\`
libfftw3-3.lib
\`

#### 4. Place DLL File
Make sure to place the file below in your executable file directory:
\`
libfftw3-3.lib
\`

> **Note:** The above instructions also apply to the `fftw3l` and `fftw3f` versions of the library.
> **Note:**This code is associated with the paper (TensorAFBS) submitted by Asiacrypt 2025, which in order to verify the correctness of polynomial multiplication over the tensor ring using the FFTW library.
