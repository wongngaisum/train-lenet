# Training LeNet without cublasSgemm

This project implements a core function (matrix multiplication with transpose) in a famous deep learning framework (CAFFE) to train a digit recognition neural network with GPU. Matrix Multiplication is regarded as the core operation for modern convolution neural networks. Actually, even the convolution is computed by matrix multiplication.

## Why CAFFE

Speed makes Caffe perfect for research experiments and industry deployment. Caffe can process over 60M images per day with a single NVIDIA K40 GPU. That’s 1 ms/image for inference and 4 ms/image for learning and more recent library versions and hardware are faster still. We believe that Caffe is among the fastest convnet implementations available.

## What I actually did

In the CAFFE project, there is file named `./src/caffe/util/math_functions.cu`, within which, there is a function named `void caffe_gpu_gemm<float>`. In this project, basically, I implemented my own `void caffe_gpu_gemm<float>`.

The original `void caffe_gpu_gemm<float>` is shown as following: Noted that **const float* A, const float* B and float* C** are GPU memory pointers. And you do not need to worry about CBLAS_TRANSPOSE, just regard it as True and False.

```c
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
 const float alpha, const float* A, const float* B, const float beta,
 float* C) {

// Note that cublas follows fortran order. int lda = (TransA == CblasNoTrans) ? K : M;
 int ldb = (TransB == CblasNoTrans) ? N : K;
 cublasOperation_t cuTransA =

(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 cublasOperation_t cuTransB =

(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,

N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));

}
```

Basically, **cublasSgemm** performs the matrix-matrix multiplication **C = αop(A)op(B) + βC**, where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions **op(A) : m × k , op(B) : k × n and C : m × n** , respectively.

Also, for matrix A, **op(A) = A if transa == CUBLAS_OP_N; op(A) = AT if transa == CUBLAS_OP_T**. And op(B) is defined similarly for matrix B.

I have implemented the following kernels to replace the original cublasSgemm.

1. The `matrix_transpose<<<??,??>>>(...)` to do Anew = op(A) or Bnew = op(B). 
2. The `matrix_multiplication_addition_<<<??,??>>>(...)` to compute R = αAnewBnew + βC.   

## How to install CAFFE

```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler 
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install cmake
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build
cmake ..
make -j16
cd ..
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./examples/mnist/train_lenet.sh
```

## How to start the training

```bash
mv "xxxxxx/math_functions.cu" "your_caffe_path/src/caffe/util/math_functions.cu"
cd "your_caffe_path/build"
make -j16
cd ..
./example/mnist/train_lenet.sh
```

## Report

See [Report](Report.pdf)
