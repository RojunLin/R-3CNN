#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/average.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void kernel_dot_forward(const int num, const int channels, const Dtype coefficient,
	const Dtype* bottom_data, Dtype* top) {
  CUDA_KERNEL_LOOP(index, num * channels) {
  	int n = index / channels;
  	int k = index % channels;
  	top[n] += bottom_data[index] * k * coefficient + 1;
  }
}

template <typename Dtype>
__global__ void kernel_dot_backward(const int num, const int channels, const Dtype coefficient,
    Dtype* bottom_diff, const Dtype* top)  {
  CUDA_KERNEL_LOOP(index, num * channels) {
  	int n = index / channels;
  	int k = index % channels;
  	bottom_diff[index] = top[n] * k * coefficient;

  }
}
template <typename Dtype>
void AverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int b_channels = bottom[0]->channels();  
  const Dtype* bottom_data = bottom[0]->cpu_data(); 
  // const int dims = bottom[0]->count(1);
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Dtype* scale_data = scale_.mutable_cpu_data();
  Dtype coefficient = 4.f / (b_channels - 1);  //scale the data

  memset(top_data, 0, sizeof(Dtype) * outer_num_);
    for (int i = 0; i < outer_num_; ++i){
          for (int k = 0; k < b_channels; ++k){
            // LOG(INFO) << k * coefficient;
            top_data[i] += bottom_data[i * b_channels + k] * k ;
            // top_data[i] += bottom_data[i * b_channels + k];
            // LOG(INFO) << "top_data";

          }
          top_data[i] = top_data[i] * coefficient + 1;
          // LOG(INFO) << "top_data" << ' ' << top_data[i];
    }

    // for(int i  = 0; i < outer_num_; ++i){
    //   // if(top_data[i] > 5) 
    //     printf("%f\n", top_data[i]);
    // }


}

template <typename Dtype>
void AverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int b_channels = bottom[0]->channels();  
  const Dtype* top_diff = top[0]->cpu_diff();
  // const int dims = bottom[0]->count(1);
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype coefficient = 4.f / (b_channels - 1);  //scale the data

    for (int i = 0; i < outer_num_; ++i){
          for (int k = 0; k < b_channels; ++k){
            bottom_diff[i * b_channels + k] = top_diff[i] * k * coefficient;
          }
    }


}
INSTANTIATE_LAYER_GPU_FUNCS(AverageLayer);
}
// template <typename Dtype>
// void AverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {

//   int count = bottom[0]->count();
//   outer_num_ = bottom[0]->count(0,1);
//   const int b_channels = bottom[0]->channels();  
//   const Dtype* bottom_data = bottom[0]->gpu_data(); 
//   Dtype* top_data = top[0]->mutable_gpu_data();
//   // LOG(INFO) << "bottom: " << bottom[0]->shape()[0] << ' ' << bottom[0]->shape()[1] << ' ' << bottom[0]->shape()[2] << ' ' << bottom[0]->shape()[3] << ' ';
//   // LOG(INFO) << "top: " << top[0]->shape()[0] << ' ' << top[0]->shape()[1] << ' ' << top[0]->shape()[2] << ' ' << top[0]->shape()[3] << ' ';

//   Dtype coefficient = 4.f / (b_channels - 1);  //scale the data

//   cudaMemset(top_data, 0, outer_num_ * sizeof(Dtype));
//   kernel_dot_forward<Dtype><<<CAFFE_GET_BLOCKS(count),
//       CAFFE_CUDA_NUM_THREADS>>>(outer_num_, b_channels, coefficient, bottom_data, top_data);


// }

// template <typename Dtype>
// void AverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

//   int count = top[0]->count();
//   outer_num_ = bottom[0]->count(0,1);
//   const int b_channels = bottom[0]->channels();  
//   const Dtype* top_diff = top[0]->gpu_diff();
//   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//   Dtype coefficient = 4.f / (b_channels - 1);  //scale the data

//   // cudaMemset(bottom_diff, 0, outer_num_ * b_channels * sizeof(Dtype));
//   kernel_dot_backward<Dtype><<<CAFFE_GET_BLOCKS(count),
//       CAFFE_CUDA_NUM_THREADS>>>(outer_num_, b_channels, coefficient, bottom_diff, top_diff);
// }
// INSTANTIATE_LAYER_GPU_FUNCS(AverageLayer);
// }