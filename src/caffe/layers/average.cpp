#include <algorithm>
#include <vector>

#include "caffe/layers/average.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  outer_num_ = bottom[0]->shape(0);  //batchsize
  vector<int> mult_dims = bottom[0]->shape(); //(N,C,H,W)

  mult_dims[1] = 1;  //(N,1,H,W)
  top[0]->Reshape(mult_dims); 
  scale_.ReshapeLike(*bottom[0]);  
  // CHECK_EQ(bottom.size(),2) << "The bottom size is wrong";
}


template <typename Dtype>
void AverageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void AverageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
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


#ifdef CPU_ONLY
STUB_GPU(AverageLayer);
#endif

INSTANTIATE_CLASS(AverageLayer);
// REGISTER_LAYER_CLASS(Average);

}  // namespace caffe
