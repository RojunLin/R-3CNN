#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_siamese_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void AccuracySiameseLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_siamese_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracySiameseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
   float rate = 0;
   Dtype multiplier;
   Dtype accuracy;
   for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < inner_num_; j++) {
      multiplier = bottom_data[i * inner_num_ + j] * bottom_label[i * inner_num_ + j ];
      // LOG(INFO)<<"predict["<<i<<"]="<<bottom_data[i*inner_num_+j]<<",and label="<<bottom_label[i*inner_num_+j];
      if (multiplier > Dtype(0.0)){
        rate++;
      }
    }
   }
   accuracy = rate / outer_num_;
   // LOG(INFO) << "Accuracy_Siamese: " << accuracy;
   top[0]->mutable_cpu_data()[0] = accuracy;
   // Accuracy layer should not be used as a loss function.
}



INSTANTIATE_CLASS(AccuracySiameseLayer);
REGISTER_LAYER_CLASS(AccuracySiamese);

}  // namespace caffe
