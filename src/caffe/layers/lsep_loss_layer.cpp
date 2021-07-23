#include <algorithm>
#include <vector>

#include "caffe/layers/lsep_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LSEPLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  if(dim != 1) {
    LOG(FATAL) << "ERROR";
    return;
  }
  for (int i = 0; i < num; ++i) {
    bottom_diff[i] = log(1 + exp(-bottom_data[i] * label[i]));
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.lsep_loss_param().norm()) {
  case LSEPLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case LSEPLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void LSEPLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    // int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i] = (label[i] / exp(bottom_diff[i])) - label[i];
    }
    

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.lsep_loss_param().norm()) {
    case LSEPLossParameter_Norm_L1:
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case LSEPLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }


  }
}

INSTANTIATE_CLASS(LSEPLossLayer);
REGISTER_LAYER_CLASS(LSEPLoss);

}  // namespace caffe

