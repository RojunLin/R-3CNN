#!/usr/bin/env sh
set -e

TOOLS=../../../build/tools

GLOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./resnet18_R2Net_hinge_solver.prototxt -weights=./snapshot/1/resnet18_R2Net_hinge_iter_0.caffemodel

#GLOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./resnet18_solver_first_stage.prototxt -weights=./snapshot/1/resnet18_imagenet.caffemodel






