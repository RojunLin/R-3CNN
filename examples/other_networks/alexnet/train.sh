#!/usr/bin/env sh
set -e

TOOLS=../../../build/tools

#GLOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./alexnet_solver_first_stage.prototxt --weights=./snapshot/1/alexnet_imagenet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./alexnet_R2Net_hinge_solver.prototxt --weights=./snapshot/1/alexnet_R2Net_hinge_iter_0.caffemodel


