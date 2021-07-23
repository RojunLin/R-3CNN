#!/usr/bin/env sh
set -e

TOOLS=../../build/tools
LOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./resnext_solver.prototxt --weights=./snapshot/1/resnext50_imagenet.caffemodel

