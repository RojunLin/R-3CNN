#!/usr/bin/env sh
set -e

TOOLS=../../build/tools
LOG_logtostderr=0 GLOG_log_dir=./log/1/ $TOOLS/caffe train --solver=./R2Net_lsep_solver.prototxt --weights=./snapshot/1/R2Net_lsep_iter_0.caffemodel

