#!/bin/bash
# Copyright 2021 The Anonymous Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


base_dir=$1

echo ${base_dir}

if [ ! -d "$base_dir/pretrained_model" ]; then
        mkdir $base_dir/pretrained_model   
else
   echo "File exists!" 
fi

if [ ! -d "${base_dir}/train" ]; then
        mkdir ${base_dir}/train
else
   echo "File exists!"
fi

if [ ! -d "${base_dir}/test" ]; then
        mkdir ${base_dir}/test
else
   echo "File exists!"
fi


if [ ! -d "${base_dir}/val" ]; then
        mkdir ${base_dir}/val
else
   echo "File exists!"
fi


cd ${base_dir}/pretrained_model
count=`ls *|wc -w`
if [ "$count" > "0" ]; then
  echo "The pretained_model file has been unzipped!"
  cd -
else
  tar -xzvf ${base_dir}/checkpoint.tar.gz
  cd -
fi


cd ${base_dir}/train
count=`ls *|wc -w`
if [ "$count" > "0" ]; then
    echo "The training data has been unzipped!"
    cd -
else
    tar -xzvf ${base_dir}/lidc_crops_train.tar.gz 
    cd -
fi


cd ${base_dir}/test
count=`ls *|wc -w`
if [ "$count" > "0" ]; then
   echo "The testing data has been unzipped!"
   cd -
else
   tar -xzvf ${base_dir}/lidc_crops_test.gz 
   cd -
fi


cd  ${base_dir}/val
count=`ls *|wc -w`
if [ "$count" > "0" ]; then
  echo "The validation data has been unzipped!"
  cd ..
else 
#   tar -xzvf ${base_dir}/lidc_crops_val.tar.gz
cd ..
fi



