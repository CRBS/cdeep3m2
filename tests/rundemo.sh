#!/bin/bash
# cdeep3m demo for testing
# taken from the nbcr training

# SET gpu_to_use TO USE A SPECIFIC GPU INSTEAD OF ALL GPU(s)
gpu_to_use=

working_folder=~/cdeep3m_demo
sbem_folder=$working_folder/sbem
sample_folder=$working_folder/mito_testsample
output_folder=$working_folder/predictout30k
cdeep3m_folder=`dirname \`which runprediction.sh\``

# FIRST THING IS FIRST, MAKE SURE DEEP3M IS INSTALLED
if [ ! -d "$cdeep3m_folder" ]; then
  echo "Cannot find cdeep3m is it in your PATH?"
  echo "script cannot continue, exiting...."
  exit 1
fi

echo "Working folder is $working_folder"

if [ ! -d "$working_folder" ]; then
  mkdir -p $working_folder
fi

cd $working_folder

if [ ! -d "$sbem_folder" ]; then
  echo "Getting trained model....."
  wget https://s3-us-west-2.amazonaws.com/cdeep3m-trainedmodels/sbem/mitochrondria/xy5.9nm40nmz/sbem_mitochrondria_xy5.9nm40nmz.tar.gz
  tar -xvf sbem_mitochrondria_xy5.9nm40nmz.tar.gz
  rm -f sbem_mitochrondria_xy5.9nm40nmz.tar.gz
  wget https://s3-us-west-2.amazonaws.com/cdeep3m-trainedmodels/sbem/mitochrondria/xy5.9nm40nmz/sbem_mitochrondria_xy5.9nm40nmz_30000iter_trainedmodel.tar.gz
  tar -xvf sbem_mitochrondria_xy5.9nm40nmz_30000iter_trainedmodel.tar.gz
  rm -f sbem_mitochrondria_xy5.9nm40nmz_30000iter_trainedmodel.tar.gz
fi

if [ ! -d "$sample_folder" ]; then
  echo "Getting mito testsample....."
  mito_folder=$cdeep3m_folder/mito_testsample
  if [ ! -d "$mito_folder" ]; then
    echo "Cannot locate cdeep3m mito_testsample, exiting"
    exit 1
  fi
  cp -r $mito_folder $working_folder
fi

#runprediction.sh --gpu=$gpu_to_use $working_folder/sbem/mitochrondria/xy5.9nm40nmz/30000iterations_train_out $working_folder/mito_testsample/testset/ $working_folder/predictout30k
echo "Starting prediction run...."

if [ ! -z "$gpu_to_use" ]; then
  runprediction.sh --gpu=$gpu_to_use $sbem_folder/mitochrondria/xy5.9nm40nmz/30000iterations_train_out $sample_folder/testset/ $output_folder
else
  runprediction.sh $sbem_folder/mitochrondria/xy5.9nm40nmz/30000iterations_train_out $sample_folder/testset/ $output_folder
fi

predict_logs=`ls /tmp/predict_seg_new* 2> /dev/null`

if [ ! -z "$predict_logs" ]; then
  mkdir $working_folder/predict_seg_logs
  cp /tmp/predict_seg_new* $working_folder/predict_seg_logs
  rm -f /tmp/predict_seg_new*
fi

echo "End of demo, check $output_folder for data"