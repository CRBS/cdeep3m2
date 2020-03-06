#!/bin/bash
# Demorun2 for CDeep3M
#
# Performs following steps:
# 1) preprocessing of a demodataset to augment the data
# 2) runs a very  short training (100 iterations) with the augmented data
# 3) runs a prediction on the testdata
# This demorun is intended to test the installation is working. For real training on your own data use between 30000 and 50000 iterations.
cdeep3m_folder=`dirname \`which runprediction.sh\``
echo 'Running Demo2 on CDeep3M Version'
cat $cdeep3m_folder/VERSION
echo "Code directory: $cdeep3m_folder"
$cdeep3m_folder/PreprocessTrainingData.py $cdeep3m_folder/mito_testsample/training/images/ $cdeep3m_folder/mito_testsample/training/labels/ $cdeep3m_folder/mito_testsample/mito_test_augmented
$cdeep3m_folder/runtraining.sh --numiterations 50 --snapshot_interval 10 $cdeep3m_folder/mito_testsample/mito_test_augmented $cdeep3m_folder/mito_testsample/demo_trained
$cdeep3m_folder/runprediction.sh $cdeep3m_folder/mito_testsample/demo_trained/ $cdeep3m_folder/mito_testsample/testset/ $cdeep3m_folder/mito_testsample/demo_predictout
$cdeep3m_folder/PlotValidation.py $cdeep3m_folder/mito_testsample/demo_predictout/1fm/log/
$cdeep3m_folder/PlotValidation.py $cdeep3m_folder/mito_testsample/demo_predictout/3fm/log/
$cdeep3m_folder/PlotValidation.py $cdeep3m_folder/mito_testsample/demo_predictout/5fm/log/
