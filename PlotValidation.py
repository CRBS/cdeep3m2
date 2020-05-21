#!/usr/bin/env python3
# Plot Validation
# Generates training vs validation loss and accuracy plots
# Syntax : python3 PlotValidation.py ~/trainingdata/1fm/log
#  => Will create csv files in same log folder and plots of loss and accuracy in same directory

import sys
import os
#import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#matplotlib.use('Agg')

# takes in log file folder path as argument
if len(sys.argv) != 2:
    print("Syntax:\n python3 PlotValidation.py ~/trainingdata/1fm/log\n")
    sys.exit()
else:
    logdir = sys.argv[1]
    if os.path.isdir(logdir):
        print("Parsing log file")
        train_file = os.path.join(logdir, "out.log.train")
        test_file = os.path.join(logdir, "out.log.test")
        os.system("python2 $CAFFE_PATH/tools/extra/parse_log.py {0} {1}".format(os.path.join(logdir, "out.log"), logdir))
    else:
        print("Invalid argument")
        sys.exit()

# gets files paths for training log and testing log files
# column format for train_output csv (NumIters,Seconds,LearningRate,loss_deconv_all)
print("Reading CSV files")
train_df = pd.read_csv(train_file, sep=',', header=0)
train_df['NumIters'] = train_df['NumIters'].astype(int)
train_df['loss_deconv_all'] = train_df['loss_deconv_all'].astype(float)

# column format for test_output csv (NumIters,Seconds,LearningRate,accuracy_conv,class_Acc,loss_deconv_all)
test_df = pd.read_csv(test_file, sep=',', header=0)
test_df['NumIters'] = test_df['NumIters'].astype(int)
test_df['loss_deconv_all'] = test_df['loss_deconv_all'].astype(float)

# plots loss and saves as a pdf and png
plt.plot(train_df['NumIters'], train_df['loss_deconv_all'], label='Training Loss', linewidth=1)
plt.plot(test_df['NumIters'], test_df['loss_deconv_all'], label='Validation Loss', linewidth=1)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(logdir, "loss.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(logdir, "loss.png"), bbox_inches='tight')
plt.close('all')

# plots accuracy and saves as a pdf and png
plt.plot(test_df['NumIters'], test_df['accuracy_conv'], label='Validation Accuracy', linewidth=1)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Validation Accuracy')
plt.grid(True)
plt.savefig(os.path.join(logdir, "accuracy.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(logdir, "accuracy.png"), bbox_inches='tight')
plt.close('all')
