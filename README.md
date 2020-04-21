[speedup]: https://github.com/CRBS/cdeep3m2/wiki/Speed-up-processing-time
[validation]: https://github.com/CRBS/cdeep3m2/wiki/Add-Validation-to-training
[transferlearning]: https://github.com/CRBS/cdeep3m2/wiki/TransferLearning
[cdeep3mbiorxiv]: https://www.biorxiv.org/content/early/2018/06/21/353425
[cdeep3mnaturemethods]: https://rdcu.be/5zIF
[dockercdeep3m]: https://hub.docker.com/r/ncmir/cdeep3m
[dockerDOCcdeep3m]: https://github.com/CRBS/cdeep3m2/wiki/CDeep3M-Docker
[colabcdeep3m]: https://github.com/haberlmatt/cdeep3m-colab
[previewcdeep3m]: https://cdeep3m.crbs.ucsd.edu/cdeep3m
[previewFAQcdeep3m]: https://cdeep3m.crbs.ucsd.edu/home/faq
[AWScdeep3m]: https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=cdeep3m-stack-py3-docker&templateURL=https://cf-templates-1i8oypshb6jhq-us-west-2.s3-us-west-2.amazonaws.com/cloud_formation_cdeep3m_py3-docker.json
[AWSdoc]: https://github.com/CRBS/cdeep3m2/wiki/CDeep3M2-on-AWS-(via-Cloud-Formation-Template)
[CDeep3m-simg]: https://doi.org/10.7295/W9CDEEP3M_SINGULARITY
[singularity]: https://github.com/CRBS/cdeep3m2/wiki/Singularity

# CDeep3M2 overview:

<img align="right" width="100" height="50" src="/media/cdeep3m_logo_1-01.png">

 * provides a plug-and-play deep learning solution for large-scale image segmentation of light, electron and X-ray microscopy.
 * is distributed as cloud formation template for AWS cloud instances, as docker container and as singularity container for local installs or supercomputer clusters.
 * is backwards compatible, allowing users to continue using models that have been trained with earlier versions of CDeep3M.
 * compared to v1.6.3 provides improvements in speed for larger datasets
 * facilitates additional augmentation strategies (secondary: noise additions, denoising, contrast modifications; tertiary: re-sizing)
 * facilitates providing multiple training volumes to train broadly tuned models.
 * provides enhanced robustness using automated image enhancements.
 * code implemented in Python 3.
 * Generates automatically enhanced images and an overlay of the segmentation with the enhanced images for visual verification.

## Running CDeep3M2

|  |  Use | Description | Link | Documentation
| ------ | ------ | ------ | ------ | ------ |
|  | **CDeep3M2-Preview:** | Extremely quick tests, fully automated instantaneous runs | [Link][previewcdeep3m] | [Documentation][previewFAQcdeep3m] |
|  | **CDeep3M2-Docker:** | Local or remote, large runs, long trainings, simple installation, GPU with min 12GB vRAM required | [Link][dockercdeep3m] | [Documentation][dockerDOCcdeep3m] |
|  | **CDeep3M2-AWS:** | Remote, large runs, long trainings, simple installation, pay for GPU/hour (entry level 0.50$/h) | [Link][AWScdeep3m] | [Documentation][AWSdoc] |
|  | **CDeep3M2-Colab:**  | Remote, short runs or re-training, simple installation, free GPU access | [Link][colabcdeep3m] | [Documentation][colabcdeep3m] |
|  | **CDeep3M2-Singularity:** | Local or cluster, large runs, long trainings, often required for compute cluster | [Link][CDeep3m-simg] | [Documentation][singularity] |


## Steps to process images with CDeep3M2:

* [Training: train the CNN to recognize objects in 3D image stacks (or 2D images)](https://github.com/CRBS/cdeep3m2/wiki/PreprocessTrainingData.py-and-runtraining.sh)

* [Prediction: image segmentation](https://github.com/CRBS/cdeep3m2/wiki/runprediction.sh)

* [Transfer Learning: adapt a pre-trained model to a new dataset or different object](https://github.com/CRBS/cdeep3m2/wiki/Transfer-Learning)

For prediction and transfer learning you can use either a [pre-trained model from the modelzoo](https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models) or your own model generated during training.
