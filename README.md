# CDeep3M Version 2
[speedup]: https://giterdone.crbs.ucsd.edu/ncmir/cdeep3m_py3/-/wikis/Speed-up-processing-time
[validation]: https://giterdone.crbs.ucsd.edu/ncmir/cdeep3m_py3/-/wikis/Add-Validation-to-training
[transferlearning]: https://giterdone.crbs.ucsd.edu/ncmir/cdeep3m_py3/-/wikis/TransferLearning
[cdeep3mbiorxiv]: https://www.biorxiv.org/content/early/2018/06/21/353425
[cdeep3mnaturemethods]: https://rdcu.be/5zIF
[dockercdeep3m]: https://hub.docker.com/r/ncmir/cdeep3m



## CDeep3M2 overview:

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

|  Use | Description |
| ------ | ------ |
| CDeep3M2-Preview: | Extremely quick tests, fully automated instantaneous runs |
| CDeep3M2-Docker: | Local or remote, large runs, long trainings, simple installation, GPU access required | 
| CDeep3M2-AWS: | Remote, large runs, long trainings, simple installation, pay for GPU/hour (entry level 0.50$/h) | 
| CDeep3M2-Colab:  | Remote, short runs or re-training, simple installation, free GPU access | 
| CDeep3M2-Singularity: | Local or cluster, large runs, long trainings, often required for compute cluster | 
