# CDeep3M 2.0

CDeep3M 2.0:
 * provides a plug-and-play deep learning solution for large-scale image segmentation of light, electron and X-ray microscopy.
 * is distributed as cloud formation template for AWS cloud instances, as docker container and as singularity container for local installs or supercomputer clusters.
 * is backwards compatible, allowing users to continue using models that have been trained with earlier versions of CDeep3M.
 * compared to v1.6.3 provides improvements in speed for larger datasets
 * facilitates additional augmentation strategies (secondary: noise additions, denoising, contrast modifications; tertiary: re-sizing)
 * facilitates providing multiple training volumes to train broadly tuned models.
 * provides enhanced robustness using automated image enhancements.
 * code implemented in Python 3.
 * Generates automatically enhanced images and an overlay of the segmentation with the enhanced images for visual verification.
 