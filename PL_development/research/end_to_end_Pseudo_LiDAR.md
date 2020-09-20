# end-to-end Pseudo LiDAR

<https://github.com/mileyan/pseudo-LiDAR_e2e>

So far these two networks have to be trained separately. In this paper, we introduce a new framework based on differentiable Change of Representation (CoR) modules that allow the entire PL
pipeline to be trained end-to-end.

**PROBLEM: Depth estimators are typically trained with a loss that penalizes errors across all pixels equally, instead of focusing on objects of interest. Consequently, it may over-emphasize nearby or non-object pixels as they are over-represented in the data. Further, if the depth network is trained to estimate disparity, its intrinsic error will be exacerbated for far-away objects.**

To enable back-propagation based end-toend
training on the final loss, the change of representation
(CoR) between the depth estimator and the object detector
must be differentiable with respect to the estimated depth.
We focus on two types of CoR modules — subsampling
and quantization — which are compatible with different
LiDAR-based object detector types.

