# Pseudo-LiDAR
<https://github.com/mileyan/pseudo_lidar>.

'3D object detection' is an essential task in autonomous
driving. Recent techniques excel with highly accurate detection rates, provided the 3D input data is obtained from precise but expensive LiDAR technology. Approaches based on cheaper monocular or stereo imagery data have, until now, resulted in drastically lower accuracies — a gap that is commonly attributed to poor image-based depth estimation.

However, this paper argures that it is not the quality of the data but its representation that accounts for the majority of the difference.

## Why

Several recent publications have explored the use of
monocular and stereo depth (disparity) estimation [13, 21,
35] for 3D object detection [5, 6, 24, 33]. However, to-date the main successes have been primarily in supplementing LiDAR approaches. For example, one of the leading algorithms [18] on the KITTI benchmark [11, 12] uses sensor fusion to improve the 3D average precision (AP) for cars from 66% for LiDAR to 73% with LiDAR and monocular images. In contrast, among algorithms that use only images, the state-of-the-art achieves a mere 10% AP [33].

image-based depth is densely estimated for each pixel and often represented as additional image channels [6, 24, 33], making far-away objects smaller and harder to detect. Even worse, pixel neighborhoods in this representation group together points from far-away regions of 3D space. This makes it hard for convolutional networks relying on 2D convolutions on these channels to reason about and precisely localize objects in 3D.

### Data representation matters

- Local patches on 2D images are only coherent physically if they are entirely contained in a single object. If they straddle object boundaries, then two pixels can be co-located next to each other in the depth map, yet can be very far away in 3D space.

- All neighborhoods cannot be operated upon in an identical manner.

  - objects that occur at multiple depths project to different scales in the depth map. A similarly sized patch might capture just a side-view mirror of a nearby car or the entire body of a far-away car. Existing 2D object detection approaches struggle with this breakdown of assumptions and have to design novel techniques such as feature pyramids [19] to deal with this challenge.

**Read up on 3D object detection and 3D convolutions**


## How

1. Estimate Depth
2. Backpropagate to 3D

*Note: Train object detection on pseudo-LiDAR*

In order to be maximally
compatible with existing LiDAR detection pipelines we apply a few additional post-processing steps on the pseudo- LiDAR data. Since real LiDAR signals only reside in a certain range of heights, we disregard pseudo-LiDAR points beyond that range.

we remove all points higher than 1m
above the fictitious LiDAR source (located on top of the autonomous vehicle). As most objects of interest (e.g., cars and pedestrians) do not exceed this height range there is little information loss. In addition to depth, LiDAR also returns the reflectance of any measured pixel (within [0,1]).
As we have no such information, we simply set the reflectance to 1.0 for every pseudo-LiDAR points.

# Conclusion

**No definative answer, must test on our own**

LiDAR and pseudo-LiDAR lead to highly accurate predictions, especially for the nearby objects. However, pseudo-LiDAR fails to detect far-away objects precisely due to inaccurate depth estimates. On the other hand, the frontal-view-based approach makes extremely inaccurate predictions, even for
nearby objects.

- Improvement in camera resolution might help.

- Not tested in real time.
