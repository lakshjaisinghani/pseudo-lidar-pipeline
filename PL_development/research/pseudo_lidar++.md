# Pseudo LiDAR V2

<https://github.com/mileyan/Pseudo_Lidar_V2>

## Process

- Create Depth Map

- Create Pseudo LiDAR point cloud

- Create a KNN graph

- Use sparse LiDAR point clouds to bias and correct depth (in pseudo LiDAR)


## Cons

- Non differentiable conversion from Depth to point cloud.

- Object detection loss and Depth loss are treated separately.
