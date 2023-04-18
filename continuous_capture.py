import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import open3d.cpu as o3d_cpu


# define voxel_size in global scope
voxel_size = 4.0 / 512.0

# configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start the RealSense camera
pipeline.start(config)

# create a visualizer object
vis = o3d.visualization.Visualizer()

# start the visualizer
vis.create_window()

# create a voxel grid to integrate the point clouds
tsdf_volume = open3d.cpu.pybind.pipelines.integration.UniformTSDFVolume(arg0: open3d.cpu.pybind.pipelines.integration.UniformTSDFVolume)
     voxel_length=voxel_size,
    sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8

try:
    while True:
        # wait for a new frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # convert the depth frame to a point cloud
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        rs_points = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [[i % depth_intrinsics.width, i // depth_intrinsics.width] for i in range(depth_intrinsics.width * depth_intrinsics.height)], depth_image.flatten()).reshape((-1, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rs_points)

        # integrate the point cloud into the TSDF volume
        tsdf_volume.integrate(
            input=pcd,
            intrinsic=depth_intrinsics,
            extrinsic=np.eye(4),
            depth_scale=1.0
        )

        # add the point cloud to the visualizer
        vis.add_geometry(pcd)

        # render the visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

except KeyboardInterrupt:
    pass

# stop the RealSense camera
pipeline.stop()

# extract the mesh from the TSDF volume
mesh = tsdf_volume.extract_triangle_mesh()

# write the mesh to a file
o3d.io.write_triangle_mesh("reconstruction.ply", mesh)
