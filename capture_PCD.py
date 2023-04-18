import open3d as o3d
import pyrealsense2 as rs
import numpy as np

# configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# start the RealSense camera
pipeline.start(config)

# create a visualizer object
vis = o3d.visualization.Visualizer()

# start the visualizer
vis.create_window()

try:
    while True:
        # wait for a new frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # convert the depth frame to a point cloud
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            image=o3d.geometry.Image(np.array(depth_frame.get_data())),
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=depth_frame.width,
                height=depth_frame.height,
                fx=depth_intrinsics[0],
                fy=depth_intrinsics[1],
                cx=depth_intrinsics[2],
                cy=depth_intrinsics[3]
            ),
            extrinsic=np.eye(4)
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

# close the visualizer window
vis.destroy_window()
