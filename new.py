import pyrealsense2 as rs
import open3d as o3d
import numpy as np

#Configure the camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Define camera intrinsic parameters
width, height = 640, 480
cx, cy = 320, 240
fx, fy = 385.0718994140625, 385.0718994140625
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

#Create an Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(o3d.geometry.PointCloud())
vis.get_render_option().background_color = np.array([0, 0, 0])

#Continuously capture frames and update the point cloud
while True:
    # Capture frames
    frames = pipeline.wait_for_frames()

    # Generate point cloud from depth data
    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    pcd = o3d.geometry.PointCloud()
    for v in range(depth_intrin.height):
        for u in range(depth_intrin.width):
            depth_value = depth_scale * depth_image[v, u]
            if depth_value > 0:
                pt3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_value)
                pcd.points.append(pt3d)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
    
    # Update point cloud in visualizer
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Check for key press to exit the loop
    if vis.poll_events():
        if vis.get_window_status() == o3d.visualization.WindowStatus.Closed:
            break
#Stop the camera pipeline and close the visualizer
pipeline.stop()
vis.destroy_window()

