import open3d as o3d
import numpy as np
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
pipeline.start(config)

# Get depth sensor and its depth scale
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create geometry for point cloud visualization
pcd = o3d.geometry.PointCloud()

# Create TSDF volume
volume = o3d.pipelines.integration.TSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# Create mesh from the TSDF volume
mesh = volume.extract_triangle_mesh()

# Main loop
while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert depth and color images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Create point cloud from depth and color images
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    points = rs.rs2_deproject_depth_to_point(depth_intrinsics, depth_image, depth_scale)
    colors = np.asarray(color_image)
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Integrate the point cloud into the TSDF volume
    volume.integrate(pcd, o3d.camera.PinholeCameraIntrinsic(depth_intrinsics.width,
                                                           depth_intrinsics.height,
                                                           depth_intrinsics.fx,
                                                           depth_intrinsics.fy,
                                                           depth_intrinsics.ppx,
                                                           depth_intrinsics.ppy))

    # Extract the mesh from the TSDF volume
    mesh = volume.extract_triangle_mesh()

    # Update Open3D visualization
    vis.remove_geometry("pointcloud")
    vis.add_geometry(pcd, name="pointcloud")
    vis.remove_geometry("mesh")
    vis.add_geometry(mesh, name="mesh")
    vis.poll_events()
    vis.update_renderer()

# Stop pipeline
pipeline.stop()
