import torch
import numpy as np
import bpy
from pi3.utils.basic import rotate_target_dim_to_last_axis, load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge

def convert(
    xyz,
    rgb=None,
) -> None:
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
    
    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    return elements


def run_model(target_dir, model, interval=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    print(f"Processing images from {target_dir}")
    imgs = load_images_as_tensor(target_dir, interval=interval).to(device)
    print("Running model inference...")
    dtype = torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension
    # masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    # non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    # masks = torch.logical_and(masks, non_edge)[0]
    print(f"Converting point cloud...")
    res = convert(res['points'][0].cpu(), imgs.permute(0, 2, 3, 1))
    torch.cuda.empty_cache()
    return res

def import_point_cloud(data):
    # Extract positions (x, y, z) from the NumPy array
    positions = np.vstack((data['x'], data['y'], data['z'])).T
    reordered_positions = positions.copy()
    reordered_positions[:, [0, 1, 2]] = positions[:, [0, 2, 1]]
    reordered_positions[:, 2] = -reordered_positions[:, 2]
    reordered_positions[:, [0, 1, 2]] = reordered_positions[:, [1, 0, 2]]
    reordered_positions[:, 0] = -reordered_positions[:, 0]
    positions = reordered_positions
    # Extract colors (red, green, blue), normalize from 0-255 to 0-1, and add alpha channel
    colors = np.vstack((data['red'], data['green'], data['blue'])).T / 255.0
    colors = np.hstack((colors, np.ones((colors.shape[0], 1))))
    
    # Create a new mesh and set vertices
    mesh = bpy.data.meshes.new(name="Points")
    vertices = positions.tolist()
    mesh.from_pydata(vertices, [], [])
    
    # Add color attribute to the mesh
    attribute = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
    color_values = colors.flatten().tolist()
    attribute.data.foreach_set("color", color_values)
    
    # Create a new object and link it to the scene
    obj = bpy.data.objects.new("Points", mesh)
    bpy.context.collection.objects.link(obj)
    
    # Create a material that uses the color attribute
    mat = bpy.data.materials.new(name="PointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "point_color"
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    output_node_material = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node_material.inputs['Surface'])
    
    # Set up Geometry Nodes modifier to convert mesh to points
    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    node_group = bpy.data.node_groups.new(name="PointCloud", type='GeometryNodeTree')
    node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    geo_mod.node_group = node_group
    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')
    mesh_to_points = node_group.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.inputs['Radius'].default_value = 0.001
    set_material_node = node_group.nodes.new('GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = mat
    node_group.links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    node_group.links.new(mesh_to_points.outputs['Points'], set_material_node.inputs['Geometry'])
    node_group.links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])