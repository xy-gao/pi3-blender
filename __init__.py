import bpy
from .dependencies import Dependencies

bl_info = {
    "name": "Pi3 Addon",
    "author": "Xiangyi Gao",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Pi3",
    "description": "Generate point clouds from images using Pi3",
    "category": "3D View",
}

def register():
    if not Dependencies.check():
        Dependencies.install()
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.register_class(operators.DownloadModelOperator)
        bpy.utils.register_class(operators.GeneratePointCloudOperator)
        bpy.utils.register_class(panels.Pi3Panel)
        bpy.types.Scene.pi3_input_type = bpy.props.EnumProperty(
            items=[('FOLDER', 'Folder', ''), ('FILE', 'File', '')],
            default='FOLDER'
        )
        bpy.types.Scene.pi3_input_folder = bpy.props.StringProperty(subtype='DIR_PATH')
        bpy.types.Scene.pi3_input_file = bpy.props.StringProperty(subtype='FILE_PATH')
        bpy.types.Scene.pi3_frame_interval = bpy.props.IntProperty(
            name="Frame Interval",
            description="Interval between frames extracted",
            default=30,
            min=1
        )
    else:
        raise ValueError("installation failed.")

def unregister():
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.unregister_class(operators.DownloadModelOperator)
        bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
        bpy.utils.unregister_class(panels.Pi3Panel)
        del bpy.types.Scene.pi3_input_type
        del bpy.types.Scene.pi3_input_folder
        del bpy.types.Scene.pi3_input_file
        del bpy.types.Scene.pi3_frame_interval

if __name__ == "__main__":
    register()