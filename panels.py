import bpy
from .operators import MODEL_PATH
import os

class Pi3Panel(bpy.types.Panel):
    bl_label = "Pi3"
    bl_idname = "VIEW3D_PT_pi3"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Pi3"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        if os.path.exists(MODEL_PATH):
            row.label(text="Model already downloaded")
        else:
            row.operator("pi3.download_model")
        
        layout.prop(scene, "pi3_input_type", expand=True)
        if scene.pi3_input_type == 'FOLDER':
            layout.prop(scene, "pi3_input_folder", text="Input Folder")
        elif scene.pi3_input_type == 'FILE':
            layout.prop(scene, "pi3_input_file", text="Input File")
            layout.prop(scene, "pi3_frame_interval", text="Frame Interval")
        
        row = layout.row()
        row.operator("pi3.generate_point_cloud")