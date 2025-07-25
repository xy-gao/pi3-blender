import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import run_model, import_point_cloud

add_on_path = Path(__file__).parent
MODELS_DIR = os.path.join(add_on_path, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.safetensors')
_URL = "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors"
model = None

def get_model():
    global model
    if model is None:
        from pi3.models.pi3 import Pi3
        model = Pi3()
        if os.path.exists(MODEL_PATH):
            from safetensors.torch import load_file
            weight = load_file(MODEL_PATH)
            model.load_state_dict(weight)
        else:
            raise FileNotFoundError("Model file not found. Please download it first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
    return model

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "pi3.download_model"
    bl_label = "Download Pi3 Model"

    def execute(self, context):
        if os.path.exists(MODEL_PATH):
            self.report({'INFO'}, "Model already downloaded.")
            return {'FINISHED'}
        try:
            print("downloading model...")
            torch.hub.download_url_to_file(_URL, MODEL_PATH)
            self.report({'INFO'}, "Model downloaded successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to download model: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return not os.path.exists(MODEL_PATH)

class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "pi3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    def execute(self, context):
        input_type = context.scene.pi3_input_type
        if input_type == 'FOLDER':
            input_path = context.scene.pi3_input_folder
            if not input_path or not os.path.isdir(input_path):
                self.report({'ERROR'}, "Please select a valid input folder.")
                return {'CANCELLED'}
            interval = 1
        elif input_type == 'FILE':
            input_path = context.scene.pi3_input_file
            if not input_path or not os.path.isfile(input_path):
                self.report({'ERROR'}, "Please select a valid input file.")
                return {'CANCELLED'}
            interval = context.scene.pi3_frame_interval
        else:
            self.report({'ERROR'}, "Invalid input type.")
            return {'CANCELLED'}

        try:
            model = get_model()
            predictions = run_model(input_path, model, interval=interval)
            import_point_cloud(predictions)
            self.report({'INFO'}, "Point cloud generated and imported successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        if not os.path.exists(MODEL_PATH):
            return False
        input_type = context.scene.pi3_input_type
        if input_type == 'FOLDER':
            return context.scene.pi3_input_folder != ""
        elif input_type == 'FILE':
            return context.scene.pi3_input_file != ""
        return False