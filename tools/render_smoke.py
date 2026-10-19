"""Check that Blender can render a frame without a desktop session."""
import bpy
from pathlib import Path
root=Path(__file__).resolve().parent.parent
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=8
scene.render.resolution_x=320
scene.render.resolution_y=240
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.filepath=str(root/'outputs/environment_render_smoke.png')
bpy.ops.render.render(write_still=True)
print('BACKGROUND_RENDER_OK')
