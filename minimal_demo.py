# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

model_path = 'tencent/Hunyuan3D-2'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)

image_path = '/home/yujunhao/data/yujunhao/projects/Hunyuan3D-2/flux-dev.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

import time

mesh_gen_start_time = time.time()
mesh = pipeline_shapegen(image=image)[0]
mesh_gen_end_time = time.time()

texture_gen_start_time = time.time()
mesh = pipeline_texgen(mesh, image=image)
texture_gen_end_time = time.time()
mesh.export('demo.glb')

print(f"mesh gen time: {mesh_gen_end_time - mesh_gen_start_time}")
print(f"texture gen time: {texture_gen_end_time - texture_gen_start_time}")
print("Total time: ", texture_gen_end_time - mesh_gen_start_time)
