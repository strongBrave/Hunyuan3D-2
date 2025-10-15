conda create -n hunyuan python=3.10
conda activate hunyuan
conda install pytorch==2.4.0 torchvision==0.19.0 mkl=2023.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install