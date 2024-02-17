
#CUDA version = 12.1

apt install libpcl-dev -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install pytorch3d -c pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

pip install -r requirements.txt

cd decoder/utils/furthestPointSampling
pip install .


# https://github.com/sshaoshuai/Pointnet2.PyTorch
cd ../Pointnet2.PyTorch/pointnet2
pip install .

# https://github.com/daerduoCarey/PyTorchEMD
cd ../../PyTorchEMD
pip install .

# used to generate partialized partial inputs
cd ../randPartial
pip install .

cd ../torch-batch-svd
pip install .

# for some reason cc1plus is missing from conda env without installing the following
conda install -c conda-forge gxx_linux-64