
#CUDA version = 12.1

apt install libpcl-dev -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
 
pip install -r requirements.txt

cd decoder/utils/furthestPointSampling
pip install .


# https://github.com/sshaoshuai/Pointnet2.PyTorch
cd decoder/utils/Pointnet2.PyTorch/pointnet2
pip install .

# https://github.com/daerduoCarey/PyTorchEMD
cd decoder/utils/PyTorchEMD
pip install .

# used to generate partialized partial inputs
cd decoder/utils/randPartial
pip install .

cd decoder/utils/torch-batch-svd
pip install .

# for some reason cc1plus is missing from conda env without installing the following
conda install -c conda-forge gxx_linux-64