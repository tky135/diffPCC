
#CUDA version = 12.1

apt install libpcl-dev -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
 
pip install -r requirements.txt

cd decoder/utils/furthestPointSampling
python3 setup.py install


# https://github.com/sshaoshuai/Pointnet2.PyTorch
cd decoder/utils/Pointnet2.PyTorch/pointnet2
python3 setup.py install

# https://github.com/daerduoCarey/PyTorchEMD
cd decoder/utils/PyTorchEMD
python3 setup.py install

# used to generate partialized partial inputs
cd decoder/utils/randPartial
python3 setup.py install

# for some reason cc1plus is missing from conda env without installing the following
conda install -c conda-forge gxx_linux-64