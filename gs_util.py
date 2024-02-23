import os
import math
import numpy as np
from typing import NamedTuple
import torch
from torch import nn
import open3d as o3d
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import kiui

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

def gs_splatting_render(elevation, azimuth, xyz, shs=None, opacity=None, scale=None, rotation=None):
    """
    Function to render the point cloud using the gaussian splatting
    
    """
    # deal with Don't Care input cases
    if shs is None:
        shs = SH2RGB(np.random.random((xyz.shape[0], 3)) / 255)
        shs = torch.from_numpy(shs).float().cuda()
    if opacity is None:
        opacity = torch.ones((xyz.shape[0], 1), dtype=torch.float32, device="cuda")
    if scale is None:
        scale = torch.ones((xyz.shape[0], 3), dtype=torch.float32, device="cuda") * 0.02  # shape of the gaussian
    if rotation is None:
        rotation = torch.zeros((xyz.shape[0], 4), dtype=torch.float32, device="cuda")   # quaternion
    # to cuda
    xyz = xyz.float().cuda()
    shs = shs.float().cuda()
    opacity = opacity.float().cuda()

    # Create the camera
    pose = orbit_camera(elevation, azimuth, radius=2)

    # MiniCam keeps intrinsic parameters and extrinsic parameters(from pose)
    cur_cam = MiniCam(pose, 256, 256, 0.8569566627292158, 0.8569566627292158, 0.1, 100) # This angle: 0.8569566627292158 in radians is 49 degrees, aligned with zero123

    # background color
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Create zero tensor. We will use it to make pytorch return GRADIENTS of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz,
            dtype=xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(cur_cam.FoVx * 0.5)
    tanfovy = math.tan(cur_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cur_cam.image_height),
        image_width=int(cur_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cur_cam.world_view_transform,
        projmatrix=cur_cam.full_proj_transform,
        sh_degree=0,
        campos=cur_cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    
    # 4-value version
    # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=shs,
    #     colors_precomp=None,
    #     opacities=opacity,
    #     scales=scale,
    #     rotations=rotation,
    #     cov3D_precomp=cov3D_precomp,
    # )

    # rendered_image = rendered_image.clamp(0, 1)
    
    # return rendered_image, rendered_depth, rendered_alpha, screenspace_points, radii

    # 2-value version
    image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scale,
        rotations=rotation,
        cov3D_precomp=cov3D_precomp,
    )
    return image, radii


def save_img(img, path):
    """
    supports: torch tensor (H, W), (3, H, W), (1, H, W)
    """
    if torch.is_tensor(img):
        img = img.detach().cpu()
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        assert len(img.shape) == 3
        img = img * 255
        img = img.type(torch.uint8)
        torchvision.io.write_png(img, path)
    else:
        raise Exception("Not implemented")
    
def save_point_cloud(points, path):
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

if __name__ == "__main__":
    import torchvision
    # initialization taken from DreamGaussian
    pc = o3d.io.read_point_cloud("__tmp__/complete.ply")
    xyz = np.asarray(pc.points)
    xyz = torch.from_numpy(xyz).float()
    sh = np.zeros((xyz.shape[0], 3))
    sh = RGB2SH(sh)

    image, radii = gs_splatting_render(0, 45, xyz.cuda(), shs=torch.from_numpy(sh).float().cuda())
    

    save_img(image, "__tmp__/rendered_image.png")
    # save_point_cloud(screen_points, "logs/screen_points.ply")
    print(radii)



