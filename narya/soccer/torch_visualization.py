from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import kornia
import torch
import cv2


"""
Torch warping function cloned from https://github.com/vcg-uvic/sportsfield_release
with some minor modifications
"""


def normalize_homo(h, **kwargs):
    """Normalize an homography by setting the last coefficient to 1.0
    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the normalized homography
    Raises:
        
    """
    return h / h[2, 2]


def horizontal_flip_homo(h, **kwargs):
    """Apply a horizontal flip to the homography
    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the horizontally flipped homography
    Raises:
        
    """
    flipper = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.matmul(h, flipper)


def vertical_flip_homo(h, **kwargs):
    """Apply a vertical flip to the homography
    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the vertically flipped homography
    Raises:
        
    """
    flipper = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.matmul(h, flipper)


def get_perspective_transform_torch(src, dst):
    """Get the homography matrix between src and dst
    Arguments:
        src: Tensor of shape (B,4,2), the four original points per image
        dst: Tensor of shape (B,4,2), the four corresponding points per image
    Returns:
        A tensor of shape (B,3,3), each homography per image
    Raises:
    """
    return kornia.get_perspective_transform(src, dst)


def get_perspective_transform_cv(src, dst):
    """Get the homography matrix between src and dst
    Arguments:
        src: np.array of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: np.array of shape (B,X,2) or (X,2), the X>3 corresponding points per image
    Returns:
        M: np.array of shape (B,3,3) or (3,3), each homography per image
    Raises:
    """
    if len(src.shape) == 2:
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    else:
        M = []
        for src_, dst_ in zip(src, dst):
            M.append(cv2.findHomography(src_, dst_, cv2.RANSAC, 5)[0])
        M = np.array(M)
    return M


def get_perspective_transform(src, dst, method="cv"):
    """Get the homography matrix between src and dst
    Arguments:
        src: Matrix of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: Matrix of shape (B,X,2) or (X,2), the X>3 corresponding points per image
        method: String in {'cv','torch'} to choose which function to use
    Returns:
        M: Matrix of shape (B,3,3) or (3,3), each homography per image
    Raises:
    """
    return (
        get_perspective_transform_cv(src, dst)
        if method == "cv"
        else get_perspective_transform_torch(src, dst)
    )


def warp_image(img, H, out_shape=None, method="cv"):
    """Apply an homography to a Matrix
    Arguments:
        img: Matrix of shape (B,C,H,W) or (C,H,W)
        H: Matrix of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
        method: String in {'cv','torch'} to choose which function to use
    Returns:
        A Matrix of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    return (
        warp_image_cv(img, H, out_shape=out_shape)
        if method == "cv"
        else warp_image_torch(img, H, out_shape=out_shape)
    )


def warp_image_torch(img, H, out_shape=None):
    """Apply an homography to a torch Tensor
    Arguments:
        img: Tensor of shape (B,C,H,W) or (C,H,W)
        H: Tensor of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
    Returns:
        A Tensor of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    if out_shape is None:
        out_shape = img.shape[-2:]
    if len(img.shape) < 4:
        img = img[None]
    if len(H.shape) < 3:
        H = H[None]
    if img.shape[0] != H.shape[0]:
        raise ValueError(
            "batch size of images ({}) do not match the batch size of homographies ({})".format(
                img.shape[0], H.shape[0]
            )
        )
    batchsize = img.shape[0]
    # create grid for interpolation (in frame coordinates)

    y, x = torch.meshgrid(
        [
            torch.linspace(-0.5, 0.5, steps=out_shape[-2]),
            torch.linspace(-0.5, 0.5, steps=out_shape[-1]),
        ]
    )
    x = x.to(img.device)
    y = y.to(img.device)
    x, y = x.flatten(), y.flatten()

    # append ones for homogeneous coordinates
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy = xy.repeat([batchsize, 1, 1])  # shape: (B, 3, N)
    # warp points to model coordinates
    xy_warped = torch.matmul(H, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # build grid
    grid = torch.stack(
        [
            x_warped.view(batchsize, *out_shape[-2:]),
            y_warped.view(batchsize, *out_shape[-2:]),
        ],
        dim=-1,
    )

    # sample warped image
    warped_img = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros"
    )

    if hasnan(warped_img):
        print("nan value in warped image! set to zeros")
        warped_img[isnan(warped_img)] = 0

    return warped_img


def warp_image_cv(img, H, out_shape=None):
    """Apply an homography to a np.array
    Arguments:
        img: np.array of shape (B,H,W,C) or (H,W,C)
        H: Tensor of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
    Returns:
        A np.array of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    if out_shape is None:
        out_shape = img.shape[-3:-1] if len(img.shape) == 4 else img.shape[:-1]
    if len(img.shape) == 3:
        return cv2.warpPerspective(img, H, dsize=out_shape)
    else:
        if img.shape[0] != H.shape[0]:
            raise ValueError(
                "batch size of images ({}) do not match the batch size of homographies ({})".format(
                    img.shape[0], H.shape[0]
                )
            )
        out_img = []
        for img_, H_ in zip(img, H):
            out_img.append(cv2.warpPerspective(img_, H_, dsize=out_shape))
        return np.array(out_img)


def warp_point(pts, homography, method="cv"):
    return (
        warp_point_cv(pts, homography)
        if method == "cv"
        else warp_point_torch(pts, homography)
    )


def warp_point_cv(pts, homography):
    dst = cv2.perspectiveTransform(np.array(pts).reshape(-1, 1, 2), homography)
    return dst[0][0]


def warp_point_torch(pts, homography, input_shape = (320,320,3)):
    img_test = np.zeros(input_shape)
    dir_ = [0, -1, 1, -2, 2, 3, -3]
    for dir_x in dir_:
        for dir_y in dir_:
            to_add_x = min(max(0, pts[0] + dir_x), input_shape[0]-1)
            to_add_y = min(max(0, pts[1] + dir_y), input_shape[1]-1)
            for i in range(3):
                img_test[to_add_y, to_add_x, i] = 1.0

    pred_warp = warp_image(
        np_img_to_torch_img(img_test), to_torch(homography), method="torch"
    )
    pred_warp = torch_img_to_np_img(pred_warp[0])
    indx = np.argwhere(pred_warp[:, :, 0] > 0.8)
    x, y = indx[:, 0].mean(), indx[:, 1].mean()
    dst = np.array([y, x])
    return dst


def get_default_corners(batch_size):
    """Get coordinates of the default corners in a soccer field
    Arguments:
        batch_size: Integer, the number of time we need the corners
    Returns:
        orig_corners: a np.array of len(batch_size)
    Raises:
        
    """
    orig_corners = np.array(
        [[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32
    )
    orig_corners = np.tile(orig_corners, (batch_size, 1, 1))
    return orig_corners


def get_corners_from_nn(batch_corners_pred):
    """Gets the corners in the right shape, from a DeepHomoModel
    Arguments:
        batch_corners_pred: np.array of shape (B,8) with the predictions
    Returns:
        corners: np.array of shape (B,4,2) with the corners in the right shape
    Raises:
        
    """
    batch_size = batch_corners_pred.shape[0]
    corners = np.reshape(batch_corners_pred, (-1, 2, 4))
    corners = np.transpose(corners, axes=(0, 2, 1))
    corners = np.reshape(corners, (batch_size, 4, 2))
    return corners


def compute_homography(batch_corners_pred):
    """Compute the homography from the predictions of DeepHomoModel
    Arguments:
        batch_corners_pred: np.array of shape (B,8) with the predictions
    Returns:
        np.array of shape (B,3,3) with the homographies
    Raises:
        
    """
    batch_size = batch_corners_pred.shape[0]
    corners = get_corners_from_nn(batch_corners_pred)
    orig_corners = get_default_corners(batch_size)
    homography = get_perspective_transform_torch(
        to_torch(orig_corners), to_torch(corners)
    )
    return to_numpy(homography)


def get_four_corners(homo_mat):
    """Inverse operation of compute_homography. Gets the 4 corners from an homography.
    Arguments:
        homo_mat: Matrix of shape (B,3,3) or (3,3), homographies
    Returns:
        xy_warped: np.array of shape (B,4,2) with the corners
    Raises:
        ValueError: If the homographies are not of shape (3,3)
    """
    if isinstance(homo_mat, np.ndarray):
        homo_mat = to_torch(homo_mat)

    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    if homo_mat.shape[1:] != (3, 3):
        raise ValueError(
            "The shape of the homography is {}, not (3,3)".format(homo_mat.shape[1:])
        )

    canon4pts = to_torch(
        np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)
    )

    assert canon4pts.shape == (4, 2)
    x, y = canon4pts[:, 0], canon4pts[:, 1]
    xy = torch.stack([x, y, torch.ones_like(x)])
    # warp points to model coordinates
    xy_warped = torch.matmul(homo_mat, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    xy_warped = xy_warped / (z_warped + 1e-8)
    xy_warped = to_numpy(xy_warped)
    return xy_warped


def torch_img_to_np_img(torch_img):
    """Convert a torch image to a numpy image
    Arguments:
        torch_img: Tensor of shape (B,C,H,W) or (C,H,W)
    Returns:
        a np.array of shape (B,H,W,C) or (H,W,C)
    Raises:
        ValueError: If this is not a Torch tensor
    """
    if isinstance(torch_img, np.ndarray):
        return torch_img
    assert isinstance(torch_img, torch.Tensor), "cannot process data type: {0}".format(
        type(torch_img)
    )
    if len(torch_img.shape) == 4 and (
        torch_img.shape[1] == 3 or torch_img.shape[1] == 1
    ):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (
        torch_img.shape[0] == 3 or torch_img.shape[0] == 1
    ):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError("cannot process this image")


def np_img_to_torch_img(np_img):
    """Convert a np image to a torch image
    Arguments:
        np_img: a np.array of shape (B,H,W,C) or (H,W,C)
    Returns:
        a Tensor of shape (B,C,H,W) or (C,H,W)
    Raises:
        ValueError: If this is not a np.array
    """
    if isinstance(np_img, torch.Tensor):
        return np_img
    assert isinstance(np_img, np.ndarray), "cannot process data type: {0}".format(
        type(np_img)
    )
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return to_torch(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return to_torch(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return to_torch(np_img)
    else:
        raise ValueError("cannot process this image")


def normalize_single_image_torch(image, img_mean=None, img_std=None):
    """Normalize a Torch tensor
    Arguments:
        image: Torch Tensor of shape (C,W,H)
        img_mean: List of mean per channel (e.g.: [0.485, 0.456, 0.406])
        img_std: List of std per channel (e.g.: [0.229, 0.224, 0.225])
    Returns:
        image: Torch Tensor of shape (C,W,H), the normalized image
    Raises:
        ValueError: If the shape of the image is not of lenth 3
        ValueError: If the image is not a torch Tensor
    """
    if len(image.shape) != 3:
        raise ValueError(
            "The len(shape) of the image is {}, not 3".format(len(image.shape))
        )
    if isinstance(image, torch.Tensor) == False:
        raise ValueError("The image is not a torch Tensor")
    if img_mean is None and img_std is None:
        img_mean = torch.mean(image, dim=(1, 2)).view(-1, 1, 1)
        img_std = image.contiguous().view(image.size(0), -1).std(-1).view(-1, 1, 1)
        image = (image - img_mean) / img_std
    else:
        image = Normalize(img_mean, img_std, inplace=False)(image)
    return image


def denormalize(x):
    """Scale image to range [0,1]
    Arguments:
        x: np.array, an image
    Returns:
        x: np.array, the scaled image
    Raises:
    """
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def to_numpy(var):
    """Parse a Torch variable to a numpy array
    Arguments:
        var: torch variable
    Returns:
        a np.array with the same value as var
    Raises:
        
    """
    try:
        return var.numpy()
    except:
        return var.detach().numpy()


def to_torch(np_array):
    """Parse a numpy array to a torch variable
    Arguments:
        np_array: a np.array 
    Returns:
        a torch Var with the same value as the np_array
    Raises:
        
    """
    tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)
import six
import numpy as np

FLIP_MAPPER = {
    0: 13,
    1: 14,
    2: 15,
    3: 16,
    4: 17,
    5: 18,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 10,
    11: 11,
    12: 12,
    13: 0,
    14: 1,
    15: 2,
    16: 3,
    17: 4,
    18: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 27,
    24: 28,
    25: 25,
    26: 26,
    27: 23,
    28: 24,
}


def _get_flip_mapper():
    return FLIP_MAPPER


INIT_HOMO_MAPPER = {
    0: [3, 3],
    1: [3, 66],
    2: [51, 65],
    3: [3, 117],
    4: [17, 117],
    5: [3, 203],
    6: [17, 203],
    7: [3, 255],
    8: [51, 254],
    9: [3, 317],
    10: [160, 3],
    11: [160, 160],
    12: [160, 317],
    13: [317, 3],
    14: [317, 66],
    15: [270, 66],
    16: [317, 118],
    17: [304, 118],
    18: [317, 203],
    19: [304, 203],
    20: [317, 255],
    21: [271, 255],
    22: [317, 317],
    23: [51, 128],
    24: [51, 193],
    25: [161, 118],
    26: [161, 203],
    27: [270, 128],
    28: [269, 192],
}


def _get_init_homo_mapper():
    return INIT_HOMO_MAPPER


def _flip_keypoint(id_kp, x_kp, y_kp, input_shape=(320, 320, 3)):
    """Flip the keypoints verticaly, according to the shape of the image
    Arguments:
        id_kep: Integer, the id of the keypoint
        x_kp, y_kp: the x,y coordinates of the keypoint
        input_shapes: Tuple, the shape of the image concerned with the keypoint
    Returns:
        new_id_kp, x_kp, new_y_kp: Tuple of integer with the flipped id and coordinates
    Raises:
        ValueError: If the id_kp is not in the list of Id
        ValueError: If the y coordinates is larger than the input_shape, or smaller than 0
    """

    if id_kp not in FLIP_MAPPER.keys():
        raise ValueError("Keypoint id {} not in the flip mapper".format(id_kp))
    if y_kp < 0 or y_kp > input_shape[0] - 1:
        raise ValueError(
            "y_kp = {}, outside of range [0,{}]".format(y_kp, input_shape[0] - 1)
        )

    new_id_kp = FLIP_MAPPER[id_kp]
    new_y_kp = input_shape[0] - 1 - y_kp

    return (new_id_kp, x_kp, new_y_kp)


def _add_mask(mask, val, x, y):
    """Takes a mask, and add a new segmentation with the value val, around the (x,y) coordinates
    Arguments:
        mask: np.array, the mask
        val: The value to add to the mask 
        x,y: the coordinates of the segmentation to add
    Returns:
        
    Raises:
        
    """
    dir_x = [0, -1, 1]
    dir_y = [0, -1, 1]
    for d_x in dir_x:
        for d_y in dir_y:
            new_x = min(max(x + d_x, 0), mask.shape[0]-1)
            new_y = min(max(y + d_y, 0), mask.shape[1]-1)
            mask[new_x][new_y] = val


def _build_mask(keypoints, mask_shape=(320, 320), nb_of_mask=29):
    """From a dict of keypoints, creates a list of mask with keypoint segmentation
    Arguments:
        keypoints: Dict, mapping each keypoint id to its location
        mask_shape: Shape of the mask to be created
        nb_of_mask: Number of mask to create (= number of different keypoints)
    Returns:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
    Raises:
    """
    mask = np.ones((mask_shape)) * nb_of_mask
    for id_kp, v in six.iteritems(keypoints):
        _add_mask(mask, id_kp, v[0], v[1])
    return mask


def _get_keypoints_from_mask(mask, treshold=0.9):
    """From a list of mask, compute the mapping of each keypoints to their location
    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        keypoints: Dict, mapping each keypoint id to its location
    Raises:
        
    """
    keypoints = {}
    indexes = np.argwhere(mask[:, :, :-1] > treshold)
    for indx in indexes:
        id_kp = indx[2]
        if id_kp in keypoints.keys():
            keypoints[id_kp][0].append(indx[0])
            keypoints[id_kp][1].append(indx[1])
        else:
            keypoints[id_kp] = [[indx[0]], [indx[1]]]

    for id_kp in keypoints.keys():
        mean_x = np.mean(np.array(keypoints[id_kp][0]))
        mean_y = np.mean(np.array(keypoints[id_kp][1]))
        keypoints[id_kp] = [mean_y, mean_x]
    return keypoints

def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon

def _points_from_mask(mask, treshold=0.9):
    """From a list of mask, compute src and dst points from the image and the 2D view of the image
    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        src_pts, dst_pts: Location of src and dst related points
    Raises:
        
    """
    list_ids = []
    src_pts, dst_pts = [], []
    available_keypoints = _get_keypoints_from_mask(mask, treshold)
    for id_kp, v in six.iteritems(available_keypoints):
        src_pts.append(v)
        dst_pts.append(INIT_HOMO_MAPPER[id_kp])
        list_ids.append(id_kp)
    src, dst = np.array(src_pts), np.array(dst_pts)

    ### Final test : return nothing if 3 points are colinear and the src has just 4 points 
    test_colinear = False
    if len(src) == 4:
        if collinear(dst_pts[0], dst_pts[1], dst_pts[2]) or collinear(dst_pts[0], dst_pts[1], dst_pts[3]) or collinear(dst_pts[1], dst_pts[2], dst_pts[3]) :
          test_colinear = True
    src = np.array([]) if test_colinear else src
    dst = np.array([]) if test_colinear else dst
    
    return src, dst