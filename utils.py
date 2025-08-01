
import numpy as np
from scipy.spatial.transform import Rotation

def readCameraFromCameraTxt(camera_txt_path):
    """
    Read camera parameters from a COLMAP camera.txt file.
    # 当前版本仅支持所有文件共用一个colmap 相机模型
    """
    with open(camera_txt_path, "r") as f_r:
        line = f_r.readline()
        while line:
            if line.startswith("#"):
                line = f_r.readline()
                continue
            line_arr = line.strip().split(" ")
            if line_arr[1] == "OPENCV":
                camera_model = "OPENCV"
                width = line_arr[2]
                height = line_arr[3]
                f = line_arr[4]
                cx = line_arr[6]
                cy = line_arr[7]
                k1 = line_arr[8]
                k2 = line_arr[9]
                k3 = 0
                p1 = line_arr[10]
                p2 = line_arr[11]
                break
    return {
        "camera_model": camera_model,
        "width": width,
        "height": height,
        "f": f,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "p1": p1,
        "p2": p2
    }

def readImagesFromImagesTxt(images_txt_path, images_prefix=""):
    """
    Read image data from a COLMAP images.txt file.
    """
    Photos = []
    with open(images_txt_path, "r") as f_r:
        line = f_r.readline()
        while line != "":
            if line.startswith("#"):
                line = f_r.readline()
                continue
            line_arr = line.replace("\n", "").split(" ")
            Rotation_ = Rotation.from_quat([float(data_) for data_ in [line_arr[2], line_arr[3], line_arr[4], line_arr[1]]]).as_matrix()
            image_T = np.array([float(x) for x in [line_arr[5], line_arr[6], line_arr[7]]])
            image_T = np.linalg.inv(-Rotation_).dot(image_T.T)
            Photo = {
                "Photo": {
                    "Id": line_arr[0],
                    "ImagePath": f'{images_prefix}{line_arr[-1]}',
                    "Component": "1",
                    "Pose": {
                        "Rotation": {
                            "M_00": Rotation_[0, 0],
                            "M_01": Rotation_[0, 1],
                            "M_02": Rotation_[0, 2],
                            "M_10": Rotation_[1, 0],
                            "M_11": Rotation_[1, 1],
                            "M_12": Rotation_[1, 2],
                            "M_20": Rotation_[2, 0],
                            "M_21": Rotation_[2, 1],
                            "M_22": Rotation_[2, 2]
                        },
                        "Center": {
                            "x": image_T[0],
                            "y": image_T[1],
                            "z": image_T[2]
                        },
                        "Metadata": {
                            "SRSId": "1"
                        }
                    },
                }
            }
            Photos.append(Photo)

            line = f_r.readline()
            line = f_r.readline()  # Skip the next line which is usually empty or a comment
    f_r.close()
    return Photos

def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    return elem
