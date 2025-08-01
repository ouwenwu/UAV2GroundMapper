import argparse
import math
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

from utils import indent, readCameraFromCameraTxt, readImagesFromImagesTxt
# 从colmap文件中获取图像的XYZ坐标
# colmap_dir: colmap文件夹路径
def getImageXYZFromColmapTxt(args):
    values = vars(args)
    colmap_dir = values['colmap_dir']
    output_path = values['output_path']
    images_path = f"{colmap_dir}/images.txt"
    f_w = open(output_path, "w")
    with open(images_path, "r") as f_r:
        line = f_r.readline()
        while line != "":
            if line.startswith("#"):
                line = f_r.readline()
                continue
            line_arr = line.replace("\n", "").split(" ")
            image_name = line_arr[9]
            qw, qx, qy, qz = map(float, line_arr[1:5])
            tx, ty, tz = map(float, line_arr[5:8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            loc = np.linalg.inv(-R) @ t.T
            f_w.write(f"{image_name.replace('.png', '.jpg')} {loc[0]} {loc[1]} {loc[2]} 0 0 0 1\n")
            line = f_r.readline()
            line = f_r.readline()
    f_r.close()
    f_w.close()

def from_photoscan_xml_2_colmap_txt(args):
    values = vars(args)
    xml_path = values['xml_path']
    output_dir = values['output_path']
    target_image_path = values['target_image_path']
    tree = ET.parse(xml_path)   
    root = tree.getroot()
    Block = root.find("Block")
    Photogroups = Block.find("Photogroups")

    camera_write = open(f"{output_dir}/cameras.txt", "w")
    camera_write.write('# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, '
                       'HEIGHT, PARAMS[]\n# Number of cameras: 1\n')
    images_write = open(f"{output_dir}/images.txt", "w")
    images_write.write('# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID'
                       ', NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: 1, mean observations '
                       'per image: 2370.29\n')
    points3D_write = open(f"{output_dir}/points3D.txt", "w")
    points3D_write.write('# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, '
                         'TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: 1, mean track length: 4.18148\n')
    
    image_id = 1
    for index, Photogroup in enumerate(Photogroups.findall("Photogroup")):
        ImageDimensions = Photogroup.find("ImageDimensions")
        camerasWidth = int(ImageDimensions.find("Width").text)
        camerasHeight = int(ImageDimensions.find("Height").text)

        f_array = Photogroup.find("FocalLengthPixels")
        if f_array is None:
            sensor_diag_mm = Photogroup.find("SensorSize").text
            sensor_diag_mm = float(sensor_diag_mm)
            focal_mm = float(Photogroup.find("FocalLength").text)
            aspect_ratio = camerasWidth / camerasHeight
            sensor_height_mm = sensor_diag_mm / math.sqrt(1 + aspect_ratio ** 2)
            sensor_width_mm = sensor_height_mm * aspect_ratio
            f_x = focal_mm / sensor_width_mm * camerasWidth
            f_y = focal_mm / sensor_height_mm * camerasHeight
            f = (f_x + f_y) / 2
        else:
            f = float(f_array.text)
        PrincipalPoint = Photogroup.find("PrincipalPoint")
        PrincipalPoint_x = float(PrincipalPoint.find("x").text)
        PrincipalPoint_y = float(PrincipalPoint.find("y").text)
        Distortion = Photogroup.find("Distortion")
        Distortion_k1 = Distortion.find("K1").text
        Distortion_k2 = Distortion.find("K2").text
        Distortion_k3 = Distortion.find("K3").text
        Distortion_P1 = Distortion.find("P1").text
        Distortion_P2 = Distortion.find("P2").text
        camera_write.write(f'{index+1} OPENCV {camerasWidth} {camerasHeight} {f} {f} {PrincipalPoint_x} '
                           f'{PrincipalPoint_y} {Distortion_k1} {Distortion_k2} {Distortion_P1} {Distortion_P2}\n')
        for Photo in Photogroup.findall("Photo"):
            image_path = Photo.find("ImagePath").text
            image_name = image_path.split("/")[-1]
            if target_image_path is not None:
                shutil.copy(image_path, f"{target_image_path}/{image_name}")
            Pose = Photo.find("Pose")
            Rotation_ = Pose.find("Rotation")
            M = []
            for m in ['M_00', 'M_01', 'M_02', 'M_10', 'M_11', 'M_12', 'M_20', 'M_21', 'M_22']:
                M.append(float(Rotation_.find(m).text))
            R = np.zeros([3, 3], dtype=float)
            for i in range(3):
                for j in range(3):
                    R[i, j] = M[i * 3 + j]
            quat = Rotation.from_matrix(R).as_quat()
            C = np.zeros(3, dtype=float)
            C[0] = float(Pose.find("Center").find("x").text)
            C[1] = float(Pose.find("Center").find("y").text)
            C[2] = float(Pose.find("Center").find("z").text)
            T = np.dot(-R, C)
            images_write.write(f'{image_id} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {T[0]} {T[1]} {T[2]} {index+1} {image_name}\n')
            images_write.write('0.0 0.0 -1\n')
            image_id += 1
    camera_write.close()
    images_write.close()
    points3D_write.close()

def rotate_image_xyz(args):
    values = vars(args)
    image_xyz_path = values['image_xyz']
    image_xyz_path_out = values['output_path']
    matrix_path = values['matrix_path']
    matrixs = []
    matrix_temp = []
    with open(matrix_path, "r") as f_r:
        line = f_r.readline()
        while line != "":
            if line == "\n":
                matrixs.append(np.array(matrix_temp))
                matrix_temp = []
                line = f_r.readline()
                continue
            line_arr = line.replace("\n", "").split(" ")
            matrix_temp.append([float(data_) for data_ in line_arr])
            line = f_r.readline()
    matrixs.append(np.array(matrix_temp))
    f_r.close()
    with open(image_xyz_path, "r") as f_r:
        with open(image_xyz_path_out, "w") as f_w:
            line = f_r.readline()
            while line != "":
                line_arr = line.replace("\n", "").split(" ")
                image_name = line_arr[0]
                loc = [float(data_) for data_ in line_arr[1:4]]
                loc = np.array(loc + [1])
                for matrix in matrixs:
                    loc = np.dot(matrix, loc)
                
                f_w.write(f"{image_name} {loc[0]} {loc[1]} {loc[2]}\n")
                line = f_r.readline()
        f_w.close()

def write_colmap_txt_2_photoscan_xml(args):
    values = vars(args)
    colmap_txt_path = values['colmap_txt_path']
    output_xml_path = values['output_xml_path']
    image_path = values['image_path']    
    # 这里需要实现将colmap txt转换为photoscan xml的逻辑
    # 由于具体的转换逻辑未提供，这里仅作为占位符
    # 实际实现需要根据colmap txt的格式和photoscan xml的要求进行转换
    camera_params = readCameraFromCameraTxt(f'{colmap_txt_path}/cameras.txt')
    print(f"读取到相机参数: {camera_params}")
    Photos = readImagesFromImagesTxt(f'{colmap_txt_path}/images.txt', image_path)
    print(f"读取到{len(Photos)}张图像")
    tree = ET.parse("./point.xml")
    root = tree.getroot()
    Block = root.find("Block")
    # 前面的配置信息
    Name = ET.Element("Name")
    Name.text = "Chunk 2"
    Block.append(Name)

    Description = ET.Element("Description")
    Description.text = "Result of aerotriangulation of Chunk 2 (2023-Dec-04 21:54:45)"
    Block.append(Description)

    SRSId = ET.Element("SRSId")
    SRSId.text = "1"
    Block.append(SRSId)

    Photogroups = ET.Element("Photogroups")
    Photogroup = ET.SubElement(Photogroups, "Photogroup")
    Name = ET.SubElement(Photogroup, "Name")
    Name.text = "unknown"

    ImageDimensions = ET.SubElement(Photogroup, "ImageDimensions")
    Width = ET.SubElement(ImageDimensions, "Width")
    Width.text = camera_params["width"]
    Height = ET.SubElement(ImageDimensions, "Height")
    Height.text = camera_params["height"]

    CameraModelType = ET.SubElement(Photogroup, "CameraModelType")
    CameraModelType.text = "Perspective"
    CameraModelBand = ET.SubElement(Photogroup, "CameraModelBand")
    CameraModelBand.text = "Visible"

    FocalLengthPixels = ET.SubElement(Photogroup, "FocalLengthPixels")
    FocalLengthPixels.text = camera_params["f"]

    PrincipalPoint = ET.SubElement(Photogroup, "PrincipalPoint")
    x = ET.SubElement(PrincipalPoint, "x")
    x.text = camera_params["cx"]
    y = ET.SubElement(PrincipalPoint, "y")
    y.text = camera_params["cy"]

    Distortion = ET.SubElement(Photogroup, "Distortion")
    K1 = ET.SubElement(Distortion, "K1")
    K1.text = str(camera_params["k1"])
    K2 = ET.SubElement(Distortion, "K2")
    K2.text = str(camera_params["k2"])
    K3 = ET.SubElement(Distortion, "K3")
    K3.text = str(camera_params["k3"])
    P1 = ET.SubElement(Distortion, "P1")
    P1.text = str(camera_params["p1"])
    P2 = ET.SubElement(Distortion, "P2")
    P2.text = str(camera_params["p2"])

    AspectRatio = ET.SubElement(Photogroup, "AspectRatio")
    AspectRatio.text = "1"
    Skew = ET.SubElement(Photogroup, "Skew")
    Skew.text = "0"

    for Photo_Info in Photos:
        Photo_Info = Photo_Info["Photo"]
        Photo = ET.SubElement(Photogroup, "Photo")
        Id = ET.SubElement(Photo, "Id")
        Id.text = Photo_Info["Id"]
        ImagePath = ET.SubElement(Photo, "ImagePath")
        ImagePath.text = Photo_Info["ImagePath"]

        Pose = ET.SubElement(Photo, "Pose")
        Rotation = ET.SubElement(Pose, "Rotation")
        Rotation_ = Photo_Info["Pose"]["Rotation"]

        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                temp = ET.SubElement(Rotation, f"M_{i}{j}")
                temp.text = str(Rotation_[f"M_{i}{j}"])

        Center = ET.SubElement(Pose, "Center")
        Center_ = Photo_Info["Pose"]["Center"]
        for i in ["x", "y", "z"]:
            temp = ET.SubElement(Center, f"{i}")
            temp.text = str(Center_[f"{i}"])

        Metadata = ET.SubElement(Pose, "Metadata")
        Metadata_ = Photo_Info["Pose"]["Metadata"]
        SRSId = ET.SubElement(Metadata, "SRSId")
        SRSId.text = Metadata_["SRSId"]

        Pose = ET.SubElement(Photo, "ExifData")
    Block.append(Photogroups)
    indent(root)
    tree.write(output_xml_path)

def merge_photoscan_xml(args):
    values = vars(args)
    xml_files = values['xml_lists']
    output_xml_path = values['output_xml_path']
    
    tree = ET.parse(xml_files[0])
    root = tree.getroot()
    Block = root.find("Block")
    Photogroups = Block.find("Photogroups")
    for xml_file in xml_files[1:]:
        tree_ = ET.parse(xml_file)
        root_ = tree_.getroot()
        Photogroup_ = root_.find("Block").find("Photogroups").find("Photogroup")
        # 将Photogroup添加到现有的Photogroups中
        Photogroups.append(Photogroup_)
    # 保存合并后的XML文件
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

parser = argparse.ArgumentParser(description='主函数')
subparsers = parser.add_subparsers(help='sub-command help')
parser_getImageXYZ = subparsers.add_parser('getImageXYZ', help='获取图像的XYZ坐标')
parser_getImageXYZ.add_argument('--colmap_dir', type=str, help='colmap文件夹路径', required=True)
parser_getImageXYZ.add_argument('--output_path', type=str, help='输出文件路径', required=True)
parser_getImageXYZ.set_defaults(func=getImageXYZFromColmapTxt)

parser_from_photoscan_xml_2_colmap_txt = subparsers.add_parser('from_photoscan_xml_2_colmap_txt', help='从photoscan xml转换为colmap txt')
parser_from_photoscan_xml_2_colmap_txt.add_argument('--xml_path', type=str, help='输入xml文件路径', required=True)
parser_from_photoscan_xml_2_colmap_txt.add_argument('--output_path', type=str, help='输出colmap txt文件路径', required=True)
parser_from_photoscan_xml_2_colmap_txt.add_argument('--target_image_path', type=str, help='输出图像路径(是否需要图像),默认不输出', required=False, default=None)
parser_from_photoscan_xml_2_colmap_txt.set_defaults(func=from_photoscan_xml_2_colmap_txt)

parser_rotate_image_xyz = subparsers.add_parser('rotate_image_xyz', help='旋转图像XYZ坐标')
parser_rotate_image_xyz.add_argument('--image_xyz', type=str, help='输入图像XYZ坐标文件路径', required=True)
parser_rotate_image_xyz.add_argument('--output_path', type=str, help='输出图像XYZ坐标文件路径', required=True)
parser_rotate_image_xyz.add_argument('--matrix_path', type=str, help='旋转矩阵文件路径', required=True)
parser_rotate_image_xyz.set_defaults(func=rotate_image_xyz)

parser_write_colmap_txt_2_photoscan_xml = subparsers.add_parser('write_colmap_txt_2_photoscan_xml', help='将colmap txt转换为photoscan xml')
parser_write_colmap_txt_2_photoscan_xml.add_argument('--colmap_txt_path', type=str, help='输入colmap txt文件路径', required=True)
parser_write_colmap_txt_2_photoscan_xml.add_argument('--output_xml_path', type=str, help='输出photoscan xml文件路径', required=True)
parser_write_colmap_txt_2_photoscan_xml.add_argument('--image_path', type=str, help='图像路径', required=True)
parser_write_colmap_txt_2_photoscan_xml.set_defaults(func=write_colmap_txt_2_photoscan_xml)

parser_merge_photoscan_xml = subparsers.add_parser('merge_photoscan_xml', help='合并多个photoscan xml文件')
parser_merge_photoscan_xml.add_argument('--xml_lists', type=str, nargs='+', help='输入xml文件列表', required=True)
parser_merge_photoscan_xml.add_argument('--output_xml_path', type=str, help='输出合并后的xml文件路径', required=True)
parser_merge_photoscan_xml.set_defaults(func=merge_photoscan_xml)

# python main.py getImageXYZ --colmap_dir J:/jyg/test/20250726162036/shading/Colmap/sparse/0 --output_path J:/jyg/test/image_xyz.txt
# python main.py from_photoscan_xml_2_colmap_txt --xml_path J:/jyg/test/camera_shouchi.xml --output_path J:/jyg/test/colmap
# python main.py rotate_image_xyz --image_xyz J:/jyg/test/image_xyz.txt --output_path J:/jyg/test/image_xyz_out.txt --matrix_path J:/jyg/test/matrix.txt
# python main.py write_colmap_txt_2_photoscan_xml --colmap_txt_path J:/jyg/test/colmap_rotate --output_xml_path J:/jyg/test/camera_shouchi_rotate.xml --image_path J:/jyg/test/20250726162036/image_undistorted/camera_front
# python main.py merge_photoscan_xml --xml_lists J:/jyg/test/camera_wrj.xml J:/jyg/test/camera_shouchi_rotate.xml --output_xml_path J:/jyg/test/merged_camera.xml
if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)