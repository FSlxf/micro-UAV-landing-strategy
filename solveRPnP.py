import cv2.aruco as aruco
import cv2
import numpy as np
from R_PnP import RPnP

winSize = (5, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 存储棋盘格角点的3D坐标
obj_points = []
aruco_size1 = 525
aruco_size2 = 75
# aruco_size = 45
# obj_points = np.array([[0, 0, 0], [aruco_size, 0, 0], [aruco_size, aruco_size, 0], [0, aruco_size, 0]], dtype=np.float32)
obj_points1 = np.array([[-aruco_size1 // 2, -aruco_size1 // 2, 0], [aruco_size1 // 2, -aruco_size1 // 2, 0], [aruco_size1 // 2, aruco_size1 // 2, 0], [-aruco_size1 // 2, aruco_size1 // 2, 0]], dtype=np.float32)
obj_points2 = np.array([[-aruco_size2 // 2, -aruco_size2 // 2, 0], [aruco_size2 // 2, -aruco_size2 // 2, 0], [aruco_size2 // 2, aruco_size2 // 2, 0], [-aruco_size2 // 2, aruco_size2 // 2, 0]], dtype=np.float32)
# obj_points = np.array([[-aruco_size // 2, -aruco_size // 2, 0], [aruco_size // 2, -aruco_size // 2, 0], [aruco_size // 2, aruco_size // 2, 0], [-aruco_size // 2, aruco_size // 2, 0]], dtype=np.float32)
f = 917
K = np.array([[917.81827939, 0.00000000e+00, 476.59532407],
              [0.00000, 917.18708274, 355.80783597],
              [0.00000, 0.00000, 1.00000]])
D = np.array([3.27275201e-03,-3.06212395e-01,-5.33889749e-05,-1.44196789e-03,1.26527880e+00])

def img_points_indice(img_points):
    # 提取第二行元素的排序索引
    sort_indices = np.argsort(img_points[1])
    # 按排序索引重新排列每一列
    img_points1 = img_points[:, sort_indices]

    # 提取前两列的数据
    first_two_columns = img_points1[:, :2]
    # 提取前两列元素的排序索引
    sort_indices = np.argsort(first_two_columns[0, :2])
    # 按排序索引重新排列前两列
    sorted_first_two_columns = first_two_columns[:, sort_indices]
    # 将重新排列的前两列与原数组的后两列组合起来
    img_points2 = np.hstack((sorted_first_two_columns, img_points1[:, 2:]))

    # 提取后两列的数据
    last_two_columns = img_points2[:, 2:]
    # 提取第一行元素的排序索引（按从大到小排序）
    sort_indices = np.argsort(-last_two_columns[0, :2])
    # 按排序索引重新排列后两列
    sorted_last_two_columns = last_two_columns[:, sort_indices]
    # 将重新排列后的后两列与原数组的前两列组合起来
    img_points3 = np.hstack((img_points2[:, :2], sorted_last_two_columns))
    return img_points3


def solvepnp(gray,corners,ids):
    # if (ids == [[18]]).all():
    #     obj_points = obj_points1
    # elif (ids == [[35]]).all():
    #     obj_points = obj_points2
    # elif (ids == [[35], [18]]).all():
    #     obj_points = obj_points2
    if ids is not None:
        for i in range(len(ids)):
            # 亚像素优化
            corners_subpix = corners[i][0].astype(np.float32)
            corners_subpix = cv2.cornerSubPix(gray, corners_subpix, winSize, (-1, -1), criteria)
            # print(obj_points)
            # print(f"ArUco {ids[i][0]} 外角点坐标:\n{corners_subpix}")

            # 解析外角点坐标
            img_points = corners_subpix
            array1 = img_points.T[0] - 476.59532407
            array2 = img_points.T[1] - 355.80783597
            img_points = np.vstack((array1, array2))
            # print(img_points)
            # print(img_points)
            # 解算 R 和 T
            # 传统方法1
            # success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
            # 传统方法2
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, K, D)
            # RPnP
            if len(ids) > 1:
                R, tvec = RPnP(obj_points1.T, img_points / f)
            elif ids == 18:
                R, tvec = RPnP(obj_points2.T, img_points / f)
            elif ids == 35:
                R, tvec = RPnP(obj_points1.T, img_points / f)

            rvec = cv2.Rodrigues(R)[0]

            # 绕x轴逆时针旋转10度的旋转矩阵
            theta_x = np.radians(-10)  # 将角度转换为弧度
            R_x_rotation = np.array([[1, 0, 0],
                                     [0, np.cos(theta_x), -np.sin(theta_x)],
                                     [0, np.sin(theta_x), np.cos(theta_x)]])
            # 计算旋转后的相机相对靶标的旋转矩阵
            R2 = np.dot(R_x_rotation, R)
            rvec2 = cv2.Rodrigues(R)[0]
            # 计算旋转后的相机相对靶标的平移向量
            tvec2 = np.dot(R_x_rotation, tvec)

            # 绕z轴逆时针旋转90度的旋转矩阵
            theta_z = np.radians(90)   # 将角度转换为弧度
            R_z_rotation = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                     [np.sin(theta_z), np.cos(theta_z), 0],
                                     [0, 0, 1]])
            # 计算旋转后的相机相对靶标的旋转矩阵
            R2 = np.dot(R_z_rotation, R2)
            rvec2 = cv2.Rodrigues(R2)[0]
            # 计算旋转后的相机相对靶标的平移向量
            tvec2 = np.dot(R_z_rotation, tvec2)

            # # 绕y轴顺时针旋转90度的旋转矩阵
            theta_y = np.radians(-90)  # 将角度转换为弧度
            R_y_rotation = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                     [0, 1, 0],
                                     [-np.sin(theta_y), 0, np.cos(theta_y)]])
            # 计算旋转后的相机相对靶标的旋转矩阵
            R3 = np.dot(R_y_rotation, R2)
            rvec3 = cv2.Rodrigues(R3)[0]
            # 计算旋转后的相机相对靶标的平移向量
            tvec3 = np.dot(R_y_rotation, tvec2)

            # 打印旋转矩阵和平移矩阵
            # print(f"111111ArUco {ids[i][0]} 相对于相机的旋转矩阵 R:\n{cv2.Rodrigues(rvec3)[0]}")
            # print(f"111111ArUco {ids[i][0]} 相对于相机的平移矩阵 T:\n{tvec3}")
            cv2.drawFrameAxes(gray, K, D, rvec, tvec, 30)

    if ids is None:
        # 解析外角点坐标
        img_points = corners

        array1 = img_points.T[0] - 476.59532407
        array2 = img_points.T[1] - 355.80783597
        img_points = np.vstack((array1, array2))
        # # 将数组排序
        img_points = img_points_indice(img_points)
        R, tvec = RPnP(obj_points1.T, img_points / f)
        rvec = cv2.Rodrigues(R)[0] # 由于没有方向性，输出的旋转方向不唯一

        # 绕x轴逆时针旋转10度的旋转矩阵
        theta_x = np.radians(-10)  # 将角度转换为弧度
        R_x_rotation = np.array([[1, 0, 0],
                                 [0, np.cos(theta_x), -np.sin(theta_x)],
                                 [0, np.sin(theta_x), np.cos(theta_x)]])
        # 计算旋转后的相机相对靶标的旋转矩阵
        R2 = np.dot(R_x_rotation, R)
        rvec2 = cv2.Rodrigues(R)[0]
        # 计算旋转后的相机相对靶标的平移向量
        tvec2 = np.dot(R_x_rotation, tvec)

        # 绕z轴逆时针旋转90度的旋转矩阵
        theta_z = np.radians(90)  # 将角度转换为弧度
        R_z_rotation = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                 [np.sin(theta_z), np.cos(theta_z), 0],
                                 [0, 0, 1]])
        # 计算旋转后的相机相对靶标的旋转矩阵
        R2 = np.dot(R_z_rotation, R2)
        rvec2 = cv2.Rodrigues(R2)[0]
        # 计算旋转后的相机相对靶标的平移向量
        tvec2 = np.dot(R_z_rotation, tvec2)

        # # 绕y轴顺时针旋转90度的旋转矩阵
        theta_y = np.radians(90)  # 将角度转换为弧度
        R_y_rotation = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                                 [0, 1, 0],
                                 [np.sin(theta_y), 0, np.cos(theta_y)]])
        # 计算旋转后的相机相对靶标的旋转矩阵
        R3 = np.dot(R_y_rotation, R2)
        rvec3 = cv2.Rodrigues(R3)[0]
        # 计算旋转后的相机相对靶标的平移向量
        tvec3 = np.dot(R_y_rotation, tvec2)

        # 打印旋转矩阵和平移矩阵
        # print(f"ArUco 相对于相机的旋转矩阵 R:\n{cv2.Rodrigues(rvec3)[0]}")
        # print(f"ArUco 相对于相机的平移矩阵 T:\n{tvec3}")
        # cv2.drawFrameAxes(gray, K, D, rvec, tvec, 30)

    # aruco.drawDetectedMarkers(gray, corners, ids)  # 绘制边框
    # image_rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    return gray, rvec, tvec, rvec3, tvec3

