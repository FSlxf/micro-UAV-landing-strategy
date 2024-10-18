import cv2
import math
import numpy as np

def find_ArucoContour(image):
    # 判断四边形的对角线长度是否接近
    def are_diagonals_close(diagonal1, diagonal2, threshold_ratio=30):
        length1 = math.sqrt(abs(diagonal1[0][0] ** 2 - diagonal1[0][1] ** 2))
        length2 = math.sqrt(abs(diagonal2[0][0] ** 2 - diagonal2[0][1] ** 2))
        # length1 = np.linalg.norm(diagonal1)
        # length2 = np.linalg.norm(diagonal2)
        return abs(length1 - length2) < threshold_ratio

    image_height, image_width = image.shape[:2]
    image_bbox = [(0, 0), (image_width, image_height)]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 形态学操作（去除噪点）
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


    # 查找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # # 创建空白图像作为文字区域
    # text_region = np.zeros_like(image)

    # # 根据轮廓绘制文字区域
    # for contour in contours:
    #     # x, y, w, h = cv2.boundingRect(contour)
    #     cv2.drawContours(text_region, [contour], -1, (255, 255, 255), cv2.FILLED)
    #
    # #提取文字区域
    # text_only = cv2.bitwise_and(image, text_region)


    # 筛选轮廓
    filtered_contours = []
    approxpoint = []
    for contour in contours:
        # 近似轮廓
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 获取轮廓的顶点数
        num_vertices = len(approx)
        if num_vertices == 4:
            # 检查四边形的对角线是否接近
            diagonal1 = approx[0] - approx[1]
            diagonal2 = approx[3] - approx[2]
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            contour_bbox = [(x, y), (x + w, y + h)]
            # 计算轮廓的面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # 计算面积和周长之比
            area_perimeter_ratio = area / perimeter
            # print(area_perimeter_ratio)
            # 如果面积和周长之比在某个范围内，则认为是矩形
            if 5 < area_perimeter_ratio < 50 and perimeter > 100 and perimeter < 1000 and (contour_bbox[0][0] > 0 and contour_bbox[0][1] > 0 and
        contour_bbox[1][0] < image_width and contour_bbox[1][1] < image_height) and are_diagonals_close(diagonal1, diagonal2):
                # print(area)
                approxpoint = approx
                filtered_contours.append(contour)

    # #绘制筛选后的轮廓
    # contour_image = cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 0), 2)

    return filtered_contours, approxpoint
