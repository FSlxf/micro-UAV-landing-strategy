import numpy as np
import cv2
from djitellopy import Tello
from solveRPnP import solvepnp
from flipsolvepnp import solvepnpflip
from flipsolvepnp import img_flip
from pid import *
from REC_DETECT import find_ArucoContour
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
import KeyPressModule as kp
import time
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# # 标定板格点数量和大小
# pattern_size = (3, 3)  # 内部角点数量
# square_size = 55  # 棋盘格方块大小（毫米）
#
# # 存储棋盘格对应的图像点坐标
# img_points = []
# # 准备棋盘格的3D坐标
# objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
# objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
# # 提取每一列的值
# col1 = objp[:, 0]
# col2 = objp[:, 1]
# col3 = objp[:, 2]
# # 创建新的数组
# objp2 = np.array([col1, col2, col3])
# 读取 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
# 获取 ArUco 参数
parameters = aruco.DetectorParameters()
K = np.array([[917.81827939, 0.00000000e+00, 476.59532407],
              [0.00000, 917.18708274, 355.80783597],
              [0.00000, 0.00000, 1.00000]])
D = np.array([3.27275201e-03,-3.06212395e-01,-5.33889749e-05,-1.44196789e-03,1.26527880e+00])
rvec = None
tvec = None
tvec3 = None
i = 0
j = 0
x_pwm_out = 0
y_pwm_out = 0
z_pwm_out = 0
tvectotal = 0
totaltvec = 0
TT = 0
q = 0
tvec_record = []
time_record = []
start_time = 0
flag = True
command1 = False
command1 = False
TIME_FLAG = True
kp.init()
tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()

###########################plot_line########################
# 初始化图形和三维轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化线条数据（为空）
x_data, y_data, z_data = [], [], []
line, = ax.plot([], [], [], 'r-')  # 注意这里有三个空列表作为初始数据

# 设置三维轴的显示范围
ax.set_xlim(-4000, 4000)
ax.set_ylim(-4000, 4000)
ax.set_zlim(-4000, 4000)  # 假设z轴的范围也是0到100
###########################plot_line########################


###########################计算帧率########################
# 初始化变量
last_time = time.time()  # 记录上一次循环的时间
last_time2 = time.time()  # 记录上一次循环的时间
fps = 0  # 帧率计数器
sumfps = 0  # 帧率计数器
frame_count = 0  # 帧数计数器
# 设定测量时间长度，例如1秒
measurement_time = 1
###########################计算帧率########################

def getKeyboardInput():
    global command1, start_time
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 60

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("q"): tello.land(); time.sleep(3)
    if kp.getKey("e"): tello.takeoff()

    if kp.getKey("o"):
        command1 = True
        start_time = time.time()

    if kp.getKey("p"):
        command1 = False
    return [lr, fb, ud, yv, command1]

def precision_landing(tvec3,start_time3):
    moveforward = abs(int(tvec3[0] / 10))
    tello.move_forward(moveforward)
    # if tvec3[1] < 0:
    #     moveleft = abs(int(tvec3[1] / 10))0
    #     if 20 > moveleft > 10:
    #         tello.move_left(26)
    #     if moveleft > 20:
    #         tello.move_left(moveleft)
    # if tvec3[1] > 0:
    #     moveright = abs(int(tvec3[1] / 10))
    #     if 20 > moveright > 10:
    #         tello.move_right(26)
    #     if moveright > 20:
    #         tello.move_right(moveright)
    time.sleep(0.5)
    start_time4 = time.time()
    print('STEP3',start_time4 - start_time3)
    # tello.move_down(50)
    tello.land()
    start_time5 = time.time()
    print('STEP4',start_time5 - start_time4)
    return start_time4, start_time5
while True:
    img = tello.get_frame_read().frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img_flip(gray)
    height, width = gray.shape[:2]
    # 对比度增强
    # gray = cv2.equalizeHist(gray)
    # # img = cv2.resize(img, (360, 240))
    # -------------------------------检测 ArUco 码--------------------------------------
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(ids)
    if ids is None:
        # filtered_contours1, approxpoint1 = find_ArucoContour(gray) # 有反转的
        filtered_contours2, approxpoint2 = find_ArucoContour(img) # 没有反转的
        if len(approxpoint2) > 0: # 检测到角点
            # 检测到标记在下方
            if approxpoint2[0][0][1] > height // 2:
                gray, rvec2, tvec2, rvec3, tvec3 = solvepnp(gray, approxpoint2, ids)
                if start_time != 0:
                    tvec_record.append(tvec3 / 1000)
                    time_record.append(time.time() - start_time)

            # 检测到标记在上方
            if approxpoint2[0][0][1] < height // 2:
                gray, rvec2, tvec2, rvec3, tvec3 = solvepnpflip(gray, approxpoint2, ids)
                if start_time != 0:
                    tvec_record.append(tvec3 / 1000)
                    time_record.append(time.time() - start_time)

            # ###########################计算帧率########################
            # # 更新帧数计数器
            # frame_count += 1
            # # 获取当前时间
            # current_time = time.time()
            # if current_time - last_time >= measurement_time:
            #     # 计算帧率
            #     fps = frame_count / (current_time - last_time)
            #     sumfps = sumfps + fps
            #     avefps = sumfps / (current_time - last_time2)
            #     # 打印帧率
            #     # print(f"FPS: {fps:.2f}")
            #     print(f"FPS: {avefps:.2f}")
            #     # 重置计数器
            #     frame_count = 0
            #     # 更新上一次记录的时间
            #     last_time = current_time
            # ###########################计算帧率########################
    if ids is not None:
        # if flag:
        #     start_time = time.time()
        #     flag = False  # 在第一次执行后将标志设置为Fals
        # 检测到标记在下方
        if corners[0][0][0][1] > height // 2:
            gray, rvec, tvec, rvec3, tvec3 = solvepnp(gray,corners,ids)
            if start_time != 0:
                tvec_record.append(tvec3/1000)
                time_record.append(time.time() - start_time)

        # 检测到标记在上方
        if corners[0][0][0][1] < height // 2:
            gray, rvec, tvec, rvec3, tvec3 = solvepnpflip(gray,corners,ids)
            if start_time != 0:
                tvec_record.append(tvec3/1000)
                time_record.append(time.time() - start_time)

        ###########################计算帧率########################
        # # 更新帧数计数器
        # frame_count += 1
        # # 获取当前时间
        # current_time = time.time()
        # if current_time - last_time >= measurement_time:
        #     # 计算帧率
        #     fps = frame_count / (current_time - last_time)
        #     sumfps = sumfps + fps
        #     avefps = sumfps / (current_time - last_time2)
        #     # 打印帧率
        #     # print(f"FPS: {fps:.2f}")
        #     print(f"FPS: {avefps:.2f}")
        #     # 重置计数器
        #     frame_count = 0
        #     # 更新上一次记录的时间
        #     last_time = current_time
        # ids = None
        ###########################计算帧率########################
            ###########################plot_line########################
    # if command1 == True:
    #     if tvec3 is not None:
    #         Atvec = tvec3[0]
    #         totaltvec = totaltvec + Atvec
    #         TT = TT + 1
    #         avertvec = totaltvec / TT
    #         Btvec = Atvec - avertvec
    #         if -500 < Btvec < 500:
    #             print(tvec3)
    #             x_new = tvec3[0]
    #             y_new = tvec3[1]
    #             z_new = tvec3[2]
    #
    #             # 将新数据添加到列表中
    #             x_data.append(x_new)
    #             y_data.append(y_new)
    #             z_data.append(z_new)
    #
    #             # 更新线条的x、y、z数据
    #             line.set_data(x_data, y_data)
    #             line.set_3d_properties(z_data)
    #
    #             # 强制matplotlib更新图形
    #             plt.draw()
    #
    #             # 暂停一段时间（以毫秒为单位）
    #         plt.pause(0.01)
    ###########################plot_line########################

    # ------------------------------垂直镜像翻转回去--------------------------------------
    flipped_top = cv2.flip(gray[:height // 2, :], 0)
    gray[:height // 2, :] = flipped_top
    if rvec is not None:
        cv2.drawFrameAxes(gray, K, D, rvec, tvec, 30)
        rvec = None

    # ------------------------------pid控制x轴速度------------------------------------
        i = i + 1
        x_encoder_value = abs(tvec3[0])
        x_target_position = 0
        x_pwm_out = position_pid_x(x_encoder_value, x_target_position, i)
    # ------------------------------pid控制y轴速度---------------------------------
        y_encoder_value = 0
        y_target_position = y_encoder_value - tvec3[1]
        y_pwm_out, x_pwm_out = position_pid_y(y_encoder_value, y_target_position, i)

    # ----------------------------pid控制z轴速度---------------------------------
    #     z_encoder_value = abs(tvec3[2])
    #     z_target_position = 0
    #     z_pwm_out = position_pid_z(z_encoder_value, z_target_position, i)

    if filtered_contours2 is not []:
        gray = cv2.drawContours(gray.copy(), filtered_contours2, -1, (0, 255, 0), 2)
        filtered_contours2 = []
    # print(tvec3)
    vals = getKeyboardInput()
    if vals[4] == False:
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # vals[2] = -z_pwm_out
    if vals[4] == True:
        vals[1] = x_pwm_out
        vals[0] = y_pwm_out
        # tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        if j < 230 and vals[1] == 0 and -4 < vals[0] < 4:
            j = j + 1
        # if 60 < j < 120:
        #     tvectotal += tvec3
        if j >= 230:
            # ------------------------------pid控制z轴速度---------------------------------
            z_encoder_value = abs(tvec3[2])
            z_target_position = 0
            z_pwm_out = position_pid_z(z_encoder_value, z_target_position, i)
            # ------------------------------调整x、y轴---------------------------------476.5 177.5
            Xtarget_position = 476.5
            Ytarget_position = 177.5
            if ids is not None:
                cornercentral_x = (corners[0][0][0][0] + corners[0][0][2][0]) / 2
                cornercentral_y = (corners[0][0][0][1] + corners[0][0][2][1]) / 2

                if cornercentral_x > (Xtarget_position + 110): y_pwm_out = 13
                elif cornercentral_x < (Xtarget_position - 110): y_pwm_out = -13
                elif (Xtarget_position - 80) < cornercentral_x < (Xtarget_position + 80): y_pwm_out = 0

                if y_pwm_out == 0:
                    if cornercentral_y > (Ytarget_position + 70): x_pwm_out = -10
                    elif cornercentral_y < (Ytarget_position - 70): x_pwm_out = 10
                    elif (Ytarget_position - 50) < cornercentral_y < (Ytarget_position + 50): x_pwm_out = 0
                vals[0] = y_pwm_out
                vals[1] = x_pwm_out
                if x_pwm_out == 0:
                    if TIME_FLAG == True:
                        start_time2 = time.time()
                        print('STEP1',start_time2 - start_time)
                    TIME_FLAG = False
                    vals[2] = z_pwm_out
                # print(tvec3)
                if abs(tvec3[2]) < 700 and y_pwm_out == 0:
                    time.sleep(0.5)

                    print(len(time_record))
                    print(len(tvec_record))
                    # 将数据转换为DataFrame
                    df = pd.DataFrame(time_record)
                    qf = pd.DataFrame(tvec_record)
                    # 将DataFrame保存到Excel文件
                    df.to_excel("time_record.xlsx", index=False)
                    qf.to_excel("tvec_record.xlsx", index=False)
                    start_time3 = time.time()
                    print('STEP2',start_time3 - start_time2)
                    start_time4, start_time5 = precision_landing(tvec3, start_time3)
            # if ids is None:
            #     cornercentral_x = (approxpoint2[0][0][0] + approxpoint2[0][2][0]) / 2
            #     cornercentral_y = (approxpoint2[0][0][1] + approxpoint2[0][2][1]) / 2
            #
            #     if cornercentral_x > (Xtarget_position + 80): y_pwm_out = -5
            #     elif cornercentral_x < (Xtarget_position - 80): y_pwm_out = 5
            #     elif (Xtarget_position - 80) < cornercentral_x < (Xtarget_position + 80): y_pwm_out = 0
            #
            #     if cornercentral_y > (Ytarget_position + 50): x_pwm_out = -5
            #     elif cornercentral_y < (Ytarget_position - 50): x_pwm_ out = 5
            #     elif (Ytarget_position - 50) < cornercentral_y < (Ytarget_position + 50): x_pwm_out = 0
            #     vals[0] = y_pwm_out
            #     vals[1] = x_pwm_out
            #     if x_pwm_out == 0 and y_pwm_out == 0:
            #         vals[2] = z_pwm_out
            #     if tvec3[2] < 500:
            #         precision_landing(tvec3)

            # tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    cv2.imshow("Image", gray)
    cv2.waitKey(1)


