# 初始化变量x
import cv2

bias_x = 0.0
integral_bias_x = 0.0
last_bias_x = 0.0
pwm_out_x = 0.0
# 初始化变量y
bias_y = 0.0
integral_bias_y = 0.0
last_bias_y = 0.0
pwm_out_y = 0.0
# 初始化变量z
bias_z = 0.0
integral_bias_z = 0.0
last_bias_z = 0.0
pwm_out_z = 0.0
tag_bias_z = 0

# 设置PID参数
Position_Kp = 0.1
Position_Ki = 0.001
Position_KD = 0.1

def position_pid_x(encoder, target, i):
    global bias_x, pwm_out_x, pwm_out_xx, integral_bias_x, last_bias_x, tag_bias_x
    bias_x = encoder - target  # 计算偏差
    # print(bias_x)
    integral_bias_x += bias_x  # 计算偏差的积分
    # 位置式PID控制器公式
    pwm_out_xx = Position_Kp * bias_x + Position_Ki * integral_bias_x + Position_KD * (bias_x - last_bias_x)
    if i == 1 or i // 10 == 0:
        tag_bias_x = bias_x
    if bias_x < tag_bias_x - 100:
        integral_bias_x = 0
    last_bias_x = bias_x  # 保存上一次偏差

    if pwm_out_xx >= 350:
        pwm_out_x = 30
    if pwm_out_xx < 350:
        pwm_out_x = int(pwm_out_xx * 0.07)
    if bias_x is not None:
        if bias_x < 1100:
            pwm_out_x = 0

    # print("xxx轴PWM输出:", pwm_out_xx)
    return pwm_out_x  # 输出

def position_pid_y(encoder, target, i):
    global bias_y, pwm_out_y, integral_bias_y, last_bias_y, tag_bias_y, pwm_out_x
    bias_y = encoder - target  # 计算偏差
    # print('bias_y:', bias_y)
    integral_bias_y += bias_y  # 计算偏差的积分
    # 位置式PID控制器公式
    pwm_out_yy = Position_Kp * bias_y + Position_KD * (bias_y - last_bias_y)
    # pwm_out_yy = Position_Kp * bias_y + Position_Ki * integral_bias_y + Position_KD * (bias_y - last_bias_y)
    if i == 1 or i // 10 == 0:
        tag_bias_y = bias_y
    if bias_y < tag_bias_y - 150 or bias_y > tag_bias_y + 150:
        integral_bias_y = 0

    if pwm_out_yy >= 350:
        pwm_out_y = 15
    if pwm_out_yy < 350:
        pwm_out_y = int(pwm_out_yy * 0.5)

    if pwm_out_y <= -4:
        pwm_out_x = 0
        pwm_out_y = pwm_out_y - 5
    if pwm_out_y >= 4:
        pwm_out_x = 0
        pwm_out_y = pwm_out_y + 5
    if 4 > pwm_out_y > -4:
        pwm_out_y = 0
    last_bias_y = bias_y  # 保存上一次偏差
    # print("yyy轴PWM输出:", pwm_out_yy)
    return pwm_out_y, pwm_out_x  # 输出

def position_pid_z(encoder, target, i):
    global bias_z, pwm_out_z, pwm_out_zz, integral_bias_z, last_bias_z, tag_bias_z
    bias_z = encoder - target  # 计算偏差
    integral_bias_z += bias_z  # 计算偏差的积分
    # 位置式PID控制器公式
    pwm_out_zz = Position_Kp * bias_z + Position_KD * (bias_z - last_bias_z)
    if i == 1 or i // 10 == 0:
        tag_bias_z = bias_z
    if bias_z < tag_bias_z - 150:
        integral_bias_z = 0
    last_bias_z = bias_z  # 保存上一次偏差

    pwm_out_z = -int(pwm_out_zz * 0.35)

    # print("zzz轴PWM输出:", pwm_out_zz)
    return pwm_out_z  # 输出


# # 测试代码
# encoder_value = 5000
# target_position = 0
# # print("PWM输出:", position_pid(encoder_value, target_position))
# for i in range(50):
#     pwm_out = position_pid(encoder_value, target_position)
#     target_position += pwm_out
#     print(target_position)
