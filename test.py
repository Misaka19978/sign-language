import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
#import pyttsx3

# 初始化文本到语音引擎
#engine = pyttsx3.init()

# 加载训练好的全连接神经网络模型
model = tf.keras.models.load_model('deep_learning_model.h5')

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 如果摄像头没打开，尝试 0 或 1
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# 定义标签字典，将预测结果映射为字符
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'ZH', 27: 'CH', 28: 'SH', 29: 'NG'
}

# 用于存储上一次预测的字符，避免重复播放
last_predicted_character = None
def extract_hand_landmarks(hand_landmarks, frame_shape):
    """提取手部关键点并归一化"""
    data_aux = []
    x_ = []
    y_ = []

    for landmark in hand_landmarks.landmark:
        x = landmark.x
        y = landmark.y
        x_.append(x)
        y_.append(y)

    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))

    return np.array(data_aux).flatten(), x_, y_

def draw_prediction(frame, predicted_character, x_, y_, frame_shape):
    """在帧上绘制预测结果"""
    H, W = frame_shape[:2]
    x1 = int(min(x_) * W) - 10
    y1 = int(min(y_) * H) - 10
    x2 = int(max(x_) * W) - 10
    y2 = int(max(y_) * H) - 10

    # 绘制绿色边界框和字符
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

def display_status(frame, text):
    """在帧上显示状态信息"""
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size

    # 创建白色背景矩形
    cv2.rectangle(frame, (10, 10), (20 + text_w, 40 + text_h), (255, 255, 255), -1)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

while True:
    ret, frame = cap.read()

    if not ret:  # 如果帧捕获失败，跳过当前循环
        print("Failed to capture frame")
        continue

    # 将帧从 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 检测手部关键点
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制关键点和连接线
            mp_drawing.draw_landmarks(
                frame,  # 图像
                hand_landmarks,  # 检测到的手部关键点
                mp_hands.HAND_CONNECTIONS,  # 手部连接
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # 提取关键点并归一化
            data_aux, x_, y_ = extract_hand_landmarks(hand_landmarks, frame.shape)

            # 使用模型进行预测
            prediction = model.predict(np.expand_dims(data_aux, axis=0))
            predicted_class = np.argmax(prediction)
            predicted_character = labels_dict[predicted_class]

            # 绘制预测结果
            draw_prediction(frame, predicted_character, x_, y_, frame.shape)

            # 如果预测的字符发生变化，播放语音
            # if predicted_character != last_predicted_character:
            #     engine.say(predicted_character)
            #     engine.runAndWait()
            #     last_predicted_character = predicted_character

        # 显示识别结果
        display_status(frame, f"Predicted: {predicted_character}")
    else:
        # 显示未检测到手的状态
        display_status(frame, "No hand detected")

    # 显示处理后的帧
    cv2.imshow('frame', frame)

    # 如果按下 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()