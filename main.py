
import cv2
import mediapipe as mp
import time
import math

# 取得攝影機 get your camera
v_camera = cv2.VideoCapture(0) # 0: 預設攝影機 0: Default camera
v_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
v_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 手勢判斷 Gesture judgment
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def calculate_angle(point1, point2, point3):
    vector1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    vector2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
    
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v**2 for v in vector1))
    magnitude2 = math.sqrt(sum(v**2 for v in vector2))
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
    return angle

def is_thumb_open(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    angle = calculate_angle(thumb_tip, thumb_ip, thumb_mcp)
    return angle > 150  # 調整閾值以適應需求 Adjust the threshold to suit your needs

def is_finger_open(hand_landmarks, tip_id, pip_id, mcp_id, angle_threshold=160):
    angle = calculate_angle(hand_landmarks.landmark[tip_id], 
                            hand_landmarks.landmark[pip_id], 
                            hand_landmarks.landmark[mcp_id])
    return angle > angle_threshold

def determine_finger_state(hand_landmarks):
    return [
        is_thumb_open(hand_landmarks),
        is_finger_open(hand_landmarks, 8, 6, 5),  # 食指
        is_finger_open(hand_landmarks, 12, 10, 9),  # 中指
        is_finger_open(hand_landmarks, 16, 14, 13),  # 無名指
        is_finger_open(hand_landmarks, 20, 18, 17)  # 小指
    ]

def main():
    # 取得數字圖片 get number images
    numberImageList = []
    for index in range(7):
        fileName = f'images/num{index}.jpg'
        numberImageList.append(cv2.resize(cv2.imread(fileName), dsize=(150,150)))

    # 初始化新功能的變量 Initialize variables required for the features
    isRun = False
    imgGirl = cv2.resize(cv2.imread("images/table.jpg"), dsize=(800, 800))

    # 取得目前時間 Get the current time
    last_time = time.time() 
    while True:
        # 讀取攝影機 Read the camera
        success, img = v_camera.read()
        if not success:
            print("Camera read failed 無法得到攝影機畫面")
            break

        # 變更圖片格式 BGR -> RGB Convert image format BGR -> RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 取得圖片寬高 Get image width and height
        height, width, _ = imgRGB.shape
        # 分析圖片手勢 Analyze image hand shape
        results = hands.process(imgRGB)

        counts = []
        total_fingers = 0
        if results.multi_hand_landmarks:
            startPoint1, startPoint2 = None, None
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

                fingers = determine_finger_state(hand_landmarks)
                count = sum(fingers)
                total_fingers += count

                # 特殊情况：數字6 Special case: number 6
                if fingers == [1, 0, 0, 0, 1]:
                    count = 6

                counts.append(count)

                # 在圖片上顯示手指狀態 Display finger status on the image
                x_offset = 10 if hand_idx == 0 else width // 2
                for i, is_open in enumerate(fingers):
                    color = (0, 255, 0) if is_open else (0, 0, 255)
                    cv2.putText(img, f"H{hand_idx+1}F{i+1}: {'Open' if is_open else 'Closed'}", 
                                (x_offset, 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 檢測食指位置  Detect index finger position
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if id == 8:  # 食指尖 finger tip
                        if startPoint1 is None:
                            startPoint1 = (cx, cy)
                        else:
                            startPoint2 = (cx, cy)

            # 處理兩個食指間的距離  Handle the distance between two index fingers
            if startPoint1 and startPoint2:
                length = math.hypot(startPoint1[0] - startPoint2[0], startPoint1[1] - startPoint2[1])
                print(f"Distance between fingers: {length}")

                if length < 60:
                    isRun = True
                
                if isRun:
                    imgStep = 20
                    imgGirlWidth = imgGirlHeight = int(min(abs(startPoint1[0] - startPoint2[0]), 720) - imgStep)

                    if imgGirlWidth > 20:
                        imgGirlLeft = int(min(startPoint1[0], startPoint2[0]) + imgStep / 2)
                        imgGirlTop = int(min(startPoint1[1], startPoint2[1]) - int(imgGirlHeight / 2) + imgStep / 2)
                        imgGirlClone = cv2.resize(imgGirl, dsize=(imgGirlWidth, imgGirlWidth))
                        img[imgGirlTop:imgGirlTop+imgGirlHeight, imgGirlLeft:imgGirlLeft+imgGirlWidth] = imgGirlClone
        else:
            isRun = False

        # 數字圖片 Number image
        for i, count in enumerate(counts):
            numImage = numberImageList[min(count, 7)]
            numHeight, numWidth, _ = numImage.shape
            if i == 0:
                img[150:150+numHeight, 20:20+numWidth] = numImage
            else:
                img[150:150+numHeight, width-numWidth-20:width-20] = numImage

        # 畫出 FPS Draw FPS
        fps_string = "{0}fps".format(int(1/(time.time()-last_time)))
        cv2.putText(img, fps_string, (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1., (85, 45, 255), 2)
        last_time = time.time()

        # 顯示目前是別的數字 Display the currently recognized number
        if len(counts) == 2:
            cv2.putText(img, f"Numbers: {counts[0]}, {counts[1]}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif len(counts) == 1:
            cv2.putText(img, f"Number: {counts[0]}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(img, "No hands detected", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 顯示總手指數 Display the total number of fingers
        cv2.putText(img, f"Total fingers: {total_fingers}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc退出
            break

    v_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
