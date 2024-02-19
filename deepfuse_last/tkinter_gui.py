import tkinter as tk
import new_fusion_gui_version
from PIL import Image, ImageTk
import cv2


eo, fus_y, y, ir, fusion, new_gray, new_fusion, masked_rgb_img = new_fusion_gui_version.image_process()
# Tkinter 창 생성
#eo = cv2.cvtColor(eo, cv2.COLOR_BGR2RGB)

# Pillow 이미지 객체로 변환
image = Image.fromarray(new_fusion)

# Tkinter 창 생성
root = tk.Tk()
root.title("이미지 출력 예제")

# Tkinter에서 사용할 수 있는 PhotoImage 객체로 변환
photo = ImageTk.PhotoImage(image=image)

# 이미지를 Tkinter Label 위젯에 표시
label = tk.Label(root, image=photo)
label.pack()

# Tkinter 이벤트 루프 시작
root.mainloop()