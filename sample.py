import os

# 대상 폴더 지정
directory = './dataset/testset/EO'  # 현재 디렉토리를 사용

# 파일 이름 변경 시작
for filename in os.listdir(directory):
    if filename.endswith(".png") and filename[:-4].isdigit():
        num = int(filename[:-4])  # 확장자를 제외한 숫자 부분
        if 1501 <= num <= 1606:
            new_filename = f"{num - 1501}.png"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

print("파일 이름 변경 완료.")
