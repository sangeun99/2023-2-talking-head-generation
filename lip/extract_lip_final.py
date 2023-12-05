import cv2
import dlib
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

# Dlib의 얼굴 탐지기와 특징점 탐지기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lip/shape_predictor_68_face_landmarks.dat")  # 다운로드 필요

def extract_lip_to_png(ori_imgs_dir, lip_imgs_dir, except_lip_imgs_dir):
    print(f'[INFO] ===== extract lip images from {ori_imgs_dir} =====')

    # lip_imgs_dir, except_lip_imgs_dir 디렉토리가 없으면 생성
    os.makedirs(lip_imgs_dir, exist_ok=True)
    os.makedirs(except_lip_imgs_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm(image_paths):
        # 이미지 읽기
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert to 4 channels
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # 얼굴 탐지
        faces = detector(gray)

        for face in faces:
            # 특징점 탐지
            landmarks = predictor(image=gray, box=face)

            # 입술 영역 특징점 인덱스 (49 ~ 68)
            lip_indices = list(range(48, 68))

            # 입술 영역에 해당하는 특징점 좌표 찾기
            lip_points = []
            for i in lip_indices:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                lip_points.append((x, y))

            # 입술 영역의 직사각형 범위 찾기
            x_min = min([point[0] for point in lip_points])
            x_max = max([point[0] for point in lip_points])
            y_min = min([point[1] for point in lip_points])
            y_max = max([point[1] for point in lip_points])

            # padding to H == W
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            l = max(x_max - x_min, y_max - y_min) // 2
            x_min = max(0, cx - l)
            x_max = min(img.shape[1], cx + l)
            y_min = max(0, cy - l)
            y_max = min(img.shape[0], cy + l)

            # 입술 영역 직사각형으로 잘라내기
            # 입술 영역 직사각형으로 잘라내기
            lip_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            lip_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max].copy()

            # 입술 영역 제외한 부분 이미지 생성
            except_lip_img = img.copy()
            # cv2.fillPoly(except_lip_img, [np.array(lip_points)], (0, 0, 0, 0))
            cv2.fillPoly(except_lip_img, [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])], (0, 0, 0, 0))

            # 이미지 저장
            lip_output_path = os.path.join(lip_imgs_dir, os.path.basename(image_path).replace('.jpg', '.png'))
            cv2.imwrite(lip_output_path, lip_img)

            except_lip_output_path = os.path.join(except_lip_imgs_dir, os.path.basename(image_path).replace('.jpg', '.png'))
            cv2.imwrite(except_lip_output_path, except_lip_img)

    print(f'[INFO] ===== extracted lip images and except lip images =====')

# 사용 예시
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)

    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    lip_imgs_dir = os.path.join(base_dir, 'lip_imgs')
    except_lip_imgs_dir = os.path.join(base_dir, 'except_lip_imgs')

    extract_lip_to_png(ori_imgs_dir, lip_imgs_dir, except_lip_imgs_dir)