import os

def resolve_folder_path(path: str) -> str:
    """
    지정된 폴더에 이미지 파일이 없고,
    하위 폴더가 딱 하나라면 그 하위 폴더 경로를 반환합니다.
    """
    # 이미지 파일 검사 (소문자 확장자 기준)
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        # 하위 폴더 탐색
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) == 1:
            new_path = os.path.join(path, subdirs[0])
            print(f"[INFO] No images found in {path}. Using single subfolder {new_path} instead.")
            return new_path
    return path