import os

def get_db_config(filename):
  db_info = {}
  base_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(base_dir, filename)

  try:
    with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          key, value = line.split("=", 1)
          db_info[key.strip()] = value.strip()
    return db_info
  except FileNotFoundError:
    print(f"에러 : {file_path} 파일을 찾을 수 없습니다.")
    return None