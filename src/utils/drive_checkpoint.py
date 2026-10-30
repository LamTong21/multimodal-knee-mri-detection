import io
import json
import os
import torch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

try:
    from google.colab import userdata
except ImportError:
    userdata = None


class DriveCheckpointManager:
    """Quản lý đọc/ghi và đồng bộ checkpoint lên Google Drive qua Service Account."""

    def __init__(self, secret_key="GDRIVE_SA_JSON", folder_secret_key="GDRIVE_FOLDER_ID_2"):
        sa_json_string = None
        self.folder_id = None

        if UserSecretsClient is not None:
            try:
                user_secrets = UserSecretsClient()
                sa_json_string = user_secrets.get_secret(secret_key)
                self.folder_id = user_secrets.get_secret(folder_secret_key)
            except Exception:
                pass

        if sa_json_string is None and userdata is not None:
            try:
                sa_json_string = userdata.get(secret_key)
                self.folder_id = userdata.get(folder_secret_key)
            except Exception:
                pass

        if not sa_json_string or not self.folder_id:
            raise ValueError("Không thể tải thông tin xác thực Google Drive từ Kaggle Secrets hoặc Colab Userdata.")

        sa_info = json.loads(sa_json_string)
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        self.drive_service = build("drive", "v3", credentials=creds)
        print("--> [Google Drive] Đã kết nối thành công qua Service Account.")

    def get_file_info(self, filename: str):
        query = f"'{self.folder_id}' in parents and name = '{filename}' and trashed = false"
        results = self.drive_service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name, size, modifiedTime)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get("files", [])
        return files[0] if files else None

    def update_checkpoint(self, checkpoint_data, filename: str, local_dir: str = "", keep_local: bool = True, keep_revision: bool = True):
        file_info = self.get_file_info(filename)
        if not file_info:
            raise FileNotFoundError(f"[Drive] File '{filename}' chưa tồn tại trong thư mục Drive chỉ định!")

        local_path = os.path.join(local_dir, filename) if local_dir else filename
        if isinstance(checkpoint_data, (dict, list, torch.Tensor)):
            torch.save(checkpoint_data, local_path)

        media = MediaFileUpload(local_path, resumable=True)
        updated_file = self.drive_service.files().update(
            fileId=file_info["id"],
            media_body=media,
            keepRevisionForever=keep_revision,
            fields="id, name, modifiedTime",
            supportsAllDrives=True
        ).execute()

        print(f"--> [Drive Update] Đã cập nhật '{filename}' lúc {updated_file.get('modifiedTime')}")
        if not keep_local and os.path.exists(local_path):
            os.remove(local_path)
        return updated_file

    def download_checkpoint(self, filename: str, save_path: str) -> bool:
        file_info = self.get_file_info(filename)
        if not file_info:
            print(f"--> [Drive Download] Không tìm thấy file '{filename}' trên Drive.")
            return False

        request = self.drive_service.files().get_media(fileId=file_info["id"])
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        with open(save_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        print(f"--> [Drive Download] Tải thành công: '{save_path}'")
        return True