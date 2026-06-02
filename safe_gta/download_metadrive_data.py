import os
import urllib.request
import tarfile

def download_dataset_directly():
    # 這是 DSRL 官方在清華大學伺服器上的公開下載連結
    url = "https://cloud.tsinghua.edu.cn/f/8a2e5d1a58d44e5d8b2d/?dl=1"
    
    # 我們先下載壓縮包
    download_dir = "data"
    os.makedirs(download_dir, exist_ok=True)
    tar_path = os.path.join(download_dir, "OfflineMetadrive-intersection-v0.tar.gz")
    
    print(f"🚀 正在直接從伺服器下載 OfflineMetadrive-intersection-v0 資料集...")
    print(f"🔗 來源網址: {url}")
    print("⏳ 這個檔案有點大，請耐心等候下載完成...")
    
    # 執行下載
    try:
        urllib.request.urlretrieve(url, tar_path)
        print("✅ 下載成功！正在解壓縮...")
        
        # 解壓縮 tar.gz 檔案
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
            
        print(f"🎉 解壓縮完成！資料夾內容: {os.listdir(download_dir)}")
        print("👉 現在你可以去跑 `prepare_real_data.py` 進行分流了！")
        
    except Exception as e:
        print(f"❌ 下載或解壓縮過程發生錯誤: {e}")

if __name__ == "__main__":
    download_dataset_directly()