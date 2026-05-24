"""
check_gpu.py — 快速診斷 PyTorch CUDA 環境
在專案資料夾執行：  python check_gpu.py
"""
import sys

print(f"Python: {sys.version}\n")

try:
    import torch
    print(f"PyTorch 版本   : {torch.__version__}")
    print(f"CUDA 可用      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本      : {torch.version.cuda}")
        print(f"GPU 數量       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}  |  VRAM: {vram_gb:.1f} GB")
        print("\n✅ 可以用 GPU，build_index.py 的 batch=64 設定正確。")
    else:
        print("\n❌ CUDA 不可用！PyTorch 很可能是 CPU-only 版本。")
        print("   請重新安裝 CUDA 版 PyTorch：")
        print()
        print("   # 先移除舊版")
        print("   pip uninstall torch torchvision torchaudio -y")
        print()
        print("   # 安裝 CUDA 12.1 版（適合你的 RTX 3050）")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("   裝完後重跑此腳本確認，再重新執行 build_index.py。")
except ImportError:
    print("❌ torch 未安裝，請先執行：pip install -r requirements.txt")
