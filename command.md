python -m pip install --upgrade pip
python -m pip install cupy-cuda13x
python -m pip install numpy opencv-python pandas matplotlib scikit-learn joblib tqdm

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"

python -c "import cupy as cp; print('CuPy:', cp.__version__); print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'])"

python gpu_classification.py --data_dir dataset --output_dir results --max_train_samples 4000

python gpu_classification.py --data_dir dataset --output_dir results --max_train_samples 8000

python gpu_classification.py --data_dir dataset --output_dir results --max_train_samples 0

python gpu_classification.py --data_dir dataset --output_dir results --force_rebuild

nvidia-smi -l 2