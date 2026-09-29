import kagglehub

# Download latest version
path = kagglehub.dataset_download("images/bsds300")

print("Path to dataset files:", path)