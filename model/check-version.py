import importlib

packages = [
    "cv2",            # OpenCV
    "matplotlib",
    "numpy",
    "PIL",            # Pillow
    "torch",
    "torchvision",
    "albumentations",
    "yaml",           # PyYAML
    "tqdm",
    "seaborn",
    "sklearn",        # scikit-learn
    "ultralytics"
]

print("📦 Installed Package Versions:\n")

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        # Special case: Pillow's version is in PIL.__version__
        if pkg == "PIL":
            print(f"{pkg:<15}: {module.__version__}")
        elif pkg == "cv2":
            print(f"{pkg:<15}: {module.__version__}")
        elif pkg == "sklearn":
            import sklearn
            print(f"{pkg:<15}: {sklearn.__version__}")
        else:
            print(f"{pkg:<15}: {module.__version__}")
    except ImportError:
        print(f"{pkg:<15}: ❌ Not Installed")

