import distutils.core
import os

dist = distutils.core.run_setup("anylearning/training/models/detectron2/setup.py")
dependencies = " ".join([f"'{x}'" for x in dist.install_requires])
os.system(f"python -m pip install {dependencies}")
