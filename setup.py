from setuptools import find_packages, setup

import re


def get_version():
    """Get package version from app_info.py file"""
    filename = "anylearning/app_info.py"
    with open(filename, encoding="utf-8") as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M)
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __version__")
    version = match.groups()[0]
    return version


def get_install_requires():
    """Get python requirements based on context.

    Two tiers, deliberately:

    * The application layer uses compatible-version ranges. These libraries have
      no coupling to the training stack, so pinning them to an exact patch only
      served to freeze known CVEs in place.
    * The ML runtime stays exact-pinned. ``torch``/``torchvision``/``onnxruntime``
      are ABI-sensitive, and detectron2 is compiled against whichever torch is
      installed, so a bump here means rebuilding it. Moving them is a staged piece
      of work tracked in ``docs/dependency_upgrade.md`` -- do not bump them here in
      isolation.
    """
    install_requires = [
        # --- Application layer ------------------------------------------
        # Lower bound = the version this is tested against; upper bound = the
        # next major, so patches and minors flow in without a manifest edit.
        "fastapi>=0.141,<1.0",
        "uvicorn>=0.52,<1.0",
        "python-multipart>=0.0.32,<1.0",
        "pydantic>=2.13,<3",
        "SQLAlchemy>=2.0.52,<3",
        "alembic>=1.19,<2",
        "requests>=2.34,<3",
        "urllib3>=2.7,<3",
        "pywebview>=6.2,<7",
        "loguru>=0.7.3,<1",
        "tqdm>=4.70,<5",
        # Direct imports that were previously only satisfied transitively.
        "cryptography>=50,<60",
        "psutil>=7.2,<8",
        "PyYAML>=6.0.3,<7",
        "numpy>=2,<3",
        "Pillow>=12,<13",
        # mediapipe depends on opencv-contrib-python, so both variants can end up
        # installed; keep the floor low enough that they can agree on a build.
        "opencv-python-headless>=4.11,<6",
        # Used by the vendored NanoDet COCO evaluator.
        "tabulate>=0.10,<1",
        # --- Structured data -------------------------------------------
        # Kept in the main runtime because CSV/XLSX/Parquet projects have to
        # remain usable offline after they are created.  Every import in the
        # structured-data path is lazy, so image-only launches do not pay the
        # import-time cost.
        "pandas>=2.3,<4",
        "openpyxl>=3.1.5,<4",
        "xlrd>=2.0.2,<3",
        "pyarrow>=18,<24",
        # Vectorized, out-of-core scans keep table paging, filtering and export
        # bounded even when the canonical Parquet dataset is much larger than
        # available RAM. DuckDB is embedded and does not start a server.
        "duckdb>=1.4,<2",
        # CatBoost is the accuracy-oriented default for mixed numerical and
        # categorical business tables.  scikit-learn supplies the transparent
        # baseline, text classifier and sparse lexical-search baseline.
        "catboost>=1.2.8,<2",
        "scikit-learn>=1.8,<2",
        # --- ML runtime (pinned, see docstring) -------------------------
        "onnxruntime>=1.28,<2",
        "torch==2.13.0",
        "torchvision==0.26.0",
        # Required by torch.onnx.export: from torch 2.6 the default exporter is
        # the dynamo one, which imports onnxscript at module load. Without it
        # every trainer's ONNX export fails, and export is the last step of
        # every training job -- not an optional extra.
        "onnxscript>=0.7,<1",
        # --- RF-DETR ----------------------------------------------------
        # Apache 2.0, code and weights alike, for the Nano/Small/Medium/Large
        # tier -- see MODEL_LICENCES.md. Pure Python: no CUDA extension and no
        # compiled wheel, which is why it needs none of the treatment
        # detectron2 does.
        #
        # Deliberately NOT `rfdetr[train,onnx]`. The `train` extra pulls
        # `roboflow` and `rf100vl`, two API clients for downloading datasets
        # from Roboflow's platform -- nothing in the rfdetr package imports
        # either, and an offline product has no business shipping them. So the
        # pieces the training and export paths genuinely import are listed
        # here, one line each.
        "rfdetr>=1.9.2,<2",
        # rfdetr's own floor. It defines the DINOv2 windowed-attention backbone
        # against the transformers v5 API; v4 cannot load it at all.
        "transformers>=5.1,<6",
        # rfdetr[train]: the Lightning stack it trains through, its COCO
        # evaluator, and the torchmetrics mAP it logs. Lightning was previously
        # only ever installed as a NanoDet dependency -- which was true and not
        # something to keep depending on, now that a second trainer needs it and
        # needs a version range of its own. The two excluded patches are
        # rfdetr's own exclusions, not ours.
        "pytorch-lightning>=2.6,<3,!=2.6.2,!=2.6.3",
        "faster-coco-eval>=1.7.2,<2",
        "torchmetrics>=1.2,<2",
        "pycocotools>=2.0.8,<3",
        "scipy>=1.11,<2",
        # rfdetr[onnx], minus polygraphy: the exporter's constant folding falls
        # back cleanly when it is absent, and a verified export proved it.
        # onnxsim >= 0.7 is the floor rfdetr pins for a reason -- 0.5 has no
        # wheel for 3.11+ and pip silently builds onnxruntime from source.
        "onnx>=1.16,<2",
        "onnxsim>=0.7,<1",
        "onnx_graphsurgeon>=0.5,<1",
    ]

    return install_requires


def get_long_description():
    """Read long description from README"""
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name="anylearning",
    version=get_version(),
    packages=find_packages(),
    description="All-in-one toolkit for training AI models yourself",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="AnyLearning Authors",
    author_email="anylearning@nrl.ai",
    url="https://github.com/nrl-ai/anylearning-oss",
    install_requires=get_install_requires(),
    license="Apache-2.0",
    keywords="AI, toolkit, training, CV",
    classifiers=[
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # 3.11 is the floor: onnxruntime 1.28 requires it, and matplotlib 3.11
    # (pulled in by the vendored NanoDet) does too. 3.13 is what CI and the
    # packaged build use.
    python_requires=">=3.11",
    package_data={
        "anylearning": [
            "anylearning/frontend-dist/**/*",
            "anylearning/frontend-dist/*",
            "anylearning/training/configs/**/*",
            "anylearning/training/configs/*",
            "anylearning/configs/*",
            "anylearning/configs/*/*",
            "anylearning/models/*",
            "anylearning/models/*/*",
        ]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "anylearning=anylearning.app:main",
            "anylearning.ingest=anylearning.ingest:main",
        ],
    },
)
