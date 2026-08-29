from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="handpose",
        version="0.0.1",
        packages=find_packages(),
        install_requires=[
            # Not an exact pin: mediapipe 0.10.x only publishes cp39-cp312
            # wheels and declares numpy<2, so pinning it here made the whole
            # project unbuildable on Python 3.13 and blocked NumPy 2 everywhere.
            # 1.0.x ships a py3-none (ABI-agnostic) wheel with no numpy cap.
            #
            # 1.0.1 excluded: it aborts on macOS in any graph containing
            # TensorsToDetectionsCalculator, which is every detector including
            # the hand landmarker. See requirements.txt for the evidence.
            "mediapipe>=1.0.0,!=1.0.1,<2",
        ],
    )
