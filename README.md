# AutoRetouch v7 — C++/Qt migration starter

This is the first native C++ implementation based on the supplied Python project. It provides a Windows desktop UI, local image open/preview/mask/save flow, face-mask fallback, and core retouch/adjustment controls.

## Current scope and important parity note

This is **not yet feature-equivalent** to Python AutoRetouch v7. The original depends on MediaPipe Face Detection/FaceMesh and Selfie Segmentation, while this initial C++ port uses OpenCV's Haar face detector and an approximate oval mask. The original Quick Selection/Lasso/History Brush, hand-painted mask undo, dodge/burn brush, spot-removal model, complete Camera Raw adjustments, EXIF/ICC preservation, and full-resolution tiled render still need to be ported. They are documented here rather than falsely claimed as complete.

## Build on Windows

Requirements:
- Visual Studio 2022 with **Desktop development with C++**
- CMake 3.21+
- Qt 6 (MSVC 64-bit kit)
- OpenCV 4.x built/installed for the same MSVC toolchain

Configure environment variables if CMake cannot find packages:

```bat
set CMAKE_PREFIX_PATH=C:\Qt\6.x.x\msvc2022_64
set OpenCV_DIR=C:\opencv\build\x64\vc17\lib
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The executable will be in `build\Release\AutoRetouchCpp.exe` (depending on generator layout).

For face detection, place `haarcascade_frontalface_default.xml` next to the EXE, in a `data` subfolder, or set `OPENCV_HAAR_CASCADE` to its full path. OpenCV's XML file is distributed with OpenCV's data files.

To package the Qt runtime after building, use `windeployqt path\to\AutoRetouchCpp.exe`, and make sure the OpenCV runtime DLLs and cascade XML are included beside the executable.

## Next migration stages

1. Port the Python mask/retouch engine routines to C++ and create numerical regression fixtures.
2. Replace Haar approximation with FaceMesh landmarks and person segmentation (native MediaPipe C++ or a compatible ONNX model).
3. Port all v7 tools, undo/history, zoom/pan, sliders, and exact view behavior.
4. Preserve EXIF/ICC on save and implement full-resolution strip rendering.
5. Build and test the Windows installer on a Windows machine.


## Build a Windows EXE with GitHub Actions

1. Extract this project and push the **contents** of the `AutoRetouchCpp` folder to the root of a GitHub repository (so `CMakeLists.txt`, `src/`, and `.github/` are at the repository root).
2. Ensure the default branch is `main` or `master`.
3. Open the repository's **Actions** tab and run **Build AutoRetouchCpp (Windows x64)** with **Run workflow**, or push a change to `src/`, `CMakeLists.txt`, or the workflow file.
4. Open the successful run and download the `AutoRetouchCpp-Windows-x64` artifact. Extract it, then keep all files in the extracted folder together and run `AutoRetouchCpp.exe`.

The workflow uses a Windows Server 2022 runner, Qt 6.8.3 for MSVC 2022 x64, and the runner's vcpkg installation to install OpenCV. It runs CMake, builds Release, runs `windeployqt`, copies vcpkg runtime DLLs, and downloads the Haar cascade XML used by this source. The artifact is a **folder artifact**, not a single self-contained EXE; Qt/OpenCV DLLs and the cascade XML must remain beside the executable.

### Workflow limitations / verification status

- The workflow file is configured for this repository layout (`CMakeLists.txt` at the repository root and `src/main.cpp`).
- This environment is not a Windows MSVC/Qt build runner, so the Windows build and runtime packaging have **not been executed or verified here**. The first GitHub Actions run is the definitive integration test; review its logs if dependency discovery or deployment fails.
- The workflow copies all DLLs from vcpkg's x64-windows `bin` directory to make OpenCV's runtime dependencies available. This is robust but can include extra DLLs and make the artifact larger.
- The current C++ program is a migration starter, not a feature-complete port of the original Python app. The face mask is an approximate OpenCV Haar/oval mask, not the original MediaPipe pipeline, and several Python tools are not yet implemented.
- The cascade is fetched from the OpenCV `4.x` branch at build time; that makes the workflow depend on GitHub network availability.
