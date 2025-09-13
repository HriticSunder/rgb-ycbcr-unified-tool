# RGB & YCbCr Multimedia Processing Tool  

A unified multimedia processing tool for **images and videos** across **RGB** and **YCbCr** color spaces.  
 

---

##  Features  

### 🔹 RGB Processing  
- Supports multiple formats: **RGB888, RGB565, RGB666, RGB101010, RGB121212, LOOSE666**  
- Frame-by-frame transformations  

### 🔹 YCbCr Processing  
- Subsampling: **4:4:4, 4:2:2, 4:2:0**  
- Bit depths: **24-bit, 20-bit, 16-bit, 12-bit**  
- Logs frames for later reconstruction  

### 🔹 Video Support  
- Handles both images and videos  
- Resizing, frame-by-frame logging  
- Multi-threaded execution for speed  
- Supports codecs: **MJPG, XVID, MP4V, AVC1**  

### 🔹 Reconstruction  
- `rebuild.py` rebuilds full videos from logged frames  
- Auto-detects FPS and resolution  
- Works for **both RGB and YCbCr pipelines**  

---

##  Usage  

### 1️⃣ Process an Image/Video  
```bash
python build.py
```
Follow prompts:  
1. Input file path (image or video)  
2. Width & height  
3. Color space (RGB / YCbCr)  
4. Format (RGB type or YCbCr subsampling/bit depth)  
5. Output folder is auto-created with timestamp  

---

### 2️⃣ Rebuild from Logs  
```bash
python rebuild.py
```
Follow prompts:  
1. Directory containing frame logs  
2. Output directory for reconstructed video  
3. Width & height  
4. FPS & format confirmation  
5. Reconstructed `.avi` video is saved  

---

## 📂 Repository Structure  

```
rgb-ycbcr-unified-tool/
 ├── build.py
 ├── rebuild.py
 ├── .gitignore
 ├── README.md
 └── sample_demo/
     ├── test.mp4
     ├── video_output20250913_183517/
     │   ├── frame_0000/
     │   ├── frame_0024/
     │   ├── frame_0048/
     │   ├── frame_0072/
     │   ├── frame_0096/
     │   └── output20250913_183517.avi
     └── rebuilded_video/
         └── reconstructed_20250913_183927.avi
```  

---

## 📂 Sample Demo  

A short demo is included in `sample_demo/` for quick review.  

### Extracted Frames  
| Frame 0 | Frame 24 | Frame 48 | Frame 72 | Frame 96 |  
|---------|----------|----------|----------|----------|  
|

### Reconstructed Output  
 [Download/View Output Video](sample_demo/video_output20250913_183517/output20250913_183517.avi)  

---

## Requirements  

- Python **3.8+**  
- [Pillow](https://pypi.org/project/Pillow/)  
- [OpenCV](https://pypi.org/project/opencv-python/)  
- [NumPy](https://pypi.org/project/numpy/)  
- [tqdm](https://pypi.org/project/tqdm/)  

Install with:  
```bash
pip install pillow opencv-python numpy tqdm
```  

---

## ⚡ Notes  

- Only **5 sample frames** are included for clarity; the tool works on full videos.  
- Large raw frame dumps are **git-ignored** to keep the repo lightweight.  
- The included `.avi` and frame snippets clearly demonstrate the **frame → video pipeline**.  

---
