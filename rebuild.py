import os
import re
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

class RGBFormatConverter:
    @staticmethod
    def rgb565_to_rgb888(rgb565):
        """Convert 16-bit RGB565 to 24-bit RGB888"""
        r5 = (rgb565 >> 11) & 0x1F
        g6 = (rgb565 >> 5) & 0x3F
        b5 = rgb565 & 0x1F
        r = (r5 << 3) | (r5 >> 2)
        g = (g6 << 2) | (g6 >> 4)
        b = (b5 << 3) | (b5 >> 2)
        return r, g, b
    
    @staticmethod
    def rgb666_to_rgb888(rgb666):
        """Convert 18-bit RGB666 to 24-bit RGB888"""
        r6 = (rgb666 >> 12) & 0x3F
        g6 = (rgb666 >> 6) & 0x3F
        b6 = rgb666 & 0x3F
        r = (r6 << 2) | (r6 >> 4)
        g = (g6 << 2) | (g6 >> 4)
        b = (b6 << 2) | (b6 >> 4)
        return r, g, b
    
    @staticmethod
    def rgb101010_to_rgb888(rgb101010):
        """Convert 30-bit RGB101010 to 24-bit RGB888"""
        r10 = (rgb101010 >> 20) & 0x3FF
        g10 = (rgb101010 >> 10) & 0x3FF
        b10 = rgb101010 & 0x3FF
        r = r10 >> 2
        g = g10 >> 2
        b = b10 >> 2
        return r, g, b
    
    @staticmethod
    def rgb121212_to_rgb888(rgb121212):
        """Convert 36-bit RGB121212 to 24-bit RGB888"""
        r12 = (rgb121212 >> 24) & 0xFFF
        g12 = (rgb121212 >> 12) & 0xFFF
        b12 = rgb121212 & 0xFFF
        r = r12 >> 4
        g = g12 >> 4
        b = b12 >> 4
        return r, g, b
    
    @staticmethod
    def loose666_to_rgb888(r6, g6, b6):
        """Convert 18-bit loosely packed RGB to 24-bit RGB888"""
        r = r6 << 2
        g = g6 << 2
        b = b6 << 2
        return r, g, b

def detect_original_fps(input_dir):
    """Try to find the original video file and detect its FPS"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(video_extensions):
            video_path = os.path.join(input_dir, file)
            try:
                video = cv2.VideoCapture(video_path)
                fps = video.get(cv2.CAP_PROP_FPS)
                video.release()
                if fps > 0:
                    return fps
            except:
                continue
    return None

def parse_rgb_log(log_path, width, height, rgb_format):
    """Parse RGB log file and return pixel data"""
    converter = RGBFormatConverter()
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    
    with open(log_path, 'r') as f:
        for line in f:
            if not line.startswith("Pixel"):
                continue
                
            parts = line.split(":")
            index = int(parts[0].split()[1])
            x = index % width
            y = index // width
            
            if rgb_format == 'RGB888':
                match = re.search(r"#([0-9A-Fa-f]{6})", line)
                if match:
                    hex_val = match.group(1)
                    r = int(hex_val[0:2], 16)
                    g = int(hex_val[2:4], 16)
                    b = int(hex_val[4:6], 16)
                    pixels[y, x] = [r, g, b]
            
            elif rgb_format == 'RGB565':
                match = re.search(r"RGB565: ([0-9A-Fa-f]{4})", line)
                if match:
                    rgb565 = int(match.group(1), 16)
                    r, g, b = converter.rgb565_to_rgb888(rgb565)
                    pixels[y, x] = [r, g, b]
            
            elif rgb_format == 'RGB666':
                match = re.search(r"RGB666: ([0-9A-Fa-f]{5})", line)
                if match:
                    rgb666 = int(match.group(1), 16)
                    r, g, b = converter.rgb666_to_rgb888(rgb666)
                    pixels[y, x] = [r, g, b]
            
            elif rgb_format == 'RGB101010':
                match = re.search(r"RGB101010: ([0-9A-Fa-f]{8})", line)
                if match:
                    rgb101010 = int(match.group(1), 16)
                    r, g, b = converter.rgb101010_to_rgb888(rgb101010)
                    pixels[y, x] = [r, g, b]
            
            elif rgb_format == 'RGB121212':
                match = re.search(r"RGB121212: ([0-9A-Fa-f]{9})", line)
                if match:
                    rgb121212 = int(match.group(1), 16)
                    r, g, b = converter.rgb121212_to_rgb888(rgb121212)
                    pixels[y, x] = [r, g, b]
            
            elif rgb_format == 'LOOSE666':
                match = re.search(r"LOOSE666:\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})", line)
                if match:
                    r6 = int(match.group(1), 16)
                    g6 = int(match.group(2), 16)
                    b6 = int(match.group(3), 16)
                    r, g, b = converter.loose666_to_rgb888(r6, g6, b6)
                    pixels[y, x] = [r, g, b]
    
    return pixels

def parse_ycbcr_log(log_path, width, height, ycbcr_format, bit_depth):
    """Parse YCbCr log file and return pixel data"""
    subsampling = 0 if "444" in ycbcr_format else 1 if "422" in ycbcr_format else 2
    
    if bit_depth == 24:
        bits = (8, 8, 8)
    elif bit_depth == 20:
        bits = (8, 6, 6)
    elif bit_depth == 16:
        bits = (8, 4, 4)
    elif bit_depth == 12:
        bits = (4, 4, 4)
    else:
        raise ValueError("Unsupported bit depth")

    ycbcr_img = np.zeros((height, width, 3), dtype=np.uint8)

    with open(log_path, 'r') as f:
        for line in f:
            if subsampling == 0 and "Pixel" in line:
                # 4:4:4 format
                parts = line.strip().split(":")[-1].strip().split()
                coords = re.search(r"Pixel \((\d+),(\d+)\)", line)
                x, y = int(coords.group(1)), int(coords.group(2))
                
                if bit_depth == 12:
                    byte0 = int(parts[0], 16)
                    byte1 = int(parts[1], 16)
                    y_val = (byte0 >> 4) << (8 - bits[0])
                    cb = (byte0 & 0xF) << (8 - bits[1])
                    cr = (byte1 >> 4) << (8 - bits[2])
                else:
                    y_val = int(parts[0], 16) << (8 - bits[0])
                    cb = int(parts[1], 16) << (8 - bits[1])
                    cr = int(parts[2], 16) << (8 - bits[2])
                
                ycbcr_img[y, x] = [y_val, cb, cr]

            elif subsampling == 1 and "Block" in line:
                # 4:2:2 format
                parts = line.strip().split(":")[-1].strip().split()
                coords = re.search(r"Block \((\d+),(\d+)\)", line)
                x, y = int(coords.group(1)), int(coords.group(2))
                
                if bit_depth == 12:
                    byte0 = int(parts[0], 16)
                    byte1 = int(parts[1], 16)
                    y0 = (byte0 >> 4) << (8 - bits[0])
                    y1 = (byte0 & 0xF) << (8 - bits[0])
                    cb = (byte1 >> 4) << (8 - bits[1])
                    cr = (byte1 & 0xF) << (8 - bits[2])
                else:
                    y0 = int(parts[0].split("=")[1], 16) << (8 - bits[0])
                    y1 = int(parts[1].split("=")[1], 16) << (8 - bits[0])
                    cb = int(parts[2].split("=")[1], 16) << (8 - bits[1])
                    cr = int(parts[3].split("=")[1], 16) << (8 - bits[2])
                
                ycbcr_img[y, x] = [y0, cb, cr]
                if x + 1 < width:
                    ycbcr_img[y, x + 1] = [y1, cb, cr]

            elif subsampling == 2 and "Block" in line:
                # 4:2:0 format
                parts = line.strip().split(":")[-1].strip().split()
                coords = re.search(r"Block \((\d+),(\d+)\)", line)
                x, y = int(coords.group(1)), int(coords.group(2))
                
                if bit_depth == 12:
                    byte0 = int(parts[0], 16)
                    byte1 = int(parts[1], 16)
                    byte2 = int(parts[2], 16)
                    cb = (byte0 >> 4) << (8 - bits[1])
                    cr = (byte0 & 0xF) << (8 - bits[2])
                    y0 = (byte1 >> 4) << (8 - bits[0])
                    y1 = (byte1 & 0xF) << (8 - bits[0])
                    y2 = (byte2 >> 4) << (8 - bits[0])
                    y3 = (byte2 & 0xF) << (8 - bits[0])
                else:
                    cb = int(parts[0].split("=")[1], 16) << (8 - bits[1])
                    cr = int(parts[1].split("=")[1], 16) << (8 - bits[2])
                    y0 = int(parts[2].split("=")[1], 16) << (8 - bits[0])
                    y1 = int(parts[3].split("=")[1], 16) << (8 - bits[0])
                    y2 = int(parts[4].split("=")[1], 16) << (8 - bits[0])
                    y3 = int(parts[5].split("=")[1], 16) << (8 - bits[0])
                
                ycbcr_img[y, x] = [y0, cb, cr]
                if x + 1 < width:
                    ycbcr_img[y, x + 1] = [y1, cb, cr]
                if y + 1 < height:
                    ycbcr_img[y + 1, x] = [y2, cb, cr]
                    if x + 1 < width:
                        ycbcr_img[y + 1, x + 1] = [y3, cb, cr]

    return ycbcr_img

def process_frame(frame_dir, frame_num, width, height, color_space, format_str=None):
    """Process a single frame from its log file"""
    log_path = os.path.join(frame_dir, f"frame_{frame_num:04d}_log.txt")
    
    if color_space == 'RGB':
        rgb_pixels = parse_rgb_log(log_path, width, height, format_str)
        return cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2BGR)
    else:
        # Split format_str into ycbcr_format and bit_depth
        parts = format_str.split('_')
        ycbcr_format = parts[0]
        bit_depth = int(parts[1].replace('bit', ''))
        
        ycbcr_img = parse_ycbcr_log(log_path, width, height, ycbcr_format, bit_depth)
        rgb_img = Image.fromarray(ycbcr_img, mode='YCbCr').convert('RGB')
        return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

def rebuild_video(input_dir, output_dir, width, height, color_space, format_str=None, fps=None):
    """Rebuild video from log files with automatic FPS detection"""
    if fps is None:
        detected_fps = detect_original_fps(input_dir)
        if detected_fps is not None:
            print(f"Detected original video FPS: {detected_fps:.2f}")
            fps = detected_fps
        else:
            fps = 30.0
            print(f"Warning: Could not detect original FPS. Using default {fps} FPS.")
    
    frame_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith('frame_')])
    if not frame_dirs:
        raise ValueError("No frame directories found in input directory")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    codecs_to_try = [
        ('MJPG', '.avi'),
        ('XVID', '.avi'),
        ('MP4V', '.mp4'),
        ('AVC1', '.mp4')
    ]
    
    video_writer = None
    video_output_path = ""
    
    for codec, ext in codecs_to_try:
        video_output_path = os.path.join(output_dir, f"reconstructed_{timestamp}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print(f"Using codec: {codec}")
            break
        else:
            video_writer = None
    
    if video_writer is None:
        raise RuntimeError("No working codec found")
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for frame_num in range(len(frame_dirs)):
            futures.append(
                executor.submit(
                    process_frame,
                    frame_dir=os.path.join(input_dir, frame_dirs[frame_num]),
                    frame_num=frame_num,
                    width=width,
                    height=height,
                    color_space=color_space,
                    format_str=format_str
                )
            )
        
        for future in tqdm(futures, desc="Rebuilding frames"):
            frame = future.result()
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Video successfully rebuilt to {video_output_path}")

def get_ycbcr_parameters():
    """Get YCbCr format and bit depth from user"""
    print("\nAvailable YCbCr formats:")
    print("1. YCBCR 444 (4:4:4)")
    print("2. YCBCR 422 (4:2:2)")
    print("3. YCBCR 420 (4:2:0)")
    format_choice = int(input("Select YCbCr format (1-3): "))
    ycbcr_format = ['YCBCR444', 'YCBCR422', 'YCBCR420'][format_choice-1]
    
    print("\nAvailable bit depths:")
    print("1. 24-bit (8-8-8)")
    print("2. 20-bit (8-6-6)")
    print("3. 16-bit (8-4-4)")
    print("4. 12-bit (4-4-4)")
    depth_choice = int(input("Select bit depth (1-4): "))
    bit_depth = [24, 20, 16, 12][depth_choice-1]
    
    return ycbcr_format, bit_depth

def main():
    print("Video Reconstruction Tool")
    print("------------------------\n")
    
    input_dir = input("Enter directory containing frame logs: ").strip()
    output_dir = input("Enter output directory for reconstructed video: ").strip()
    width = int(input("Enter video width: ").strip())
    height = int(input("Enter video height: ").strip())
    
    # Auto-detect FPS with option to override
    detected_fps = detect_original_fps(input_dir)
    if detected_fps is not None:
        print(f"\nDetected original video FPS: {detected_fps:.2f}")
        use_detected = input("Use this FPS? (Y/n): ").strip().lower() != 'n'
        fps = detected_fps if use_detected else float(input("Enter custom FPS: "))
    else:
        print("\nWarning: Could not detect original FPS.")
        fps = float(input("Enter video FPS (default 30): ") or 30)
    
    print("\nSelect color space:")
    print("1. RGB")
    print("2. YCbCr")
    color_space_choice = int(input("Enter choice (1-2): "))
    
    if color_space_choice == 1:  # RGB
        print("\nAvailable RGB formats:")
        print("1. RGB888 (24-bit)")
        print("2. RGB565 (16-bit)")
        print("3. RGB666 (18-bit)")
        print("4. RGB101010 (30-bit)")
        print("5. RGB121212 (36-bit)")
        print("6. LOOSE666 (18-bit loosely packed)")
        
        choice = int(input("Select RGB format (1-6): "))
        format_map = {
            1: 'RGB888',
            2: 'RGB565',
            3: 'RGB666',
            4: 'RGB101010',
            5: 'RGB121212',
            6: 'LOOSE666'
        }
        format_str = format_map.get(choice, 'RGB888')
        rebuild_video(input_dir, output_dir, width, height, 'RGB', format_str, fps)
    else:  # YCbCr
        ycbcr_format, bit_depth = get_ycbcr_parameters()
        # For YCbCr, we encode the format and bit depth in the format_str
        format_str = f"{ycbcr_format}_{bit_depth}bit"
        rebuild_video(input_dir, output_dir, width, height, 'YCbCr', format_str, fps)

if __name__ == "__main__":
    main()
