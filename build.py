from PIL import Image
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor

class RGBFormatConverter:
    @staticmethod
    def rgb888_to_rgb565(r, g, b):
        """Convert 24-bit RGB888 to 16-bit RGB565"""
        r5 = (r >> 3) & 0x1F
        g6 = (g >> 2) & 0x3F
        b5 = (b >> 3) & 0x1F
        return (r5 << 11) | (g6 << 5) | b5
    
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
    def rgb888_to_rgb666(r, g, b):
        """Convert 24-bit RGB888 to 18-bit RGB666"""
        r6 = (r >> 2) & 0x3F
        g6 = (g >> 2) & 0x3F
        b6 = (b >> 2) & 0x3F
        return (r6 << 12) | (g6 << 6) | b6
    
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
    def rgb888_to_rgb101010(r, g, b):
        """Convert 24-bit RGB888 to 30-bit RGB101010"""
        r10 = (r << 2) | (r >> 6)
        g10 = (g << 2) | (g >> 6)
        b10 = (b << 2) | (b >> 6)
        return (r10 << 20) | (g10 << 10) | b10
    
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
    def rgb888_to_rgb121212(r, g, b):
        """Convert 24-bit RGB888 to 36-bit RGB121212"""
        r12 = (r << 4) | (r >> 4)
        g12 = (g << 4) | (g >> 4)
        b12 = (b << 4) | (b >> 4)
        return (r12 << 24) | (g12 << 12) | b12
    
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
    def rgb888_to_loose666(r, g, b):
        """Convert 24-bit RGB888 to 18-bit loosely packed RGB (8-8-8 but only 6-6-6 significant bits)"""
        r6 = r >> 2
        g6 = g >> 2
        b6 = b >> 2
        return r6, g6, b6
    
    @staticmethod
    def loose666_to_rgb888(r6, g6, b6):
        """Convert 18-bit loosely packed RGB to 24-bit RGB888"""
        r = r6 << 2
        g = g6 << 2
        b = b6 << 2
        return r, g, b

def convert_bit_depth(pixel_grid, bit_depth):
    """Convert 8-bit YCbCr values to the target bit depth"""
    if bit_depth == 24:  # 8-8-8 (no change)
        return pixel_grid
    
    # Determine bits per component
    if bit_depth == 20:  # 8-6-6
        y_bits, cb_bits, cr_bits = 8, 6, 6
    elif bit_depth == 16:  # 8-4-4
        y_bits, cb_bits, cr_bits = 8, 4, 4
    elif bit_depth == 12:  # 4-4-4
        y_bits, cb_bits, cr_bits = 4, 4, 4
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    new_grid = []
    for row in pixel_grid:
        new_row = []
        for y, cb, cr in row:
            # Apply bit reduction
            new_y = y >> (8 - y_bits) if y_bits < 8 else y
            new_cb = cb >> (8 - cb_bits) if cb_bits < 8 else cb
            new_cr = cr >> (8 - cr_bits) if cr_bits < 8 else cr
            new_row.append((new_y, new_cb, new_cr))
        new_grid.append(new_row)
    
    return new_grid

def log_hex_values(pixel_grid, log_filename, subsampling, bit_depth):
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    with open(log_filename, 'w') as f:
        # Header based on format
        ss_label = ['4:4:4', '4:2:2', '4:2:0'][subsampling]
        f.write(f"Packed Pixel Data (Hex) - Intel Style - {ss_label}, {bit_depth}-bit\n")
        
        # 4:4:4 Format (Full resolution)
        if subsampling == 0:
            f.write("Format: [Y Cb Cr] per pixel\n")
            for y in range(height):
                for x in range(width):
                    Y, Cb, Cr = pixel_grid[y][x]
                    if bit_depth == 12:  # Pack into 2 bytes (3 nibbles)
                        byte0 = (Y << 4) | Cb
                        byte1 = Cr << 4  # Last nibble unused
                        f.write(f"Pixel ({x},{y}): {format(byte0, '02x')} {format(byte1, '02x')}\n")
                    else:
                        f.write(f"Pixel ({x},{y}): {format(Y, '02x')} {format(Cb, '02x')} {format(Cr, '02x')}\n")
        
        # 4:2:2 Format (Horizontal subsampling)
        elif subsampling == 1:
            f.write("Format: [Y0 Y1 Cb Cr] per horizontal 2x1 block\n")
            for y in range(height):
                for x in range(0, width, 2):
                    # Chroma from left pixel
                    Cb, Cr = pixel_grid[y][x][1], pixel_grid[y][x][2]
                    Y0 = pixel_grid[y][x][0]
                    Y1 = pixel_grid[y][x+1][0] if x+1 < width else 0
                    
                    if bit_depth == 12:  # 4-bit components
                        byte0 = (Y0 << 4) | Y1
                        byte1 = (Cb << 4) | Cr
                        f.write(f"Block ({x},{y}): {format(byte0, '02x')} {format(byte1, '02x')}\n")
                    elif bit_depth == 16:  # 8-4-4
                        f.write(f"Block ({x},{y}): Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Cb={format(Cb, '01x')} Cr={format(Cr, '01x')}\n")
                    elif bit_depth == 20:  # 8-6-6
                        f.write(f"Block ({x},{y}): Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Cb={format(Cb, '02x')} Cr={format(Cr, '02x')}\n")
                    else:  # 24-bit
                        f.write(f"Block ({x},{y}): Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Cb={format(Cb, '02x')} Cr={format(Cr, '02x')}\n")
        
        # 4:2:0 Format (Both horizontal and vertical subsampling)
        elif subsampling == 2:
            f.write("Format: [Cb Cr Y0 Y1 Y2 Y3] per 2x2 block\n")
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    # Chroma from top-left pixel
                    Cb, Cr = pixel_grid[y][x][1], pixel_grid[y][x][2]
                    Y0 = pixel_grid[y][x][0]
                    Y1 = pixel_grid[y][x+1][0] if x+1 < width else 0
                    Y2 = pixel_grid[y+1][x][0] if y+1 < height else 0
                    Y3 = pixel_grid[y+1][x+1][0] if (x+1 < width and y+1 < height) else 0
                    
                    if bit_depth == 12:  # 4-bit components
                        byte0 = (Cb << 4) | Cr
                        byte1 = (Y0 << 4) | Y1
                        byte2 = (Y2 << 4) | Y3
                        f.write(f"Block ({x},{y}): {format(byte0, '02x')} {format(byte1, '02x')} {format(byte2, '02x')}\n")
                    elif bit_depth == 16:  # 8-4-4
                        f.write(f"Block ({x},{y}): Cb={format(Cb, '01x')} Cr={format(Cr, '01x')} Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Y2={format(Y2, '02x')} Y3={format(Y3, '02x')}\n")
                    elif bit_depth == 20:  # 8-6-6
                        f.write(f"Block ({x},{y}): Cb={format(Cb, '02x')} Cr={format(Cr, '02x')} Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Y2={format(Y2, '02x')} Y3={format(Y3, '02x')}\n")
                    else:  # 24-bit
                        f.write(f"Block ({x},{y}): Cb={format(Cb, '02x')} Cr={format(Cr, '02x')} Y0={format(Y0, '02x')} Y1={format(Y1, '02x')} Y2={format(Y2, '02x')} Y3={format(Y3, '02x')}\n")


def process_ycbcr_frame(frame, frame_num, output_dir, new_width, new_height, ycbcr_format, bit_depth):
   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    subsampling_map = {'YCBCR444': 0, 'YCBCR422': 1, 'YCBCR420': 2}
    subsampling = subsampling_map[ycbcr_format]
    

    img = Image.fromarray(frame_rgb).convert("YCbCr")
    reimg = img.resize((new_width, new_height), Image.LANCZOS)
    width, height = reimg.size
    
   
    pixels = list(reimg.getdata())
    pixel_grid = [pixels[i*width:(i+1)*width] for i in range(height)]
    log_grid = convert_bit_depth(pixel_grid, bit_depth)
    

    frame_dir = os.path.join(output_dir, f"frame_{frame_num:04d}")
    os.makedirs(frame_dir, exist_ok=True)
    
    
    ycbcr_path = os.path.join(frame_dir, f"frame_{frame_num:04d}_ycbcr.jpg")
    reimg.save(ycbcr_path, subsampling=subsampling, quality=95)
    
    
    log_path = os.path.join(frame_dir, f"frame_{frame_num:04d}_log.txt")
    log_hex_values(log_grid, log_path, subsampling, bit_depth)
    
    
    return cv2.cvtColor(np.array(reimg.convert('RGB')), cv2.COLOR_RGB2BGR)

def process_video(input_file, new_width, new_height, processing_func, **kwargs):
    
    """Process video with both video output and frame logging"""
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
   
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    output_dir = kwargs.get('output_dir', f"video_output{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
   
    video_output_path = os.path.join(output_dir, f"processed_video_{timestamp}.mp4")
    codecs_to_try = [
    ('MJPG', '.avi'),  # Motion-JPEG
    ('XVID', '.avi'),  # XVID
    ('MP4V', '.mp4'),  # MPEG-4
    ('AVC1', '.mp4')   # H.264
    ]

    for codec, ext in codecs_to_try:
        video_output_path = os.path.join(output_dir, f"output{timestamp}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (new_width, new_height))
        if out.isOpened():
            break
        else:
            raise RuntimeError("No working codec found")
    
    print(f"Processing {frame_count} frames...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for frame_num in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            futures.append(
                executor.submit(
                    processing_func,
                    frame=frame,
                    frame_num=frame_num,
                    output_dir=output_dir,
                    new_width=new_width,
                    new_height=new_height,
                    **{k:v for k,v in kwargs.items() if k != 'output_dir'}
                )
            )
            
            if len(futures) % 10 == 0:
                print(f"Submitted {len(futures)}/{frame_count} frames")
        
        # Write frames to video in order
        for i, future in enumerate(futures):
            processed_frame = future.result()
            out.write(processed_frame)
            if i % 10 == 0:
                print(f"Processed {i+1}/{frame_count} frames")
    
    cap.release()
    out.release()
    
    print(f"\nProcessing completed!")
    print(f"- Video output: {video_output_path}")
    print(f"- Individual frames and logs in: {output_dir}")

def resize_image(input_path, new_width, new_height, rgb_format):
    """Resize image with specified RGB format"""
    converter = RGBFormatConverter()
    
    with Image.open(input_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        pixels = np.array(resized_img)
        
        if rgb_format == 'RGB565':
            converted_pixels = []
            for row in pixels:
                new_row = []
                for r, g, b in row:
                    rgb565 = converter.rgb888_to_rgb565(r, g, b)
                    r, g, b = converter.rgb565_to_rgb888(rgb565)
                    new_row.append([r, g, b])
                converted_pixels.append(new_row)
            pixels = np.array(converted_pixels, dtype=np.uint8)
        
        elif rgb_format == 'RGB666':
            converted_pixels = []
            for row in pixels:
                new_row = []
                for r, g, b in row:
                    rgb666 = converter.rgb888_to_rgb666(r, g, b)
                    r, g, b = converter.rgb666_to_rgb888(rgb666)
                    new_row.append([r, g, b])
                converted_pixels.append(new_row)
            pixels = np.array(converted_pixels, dtype=np.uint8)
        
        elif rgb_format == 'RGB101010':
            converted_pixels = []
            for row in pixels:
                new_row = []
                for r, g, b in row:
                    rgb101010 = converter.rgb888_to_rgb101010(r, g, b)
                    r, g, b = converter.rgb101010_to_rgb888(rgb101010)
                    new_row.append([r, g, b])
                converted_pixels.append(new_row)
            pixels = np.array(converted_pixels, dtype=np.uint8)
        
        elif rgb_format == 'RGB121212':
            converted_pixels = []
            for row in pixels:
                new_row = []
                for r, g, b in row:
                    rgb121212 = converter.rgb888_to_rgb121212(r, g, b)
                    r, g, b = converter.rgb121212_to_rgb888(rgb121212)
                    new_row.append([r, g, b])
                converted_pixels.append(new_row)
            pixels = np.array(converted_pixels, dtype=np.uint8)
        
        elif rgb_format == 'LOOSE666':
            converted_pixels = []
            for row in pixels:
                new_row = []
                for r, g, b in row:
                    r6, g6, b6 = converter.rgb888_to_loose666(r, g, b)
                    r, g, b = converter.loose666_to_rgb888(r6, g6, b6)
                    new_row.append([r, g, b])
                converted_pixels.append(new_row)
            pixels = np.array(converted_pixels, dtype=np.uint8)
        
        output_img = Image.fromarray(pixels)
        output_filename = f"output_{rgb_format.lower()}.jpg"
        output_img.save(output_filename)
        return pixels, output_filename

def get_pixel_values(pixels, rgb_format):
    """Get pixel values in the specified format"""
    converter = RGBFormatConverter()
    pixel_values = []
    
    for row in pixels:
        for r, g, b in row:
            if rgb_format == 'RGB565':
                val = converter.rgb888_to_rgb565(r, g, b)
                pixel_values.append(f"RGB565: {val:04X}")
            elif rgb_format == 'RGB666':
                val = converter.rgb888_to_rgb666(r, g, b)
                pixel_values.append(f"RGB666: {val:05X}")
            elif rgb_format == 'RGB101010':
                val = converter.rgb888_to_rgb101010(r, g, b)
                pixel_values.append(f"RGB101010: {val:08X}")
            elif rgb_format == 'RGB121212':
                val = converter.rgb888_to_rgb121212(r, g, b)
                pixel_values.append(f"RGB121212: {val:09X}")
            elif rgb_format == 'LOOSE666':
                r6, g6, b6 = converter.rgb888_to_loose666(r, g, b)
                pixel_values.append(f"LOOSE666: {r6:02X} {g6:02X} {b6:02X}")
            else:  # RGB888
                pixel_values.append(f"RGB888: #{r:02X}{g:02X}{b:02X}")
    
    return pixel_values

def process_rgb(input_file, new_width, new_height, rgb_format):
    """Process RGB image or video"""
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process as video
        process_video(
            input_file=input_file,
            new_width=new_width,
            new_height=new_height,
            processing_func=process_rgb_frame,
            rgb_format=rgb_format
        )
    else:
        # Process as image
        pixels, output_filename = resize_image(input_file, new_width, new_height, rgb_format)
        
        pixel_values = get_pixel_values(pixels, rgb_format)
        log_filename = f"output_{rgb_format.lower()}_pixels.txt"
        with open(log_filename, 'w') as f:
            for i, val in enumerate(pixel_values):
                f.write(f"Pixel {i}: {val}\n")
                if i == 0:
                    print(f"First pixel ({rgb_format}): {val}")
        
        print(f"\nProcessing completed. Output files:")
        print(f"- Image: {output_filename}")
        print(f"- Log: {log_filename}")

def process_rgb_frame(frame, frame_num, output_dir, new_width, new_height, rgb_format):
    """Process a single RGB frame for video processing"""
    converter = RGBFormatConverter()
    
    # Resize frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    reimg = img.resize((new_width, new_height), Image.LANCZOS)
    pixels = np.array(reimg)
    
    # Convert to target format and back for display
    converted_pixels = []
    for row in pixels:
        new_row = []
        for r, g, b in row:
            if rgb_format == 'RGB565':
                rgb565 = converter.rgb888_to_rgb565(r, g, b)
                r, g, b = converter.rgb565_to_rgb888(rgb565)
            elif rgb_format == 'RGB666':
                rgb666 = converter.rgb888_to_rgb666(r, g, b)
                r, g, b = converter.rgb666_to_rgb888(rgb666)
            elif rgb_format == 'RGB101010':
                rgb101010 = converter.rgb888_to_rgb101010(r, g, b)
                r, g, b = converter.rgb101010_to_rgb888(rgb101010)
            elif rgb_format == 'RGB121212':
                rgb121212 = converter.rgb888_to_rgb121212(r, g, b)
                r, g, b = converter.rgb121212_to_rgb888(rgb121212)
            elif rgb_format == 'LOOSE666':
                r6, g6, b6 = converter.rgb888_to_loose666(r, g, b)
                r, g, b = converter.loose666_to_rgb888(r6, g6, b6)
            new_row.append([r, g, b])
        converted_pixels.append(new_row)
    
    # Save frame and log
    frame_dir = os.path.join(output_dir, f"frame_{frame_num:04d}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Save image
    output_img = Image.fromarray(np.array(converted_pixels, dtype=np.uint8))
    img_path = os.path.join(frame_dir, f"frame_{frame_num:04d}.jpg")
    output_img.save(img_path)
    
    # Save log
    pixel_values = get_pixel_values(converted_pixels, rgb_format)
    log_path = os.path.join(frame_dir, f"frame_{frame_num:04d}_log.txt")
    with open(log_path, 'w') as f:
        for i, val in enumerate(pixel_values):
            f.write(f"Pixel {i}: {val}\n")
    
    # Return frame for video output
    return cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)

def process_ycbcr(input_file, new_width, new_height, ycbcr_format, bit_depth):
    """Process YCbCr image or video"""
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process as video
        process_video(
            input_file=input_file,
            new_width=new_width,
            new_height=new_height,
            processing_func=process_ycbcr_frame,
            ycbcr_format=ycbcr_format,
            bit_depth=bit_depth
        )
    else:
        # Process as image
        with Image.open(input_file) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_ycbcr = img.convert("YCbCr")
            reimg = img_ycbcr.resize((new_width, new_height), Image.LANCZOS)
            
            # Get pixel data for logging
            pixels = list(reimg.getdata())
            width, height = reimg.size
            pixel_grid = [pixels[i*width:(i+1)*width] for i in range(height)]
            pixel_grid = convert_bit_depth(pixel_grid, bit_depth)
            
            # Save output
            output_filename = f"output_ycbcr_{ycbcr_format.lower()}_{bit_depth}bit.jpg"
            reimg.save(output_filename, format="JPEG", 
                      subsampling={'YCBCR444': 0, 'YCBCR422': 1, 'YCBCR420': 2}[ycbcr_format], 
                      quality=95)
            
            # Save log file
            log_filename = f"output_ycbcr_{ycbcr_format.lower()}_{bit_depth}bit_pixels.txt"
            log_hex_values(pixel_grid, log_filename, 
                          {'YCBCR444': 0, 'YCBCR422': 1, 'YCBCR420': 2}[ycbcr_format], 
                          bit_depth)
            
            print(f"\nProcessing completed. Output files:")
            print(f"- Image: {output_filename}")
            print(f"- Log: {log_filename}")

if __name__ == "__main__":
    try:
        input_file = input("Enter input file path (image or video): ")
        
        # Get common parameters
        new_width = int(input("Enter new width: "))
        new_height = int(input("Enter new height: "))
        
        # Choose color space
        print("\nSelect color space:")
        print("1. RGB")
        print("2. YCbCr")
        color_space = int(input("Enter choice (1-2): "))
        
        if color_space == 1:  # RGB
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
            rgb_format = format_map.get(choice, 'RGB888')
            
            process_rgb(input_file, new_width, new_height, rgb_format)
            
        elif color_space == 2:  # YCbCr
            print("\nAvailable YCbCr formats:")
            print("1. YCBCR 444 (4:4:4)")
            print("2. YCBCR 422 (4:2:2)")
            print("3. YCBCR 420 (4:2:0)")
            choice = int(input("Select YCbCr format (1-3): "))
            ycbcr_format = ['YCBCR444', 'YCBCR422', 'YCBCR420'][choice-1]
            
            print("\nAvailable bit depths:")
            print("1. 24-bit (8-8-8)")
            print("2. 20-bit (8-6-6)")
            print("3. 16-bit (8-4-4)")
            print("4. 12-bit (4-4-4)")
            depth_choice = int(input("Select bit depth (1-4): "))
            bit_depth = [24, 20, 16, 12][depth_choice-1]
            
            process_ycbcr(input_file, new_width, new_height, ycbcr_format, bit_depth)
            
        else:
            print("Invalid color space choice")
            
    except Exception as e:
        print(f"Error: {str(e)}")
