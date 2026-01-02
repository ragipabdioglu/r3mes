#!/usr/bin/env python3
"""
Create placeholder icons for R3MES Desktop Launcher
This script creates simple colored square icons as placeholders
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL (Pillow) not available. Install with: pip install Pillow")

import os

def create_placeholder_icon(size, filename, text="R3"):
    """Create a simple placeholder icon"""
    if not PIL_AVAILABLE:
        print(f"Cannot create {filename} - PIL not available")
        return
    
    # Create image with R3MES brand colors
    img = Image.new('RGBA', (size, size), (6, 182, 212, 255))  # Cyan-500
    draw = ImageDraw.Draw(img)
    
    # Add border
    border_width = max(1, size // 32)
    draw.rectangle([0, 0, size-1, size-1], outline=(3, 105, 161, 255), width=border_width)  # Cyan-700
    
    # Add text
    try:
        font_size = size // 3
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    img.save(filename)
    print(f"Created {filename}")

def main():
    if not PIL_AVAILABLE:
        print("Creating placeholder files instead of actual images...")
        # Create empty files as placeholders
        files = [
            "32x32.png",
            "128x128.png", 
            "128x128@2x.png",
            "icon.png",
            "icon.ico",
            "icon.icns"
        ]
        
        for filename in files:
            with open(filename, 'w') as f:
                f.write(f"# Placeholder for {filename}\n")
                f.write("# Replace with actual icon file\n")
            print(f"Created placeholder {filename}")
        return
    
    # Create PNG icons
    create_placeholder_icon(32, "32x32.png")
    create_placeholder_icon(128, "128x128.png")
    create_placeholder_icon(256, "128x128@2x.png")  # High DPI version
    create_placeholder_icon(64, "icon.png")  # System tray icon
    
    # For ICO and ICNS, we'll create PNG versions as placeholders
    # In production, these should be converted to proper formats
    create_placeholder_icon(256, "icon.ico.png")  # Will need conversion to ICO
    create_placeholder_icon(512, "icon.icns.png")  # Will need conversion to ICNS
    
    print("\nPlaceholder icons created!")
    print("Note: ICO and ICNS files need to be converted from PNG versions")
    print("Use online converters or proper tools for production builds")

if __name__ == "__main__":
    main()