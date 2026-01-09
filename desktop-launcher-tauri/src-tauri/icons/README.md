# R3MES Launcher Icons

This directory contains the application icons for the R3MES Desktop Launcher.

## Required Icon Files

The following icon files are referenced in tauri.conf.json and need to be created:

- `32x32.png` - 32x32 pixel PNG icon
- `128x128.png` - 128x128 pixel PNG icon  
- `128x128@2x.png` - 256x256 pixel PNG icon (high DPI)
- `icon.icns` - macOS icon file
- `icon.ico` - Windows icon file
- `icon.png` - System tray icon

## Creating Icons

To create proper icons:

1. Design a 512x512 pixel icon in PNG format
2. Use online tools or software like:
   - https://www.icoconverter.com/ (for ICO files)
   - https://iconverticons.com/ (for ICNS files)
   - Image editing software to resize for different sizes

## Temporary Solution

For development purposes, you can:
1. Copy any PNG file and rename it to the required names
2. Use placeholder icons until proper branding is ready
3. The application will still work with missing icons (may show default system icons)

## Icon Design Guidelines

- Use the R3MES brand colors and logo
- Ensure icons are clear at small sizes (32x32)
- Use transparent backgrounds where appropriate
- Follow platform-specific icon guidelines (Windows, macOS, Linux)