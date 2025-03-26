#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define Fobos SDR vendor and product IDs
VENDOR_ID="16d0"
PRODUCT_ID="132e"

echo "Checking for Fobos SDR device (ID ${VENDOR_ID}:${PRODUCT_ID})..."

# Check if lsusb is available
if ! command -v lsusb &> /dev/null; then
    echo -e "${RED}Error: lsusb command not found. Please install usbutils package.${NC}"
    echo "You can install it with: sudo apt install usbutils"
    exit 1
fi

# Check if the device is present using lsusb
DEVICE_INFO=$(lsusb | grep -i "${VENDOR_ID}:${PRODUCT_ID}")

if [ -n "$DEVICE_INFO" ]; then
    echo -e "${GREEN}Fobos SDR device found:${NC}"
    echo -e "${GREEN}$DEVICE_INFO${NC}"
    
    # Extract bus and device numbers for more detailed checks
    BUS=$(echo $DEVICE_INFO | grep -o "Bus [0-9]*" | cut -d' ' -f2)
    DEVICE=$(echo $DEVICE_INFO | grep -o "Device [0-9]*" | cut -d' ' -f2)
    
    # Check if the device has the correct permissions
    DEV_PATH="/dev/bus/usb/$BUS/$DEVICE"
    if [ -e "$DEV_PATH" ]; then
        PERMISSIONS=$(ls -l "$DEV_PATH" | awk '{print $1}')
        echo "Device path: $DEV_PATH"
        echo "Current permissions: $PERMISSIONS"
        
        # Check if current user has access
        if [ -r "$DEV_PATH" ] && [ -w "$DEV_PATH" ]; then
            echo -e "${GREEN}Current user has read/write access to the device.${NC}"
        else
            echo -e "${YELLOW}Warning: Current user may not have proper access to the device.${NC}"
            echo "You might need to set up udev rules or run your application with sudo."
        fi
    fi
    
    # Check if the device appears in dmesg
    echo -e "\nRecent kernel messages about this device:"
    dmesg | grep -i usb | grep -i "${VENDOR_ID}:${PRODUCT_ID}" | tail -5
    
    exit 0
else
    echo -e "${RED}Fobos SDR device not found!${NC}"
    echo -e "Make sure the device is properly connected and powered on."
    echo -e "\nAll connected USB devices:"
    lsusb
    exit 1
fi
