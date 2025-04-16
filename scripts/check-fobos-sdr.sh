#!/bin/bash
# Enhanced Script to check Fobos SDR device connectivity and permissions

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define Fobos SDR vendor and product IDs
VENDOR_ID="16d0"
PRODUCT_ID="132e"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       Fobos SDR Device Check Tool         ${NC}"
echo -e "${BLUE}============================================${NC}"
echo

echo "Checking for Fobos SDR device (ID ${VENDOR_ID}:${PRODUCT_ID})..."

# Check if lsusb is available
if ! command -v lsusb &> /dev/null; then
    echo -e "${RED}Error: lsusb command not found. Please install usbutils package.${NC}"
    echo "  On Debian/Ubuntu: sudo apt install usbutils"
    echo "  On Fedora/RHEL: sudo dnf install usbutils"
    echo "  On Arch: sudo pacman -S usbutils"
    exit 1
fi

# Check if script is run with sudo and warn if it is
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: You are running this script as root. This is useful for diagnosis but"
    echo -e "proper setup should allow your regular user to access the device.${NC}"
fi

# Check if the device is present using lsusb
DEVICE_INFO=$(lsusb | grep -i "${VENDOR_ID}:${PRODUCT_ID}")

if [ -n "$DEVICE_INFO" ]; then
    echo -e "${GREEN}✓ Fobos SDR device found:${NC}"
    echo -e "${GREEN}$DEVICE_INFO${NC}"
    
    # Extract bus and device numbers for more detailed checks
    BUS=$(echo $DEVICE_INFO | grep -o "Bus [0-9]*" | cut -d' ' -f2)
    DEVICE=$(echo $DEVICE_INFO | grep -o "Device [0-9]*" | cut -d' ' -f2)
    
    # Check if the device has the correct permissions
    DEV_PATH="/dev/bus/usb/$BUS/$DEVICE"
    if [ -e "$DEV_PATH" ]; then
        PERMISSIONS=$(ls -l "$DEV_PATH" | awk '{print $1}')
        GROUP=$(ls -l "$DEV_PATH" | awk '{print $4}')
        echo "Device path: $DEV_PATH"
        echo "Current permissions: $PERMISSIONS"
        echo "Current group: $GROUP"
        
        # Check if current user has access
        if [ -r "$DEV_PATH" ] && [ -w "$DEV_PATH" ]; then
            echo -e "${GREEN}✓ Current user has read/write access to the device.${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: Current user may not have proper access to the device.${NC}"
            
            # Check for udev rules
            if [ -f "/etc/udev/rules.d/99-fobos-sdr.rules" ]; then
                echo -e "${BLUE}ℹ Fobos SDR udev rules file exists:${NC}"
                echo "  $(cat /etc/udev/rules.d/99-fobos-sdr.rules)"
                echo -e "${YELLOW}Try disconnecting and reconnecting the device, or log out and log back in.${NC}"
            else
                echo -e "${YELLOW}Fobos SDR udev rules file not found.${NC}"
                echo "Run the setup script to create proper udev rules:"
                echo "  sudo ./setup/setup-fobos-sdr.sh"
            fi
            
            # Check user groups
            if groups | grep -qE '(plugdev|dialout)'; then
                echo -e "${GREEN}✓ User is in required groups:${NC}"
                groups | grep -E 'plugdev|dialout'
            else
                echo -e "${YELLOW}⚠ User is not in required groups.${NC}"
                echo "You should be in these groups: plugdev, dialout"
                echo "Run the setup script or use these commands:"
                echo "  sudo usermod -a -G plugdev,dialout $USER"
                echo "  (Log out and log back in afterward)"
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Could not access device path: $DEV_PATH${NC}"
    fi
    
    # Check if the device appears in dmesg
    echo -e "\n${BLUE}Recent kernel messages about this device:${NC}"
    if dmesg 2>/dev/null | grep -i usb | grep -i "${VENDOR_ID}:${PRODUCT_ID}" | tail -5 | grep -q .; then
        dmesg 2>/dev/null | grep -i usb | grep -i "${VENDOR_ID}:${PRODUCT_ID}" | tail -5
    else
        echo "No recent kernel messages found for this device."
        echo "Try running: sudo dmesg -c && sudo dmesg --follow"
        echo "Then disconnect and reconnect the device to see messages."
    fi
    
    # Check if Python and the shared module can import properly
    echo -e "\n${BLUE}Testing Python module import:${NC}"
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    if [ -f "$PROJECT_ROOT/shared/fwrapper.py" ]; then
        echo -e "${GREEN}✓ Found fwrapper.py module at: $PROJECT_ROOT/shared/fwrapper.py${NC}"
        
        # Try to import the module
        cd "$PROJECT_ROOT"
        if python3 -c "import sys; sys.path.append('$PROJECT_ROOT'); from shared.fwrapper import FobosSDR; print('✓ Successfully imported FobosSDR module')" 2>/dev/null; then
            echo -e "${GREEN}✓ Python import test passed${NC}"
        else
            echo -e "${YELLOW}⚠ Could not import FobosSDR module in Python${NC}"
            echo "Check for errors by running:"
            echo "  cd $PROJECT_ROOT && python3 -c \"import sys; sys.path.append('$PROJECT_ROOT'); from shared.fwrapper import FobosSDR\""
        fi
    else
        echo -e "${YELLOW}⚠ Could not find fwrapper.py module${NC}"
        echo "Expected at: $PROJECT_ROOT/shared/fwrapper.py"
        echo "Make sure you're running this script from the scripts directory."
    fi
    
    # Final status
    echo -e "\n${GREEN}✓ Device check completed.${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo "If you continue to have issues, try:"
    echo "1. Disconnect and reconnect the device"
    echo "2. Run the setup script: sudo ./setup/setup-fobos-sdr.sh"
    echo "3. Log out and log back in to apply group changes"
    echo "4. Make sure the Fobos SDR library is installed"
    
    exit 0
else
    echo -e "${RED}✗ Fobos SDR device not found!${NC}"
    echo -e "Make sure the device is properly connected and powered on."
    
    # List all USB devices
    echo -e "\n${BLUE}All connected USB devices:${NC}"
    lsusb
    
    # Check if the Fobos SDR library is installed
    echo -e "\n${BLUE}Checking for Fobos SDR library:${NC}"
    if [ -f "/usr/lib/libfobos.so" ] || [ -f "/usr/local/lib/libfobos.so" ]; then
        echo -e "${GREEN}✓ Fobos SDR library found${NC}"
    elif [ -f "./libfobos.so" ] || [ -f "../libfobos.so" ]; then
        echo -e "${GREEN}✓ Fobos SDR library found in local directory${NC}"
    else
        echo -e "${YELLOW}⚠ Could not find Fobos SDR library in standard locations${NC}"
        echo "Make sure the library is installed or place it in the project directory."
    fi
    
    echo -e "\n${RED}Device check failed.${NC}"
    echo -e "${BLUE}============================================${NC}"
    exit 1
fi
