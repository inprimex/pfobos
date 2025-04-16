#!/bin/bash

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "This script must be run as root (with sudo). Run: sudo ./setup/setup-fobos-sdr.sh from project root"
  exit 1
fi

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Setting up Fobos SDR device permissions..."

# Get the username of the user who ran sudo
if [ -n "$SUDO_USER" ]; then
  ACTUAL_USER="$SUDO_USER"
else
  ACTUAL_USER="$(whoami)"
fi
echo "Setting up for user: $ACTUAL_USER"

# Create the udev rules file with complete rules
UDEV_RULES_FILE="/etc/udev/rules.d/99-fobos-sdr.rules"
cat > "$UDEV_RULES_FILE" << 'EOL'
# Fobos SDR Device
SUBSYSTEMS=="usb", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="132e", MODE="0666", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
EOL

# Check if file was created successfully
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Udev rules file created successfully.${NC}"
else
  echo -e "${RED}Failed to create udev rules file.${NC}"
  exit 1
fi

# Set proper permissions
chmod 644 "$UDEV_RULES_FILE"

# Make sure the plugdev group exists
if ! getent group plugdev > /dev/null; then
  echo "Creating plugdev group..."
  groupadd plugdev
fi

# Add the user to the plugdev group
if ! groups "$ACTUAL_USER" | grep -q plugdev; then
  echo "Adding user $ACTUAL_USER to plugdev group..."
  usermod -a -G plugdev "$ACTUAL_USER"
  GROUPS_CHANGED=1
fi

# Add user to dialout group (often needed for serial devices)
if ! groups "$ACTUAL_USER" | grep -q dialout; then
  echo "Adding user $ACTUAL_USER to dialout group..."
  usermod -a -G dialout "$ACTUAL_USER"
  GROUPS_CHANGED=1
fi

# Allow dmesg access for regular users
SYSCTL_CONF="/etc/sysctl.d/10-kernel-hardening.conf"
if [ -f "$SYSCTL_CONF" ]; then
  if grep -q "kernel.dmesg_restrict" "$SYSCTL_CONF"; then
    sed -i 's/kernel.dmesg_restrict = 1/kernel.dmesg_restrict = 0/' "$SYSCTL_CONF"
    echo "Updated kernel.dmesg_restrict in sysctl configuration."
  else
    echo "kernel.dmesg_restrict = 0" >> "$SYSCTL_CONF"
  fi
else
  echo "kernel.dmesg_restrict = 0" > "$SYSCTL_CONF"
fi

# Apply sysctl changes
sysctl -p "$SYSCTL_CONF"

# Reload udev rules
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo -e "${GREEN}Setup completed successfully!${NC}"

# Print summary and next steps
if [ -n "$GROUPS_CHANGED" ]; then
    echo -e "${YELLOW}Important: You need to log out and log back in for group changes to take effect.${NC}"
fi

# Check if device is currently connected
FOBOS_DEVICE=$(lsusb | grep "16d0:132e")
if [ -n "$FOBOS_DEVICE" ]; then
    echo -e "\nFobos SDR device detected!"
    echo "Device details: $FOBOS_DEVICE"
    echo -e "${YELLOW}Please disconnect and reconnect the device for the new rules to take effect.${NC}"
else
    echo -e "\n${YELLOW}No Fobos SDR device currently detected. Connect the device and run 'lsusb' to verify detection.${NC}"
fi

exit 0