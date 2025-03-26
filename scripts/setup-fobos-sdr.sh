#!/bin/bash

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "This script must be run as root (with sudo)"
  exit 1
fi

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Setting up improved udev rules for Fobos SDR..."

# Get the username of the user who ran sudo
if [ -n "$SUDO_USER" ]; then
  ACTUAL_USER="$SUDO_USER"
else
  ACTUAL_USER="$(whoami)"
fi
echo "Setting up for user: $ACTUAL_USER"

# Create the udev rules file with more complete rules
cat > /etc/udev/rules.d/99-fobos-sdr.rules << 'EOL'
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
chmod 644 /etc/udev/rules.d/99-fobos-sdr.rules

# Make sure the plugdev group exists
if ! getent group plugdev > /dev/null; then
  echo "Creating plugdev group..."
  groupadd plugdev
fi

# Add the user to the plugdev group
if ! groups "$ACTUAL_USER" | grep -q plugdev; then
  echo "Adding user $ACTUAL_USER to plugdev group..."
  usermod -a -G plugdev "$ACTUAL_USER"
  echo -e "${YELLOW}Note: You may need to log out and log back in for group changes to take effect.${NC}"
fi

# Add user to dialout group (often needed for serial devices)
if ! groups "$ACTUAL_USER" | grep -q dialout; then
  echo "Adding user $ACTUAL_USER to dialout group..."
  usermod -a -G dialout "$ACTUAL_USER"
fi

# Allow dmesg access for regular users
if [ -f /etc/sysctl.d/10-kernel-hardening.conf ]; then
  if grep -q "kernel.dmesg_restrict" /etc/sysctl.d/10-kernel-hardening.conf; then
    sed -i 's/kernel.dmesg_restrict = 1/kernel.dmesg_restrict = 0/' /etc/sysctl.d/10-kernel-hardening.conf
    echo "Updated kernel.dmesg_restrict in sysctl configuration."
  else
    echo "kernel.dmesg_restrict = 0" >> /etc/sysctl.d/10-kernel-hardening.conf
  fi
else
  echo "kernel.dmesg_restrict = 0" > /etc/sysctl.d/10-kernel-hardening.conf
fi

# Apply sysctl changes
sysctl -p /etc/sysctl.d/10-kernel-hardening.conf

# Reload udev rules
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}Important: You should log out and log back in for group changes to take effect fully.${NC}"

# Optionally check if device is currently connected
if lsusb | grep -q "16d0:132e"; then
  echo "Fobos SDR device detected. Please disconnect and reconnect it for the rules to take effect."
fi

exit 0
