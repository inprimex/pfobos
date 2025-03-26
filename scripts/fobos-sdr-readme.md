# Fobos SDR Setup Script

This script automates the process of creating and installing udev rules for the Fobos SDR device on Ubuntu and other Linux distributions. It allows non-root users to access the SDR hardware without requiring sudo privileges.

## Device Information

- **Device**: Fobos SDR
- **USB ID**: 16d0:132e (Vendor ID: 16d0, Product ID: 132e)
- **Manufacturer**: MCS

## Installation

1. Download the `setup-fobos-sdr.sh` script
2. Make the script executable:
   ```bash
   chmod +x setup-fobos-sdr.sh
   ```
3. Run the script with sudo:
   ```bash
   sudo ./setup-fobos-sdr.sh
   ```
4. Disconnect and reconnect your Fobos SDR device for the changes to take effect

## What the Script Does

The script performs the following operations:

1. Checks if it's running with root privileges
2. Creates a udev rule file (`/etc/udev/rules.d/99-fobos-sdr.rules`) with the following configuration:
   ```
   SUBSYSTEMS=="usb", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="132e", MODE="0660", TAG+="uaccess"
   ```
3. Sets the correct permissions for the rules file
4. Reloads the udev rules
5. Triggers the udev rules to apply them
6. Verifies if the Fobos SDR device is currently connected

## Verification

After running the script, you can verify that the rules have been applied by:

1. Connecting your Fobos SDR device
2. Running `lsusb` to confirm the device is detected
3. Trying to access the device with your software without using sudo

## Troubleshooting

If you encounter issues after running the script:

- Check that the device is properly connected using `lsusb`
- Verify the udev rule file exists: `cat /etc/udev/rules.d/99-fobos-sdr.rules`
- Reload the rules manually: `sudo udevadm control --reload-rules && sudo udevadm trigger`
- Check system logs for any USB-related errors: `dmesg | grep -i usb`
- Restart your system if all else fails

## System Requirements

- Ubuntu or other Linux distribution with udev
- Root access (sudo)
- USB port

