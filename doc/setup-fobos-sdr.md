# Fobos SDR Setup Script

This script automates the process of creating and installing udev rules for the Fobos SDR device on Ubuntu and other Linux distributions. It allows non-root users to access the SDR hardware without requiring sudo privileges.

## Device Information

- **Device**: Fobos SDR
- **USB ID**: 16d0:132e (Vendor ID: 16d0, Product ID: 132e)
- **Manufacturer**: MCS

## Project Structure

```
project_root/
├── doc/
│   ├── rtanalyzer.md
│   └── setup-fobos-sdr.md
├── requirements.txt
├── rtanalyzer/
│   ├── __init__.py
│   └── rtanalyzer.py
├── run_rtanalyzer.py
├── run_setup.py
└── setup/
    └── setup-fobos-sdr.sh
```

## Installation

1. Clone or download the project repository
2. Navigate to the project root directory
3. Run the hardware setup script with sudo:
   ```bash
   sudo ./setup/setup-fobos-sdr.sh
   ```
4. Run the Python environment verification:
   ```bash
   python3 run_setup.py
   ```
5. Disconnect and reconnect your Fobos SDR device for the changes to take effect

## What the Scripts Do

### Hardware Setup Script (setup-fobos-sdr.sh)

The script performs the following operations:

1. Checks if it's running with root privileges
2. Creates a udev rule file (`/etc/udev/rules.d/99-fobos-sdr.rules`) with the following configuration:
   ```
   SUBSYSTEMS=="usb", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="132e", MODE="0666", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
   ```
3. Creates necessary user groups (plugdev, dialout) if they don't exist
4. Adds the current user to required groups
5. Configures system permissions for device access
6. Reloads the udev rules
7. Verifies if the Fobos SDR device is currently connected

### Python Setup Script (run_setup.py)

The script verifies:

1. Python version compatibility
2. Project structure integrity
3. Required Python dependencies
4. Hardware configuration status

## Verification

After running both scripts, you can verify the setup by:

1. Checking group membership:
   ```bash
   groups $USER
   ```
   Should show 'plugdev' and 'dialout' among the groups

2. Verifying device detection:
   ```bash
   lsusb | grep "16d0:132e"
   ```

3. Checking udev rules:
   ```bash
   cat /etc/udev/rules.d/99-fobos-sdr.rules
   ```

## Troubleshooting

If you encounter issues after running the scripts:

### Hardware Issues

- Check that the device is properly connected using `lsusb`
- Verify the udev rule file exists and has correct content
- Reload the rules manually: `sudo udevadm control --reload-rules && sudo udevadm trigger`
- Check system logs: `dmesg | grep -i usb`
- Log out and log back in for group changes to take effect
- Restart your system if all else fails

### Software Issues

- Verify Python version: `python3 --version`
- Check installed packages: `pip list`
- Install missing dependencies: `pip install -r requirements.txt`
- Verify project file permissions: `ls -l`

## System Requirements

- Ubuntu or other Linux distribution with udev
- Python 3.7 or higher
- Root access (sudo) for hardware setup
- USB port

## Additional Notes

- After adding user to new groups, you must log out and log back in for changes to take effect
- The hardware setup script must be run as root using sudo
- The Python setup script should be run as a regular user
- Make sure to run scripts from the project root directory