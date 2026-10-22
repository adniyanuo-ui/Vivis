#!/usr/bin/env python3
"""Temporarily switch only Xiaobai's FE-URT-2 between CDC and WCH drivers.

No servo packets, EEPROM changes, system-wide driver unload, or persistent install.
Run `status` without sudo; `vendor` and `cdc` require root. Failed vendor switches
attempt to restore cdc_acm on the same device.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SERIAL = os.environ.get('XIAOBAI_USB_SERIAL', '5B79033160')
MODULE = ROOT / 'xiaobai/reference/ch343ser_linux/driver/ch343.ko'


def identity(path, name):
    try:
        return (path / name).read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return None


def find_interface():
    matches = [p for p in Path('/sys/bus/usb/devices').iterdir()
               if identity(p, 'idVendor') == '1a86' and identity(p, 'idProduct') == '55d3'
               and identity(p, 'serial') == SERIAL]
    if len(matches) != 1:
        raise RuntimeError('Need exactly one matching FE-URT-2 serial number')
    device = matches[0].resolve()
    interfaces = [p for p in device.iterdir()
                  if identity(p, 'bInterfaceClass') == '02']
    if len(interfaces) != 1:
        raise RuntimeError('Unexpected USB control interface layout')
    return device, interfaces[0]


def driver(interface):
    link = interface / 'driver'
    return link.resolve().name if link.exists() else None


def tty_devices(device):
    return sorted('/dev/' + p.name for p in Path('/sys/class/tty').iterdir()
                  if p.resolve().is_relative_to(device))


def bind(interface, target):
    current = driver(interface)
    if current == target:
        return
    if current not in [None, 'cdc_acm', 'usb_ch343']:
        raise RuntimeError('Unexpected driver; refusing to unbind ' + current)
    if current:
        (interface / 'driver/unbind').write_text(interface.name)
    if driver(interface) != target:
        (Path('/sys/bus/usb/drivers') / target / 'bind').write_text(interface.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'vendor', 'cdc'])
    args = parser.parse_args()
    access_user = os.environ.get('SUDO_USER') or os.environ.get('USER')
    if args.action != 'status' and not access_user:
        raise SystemExit('Cannot determine serial access user; set SUDO_USER.')
    device, interface = find_interface()
    report = {'serial': SERIAL, 'usb_device': str(device), 'interface': interface.name,
              'driver_before': driver(interface), 'ttys_before': tty_devices(device),
              'action': args.action, 'permanent_install': False}
    if args.action != 'status':
        if os.geteuid() != 0:
            raise SystemExit('This action needs sudo to load/bind a kernel driver.')
        for tty in tty_devices(device):
            result = subprocess.run(['fuser', tty], capture_output=True, text=True)
            if result.returncode == 0:
                raise RuntimeError('Serial device is in use; stop its owner first: ' + result.stdout)
            if result.returncode != 1:
                raise RuntimeError('Cannot establish whether serial device is in use: ' + result.stderr)
        if args.action == 'vendor':
            vermagic = subprocess.check_output(['modinfo', '-F', 'vermagic', str(MODULE)], text=True)
            if vermagic.split()[0] != os.uname().release:
                raise RuntimeError('Driver was built for a different kernel')
            if not Path('/sys/module/ch343').exists():
                subprocess.run(['insmod', str(MODULE)], check=True)
            target = 'usb_ch343'
        else:
            subprocess.run(['modprobe', 'cdc_acm'], check=True)
            target = 'cdc_acm'
        try:
            bind(interface, target)
            subprocess.run(['udevadm', 'settle', '--timeout=5'], check=True)
            for _ in range(50):
                ttys = tty_devices(device)
                if ttys and all(Path(p).exists() for p in ttys):
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('No serial node appeared after binding')
            if driver(interface) != target:
                raise RuntimeError('Driver binding did not stick')
            for tty in ttys:
                subprocess.run(['setfacl', '-m', f'u:{access_user}:rw', tty], check=True)
        except Exception:
            if args.action == 'vendor':
                subprocess.run(['modprobe', 'cdc_acm'], check=True)
                bind(interface, 'cdc_acm')
                subprocess.run(['udevadm', 'settle', '--timeout=5'], check=False)
                for tty in tty_devices(device):
                    subprocess.run(['setfacl', '-m', f'u:{access_user}:rw', tty], check=False)
            raise
    report.update(driver_after=driver(interface), ttys_after=tty_devices(device))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
