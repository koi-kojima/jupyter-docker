#!/usr/bin/env python3
import cv2
import torch

print(f"{torch.cuda.is_available()=}")
print(f"{cv2.cuda.getCudaEnabledDeviceCount()=}")
if torch.cuda.is_available():
    for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
        print(f"Device {i}", '"' + torch.cuda.get_device_name(i) + '"', torch.cuda.get_device_capability(i))
