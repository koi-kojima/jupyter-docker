#!/usr/bin/env python3
import torch
import cv2

print(f"{torch.cuda.is_available()=}")
print(f"{cv2.cuda.getCudaEnabledDeviceCount()=}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_capability())

