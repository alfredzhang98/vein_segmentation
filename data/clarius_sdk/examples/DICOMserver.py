#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pynetdicom import (
    AE, evt, debug_logger,
    StoragePresentationContexts,
    VerificationPresentationContexts
)
from pydicom.dataset import Dataset

HOST = '192.168.1.104'
PORT = 2048
AE_TITLE = b'TEST123'
STORAGE_DIR = './dicom_storage'

# debug_logger()  # 如需网络级别调试可打开

os.makedirs(STORAGE_DIR, exist_ok=True)

def handle_store(event):
    ds: Dataset = event.dataset
    ds.file_meta = event.file_meta

    sop_uid = ds.SOPInstanceUID
    filename = f"{sop_uid}.dcm"
    out_path = os.path.join(STORAGE_DIR, filename)

    if os.path.exists(out_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(STORAGE_DIR, f"{base}-{i}{ext}")):
            i += 1
        out_path = os.path.join(STORAGE_DIR, f"{base}-{i}{ext}")

    ds.save_as(out_path, write_like_original=False)
    print(f"Stored DICOM: {out_path}")
    return 0x0000

def handle_echo(event):
    """C-ECHO 请求处理：直接返回 Success"""
    print("Received C-ECHO from", event.assoc.requestor.address)
    return 0x0000

def main():
    ae = AE(ae_title=AE_TITLE)

    # 支持 C-STORE
    for ctx in StoragePresentationContexts:
        ae.add_supported_context(ctx.abstract_syntax, ctx.transfer_syntax)

    # 支持 C-ECHO
    for ctx in VerificationPresentationContexts:
        ae.add_supported_context(ctx.abstract_syntax, ctx.transfer_syntax)

    handlers = [
        (evt.EVT_C_STORE, handle_store),
        (evt.EVT_C_ECHO, handle_echo),
    ]

    print(f"Starting DICOM SCP on {HOST}:{PORT}  AE_TITLE={AE_TITLE.decode()}")
    ae.start_server((HOST, PORT), evt_handlers=handlers, block=True)

if __name__ == '__main__':
    main()
