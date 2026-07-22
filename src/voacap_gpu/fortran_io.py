"""Readers for the gfortran binary layouts used by golden fixtures.

The oracle (gfortran, x86-64 WSL) writes unformatted sequential files as
little-endian records framed by 4-byte length markers:

    <u4 length> <payload bytes> <u4 length>

(Subrecord splitting only occurs past 2 GiB — never in this project.)
Direct-access files (``access='direct'``) have no markers at all: the file
is exactly ``recl`` bytes per record, back to back.
"""

import pathlib


def read_sequential_records(path) -> list[bytes]:
    """Return the payload of every record in a sequential unformatted file."""
    data = pathlib.Path(path).read_bytes()
    records = []
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError(f"{path}: truncated header at byte {pos}")
        n = int.from_bytes(data[pos:pos + 4], "little")
        end = pos + 4 + n
        if end + 4 > len(data):
            raise ValueError(f"{path}: truncated record at byte {pos}")
        trailer = int.from_bytes(data[end:end + 4], "little")
        if trailer != n:
            raise ValueError(
                f"{path}: marker mismatch at byte {pos}: {n} != {trailer}"
            )
        records.append(data[pos + 4:end])
        pos = end + 4
    return records


def read_direct_records(path, recl: int) -> list[bytes]:
    """Return the fixed-length records of a direct-access unformatted file."""
    data = pathlib.Path(path).read_bytes()
    if len(data) % recl:
        raise ValueError(f"{path}: size {len(data)} not a multiple of recl={recl}")
    return [data[i:i + recl] for i in range(0, len(data), recl)]
