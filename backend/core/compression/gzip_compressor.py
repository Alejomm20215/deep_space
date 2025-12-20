from __future__ import annotations

import gzip
import os
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompressionResult:
    input_path: str
    output_path: str
    method: str
    input_bytes: int
    output_bytes: int


def gzip_file(input_path: str, output_path: Optional[str] = None, compresslevel: int = 9) -> CompressionResult:
    """
    Gzip a file (useful for shipping large ASCII PLY outputs).
    """
    if output_path is None:
        output_path = input_path + ".gz"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=compresslevel) as f_out:
        shutil.copyfileobj(f_in, f_out)

    in_b = int(os.path.getsize(input_path))
    out_b = int(os.path.getsize(output_path))

    return CompressionResult(
        input_path=input_path,
        output_path=output_path,
        method="gzip",
        input_bytes=in_b,
        output_bytes=out_b,
    )

