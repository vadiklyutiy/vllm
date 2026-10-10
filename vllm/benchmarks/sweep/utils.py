# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
def sanitize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')


def escape_filename(filename: str) -> str:
    """Unlike `sanitize_filename`, distinct inputs never share an output."""
    return filename.replace("%", "%25").replace("/", "%2F")
