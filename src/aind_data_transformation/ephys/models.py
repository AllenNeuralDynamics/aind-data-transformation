from enum import Enum
from typing import Literal

from numcodecs import Blosc
from pydantic import Field
from pydantic_settings import BaseSettings


class CompressorName(str, Enum):
    """Enum for compression algorithms a user can select"""

    BLOSC = Blosc.codec_id
    WAVPACK = "wavpack"


class RecordingBlockPrefixes(Enum):
    """Enum for types of recording blocks."""

    neuropix = "Neuropix"
    nidaq = "NI-DAQ"


class EcephysCompressionParameters(BaseSettings):
    """Compression parameters for ephys job."""

    # Clip settings
    clip_n_frames: int = Field(
        default=100,
        description="Number of frames to clip the data.",
        title="Clip N Frames",
    )
    # Compress settings
    compress_write_output_format: Literal["zarr"] = Field(
        default="zarr",
        description=(
            "Output format for compression. Currently, only zarr supported."
        ),
        title="Write Output Format",
    )
    compress_max_windows_filename_len: int = Field(
        default=150,
        description=(
            "Windows OS max filename length is 256. The zarr write will "
            "raise an error if it detects that the destination directory has "
            "a long name."
        ),
        title="Compress Max Windows Filename Len",
    )
    compressor_name: CompressorName = Field(
        default=CompressorName.WAVPACK,
        description="Type of compressor to use.",
        title="Compressor Name.",
    )
    compressor_kwargs: dict = Field(
        default={"level": 3},
        description="Arguments to be used for the compressor.",
        title="Compressor Kwargs",
    )
    compress_job_save_kwargs: dict = Field(
        default={"n_jobs": -1},  # -1 to use all available cpu cores.
        description="Arguments for recording save method.",
        title="Compress Job Save Kwargs",
    )
    compress_chunk_duration: str = Field(
        default="1s",
        description="Duration to be used for chunks.",
        title="Compress Chunk Duration",
    )

    # Scale settings
    scale_num_chunks_per_segment: int = Field(
        default=100,
        description="Num of chunks per segment to scale.",
        title="Scale Num Chunks Per Segment",
    )
    scale_chunk_size: int = Field(
        default=10000,
        description="Chunk size to scale.",
        title="Scale Chunk Size",
    )
