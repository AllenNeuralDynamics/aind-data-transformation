import logging
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import spikeinterface.preprocessing as spre
from numcodecs import Blosc
from numpy import memmap
from spikeinterface import extractors as se
from wavpack_numcodecs import WavPack

from aind_data_transformation.core import TransformationJob
from aind_data_transformation.ephys.models import (
    CompressorName,
    RecordingBlockPrefixes,
)
from aind_data_transformation.ephys.npopto_correction import (
    correct_np_opto_electrode_locations,
)
from aind_data_transformation.models import (
    JobResponse,
    TransformationJobConfig,
)


class EphysCompressionJob(TransformationJob):

    _READER_NAME = "openephys"

    def __int__(self, job_definition: TransformationJobConfig):
        super().__init__(job_definition=job_definition)

    def _get_read_blocks(self) -> Iterator:
        """
        Given a reader name and input directory, generate a stream of
        recording blocks.

        Returns:
            A generator of read blocks. A read_block is dict of
            {'recording', 'experiment_name', 'stream_name'}.

        """
        nblocks = se.get_neo_num_blocks(
            self._READER_NAME, self.job_configs.input_source
        )
        stream_names, stream_ids = se.get_neo_streams(
            self._READER_NAME, self.job_configs.input_source
        )
        # load first stream to map block_indices to experiment_names
        rec_test = se.read_openephys(
            self.job_configs.input_source,
            block_index=0,
            stream_name=stream_names[0],
        )
        record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
        experiments = rec_test.neo_reader.folder_structure[record_node][
            "experiments"
        ]
        exp_ids = list(experiments.keys())
        experiment_names = [
            experiments[exp_id]["name"] for exp_id in sorted(exp_ids)
        ]
        for block_index in range(nblocks):
            for stream_name in stream_names:
                rec = se.read_openephys(
                    self.job_configs.input_source,
                    stream_name=stream_name,
                    block_index=block_index,
                    load_sync_timestamps=True,
                )
                yield (
                    {
                        "recording": rec,
                        "experiment_name": experiment_names[block_index],
                        "stream_name": stream_name,
                    }
                )

    def _get_streams_to_clip(self) -> Iterator:
        stream_names, stream_ids = se.get_neo_streams(
            self._READER_NAME, self.job_configs.input_source
        )
        for dat_file in self.job_configs.input_source.glob("**/*.dat"):
            oe_stream_name = dat_file.parent.name
            si_stream_name = [
                stream_name
                for stream_name in stream_names
                if oe_stream_name in stream_name
            ][0]
            n_chan = se.read_openephys(
                self.job_configs.input_source,
                block_index=0,
                stream_name=si_stream_name,
            ).get_num_channels()
            data = np.memmap(
                str(dat_file), dtype="int16", order="C", mode="r"
            ).reshape(-1, n_chan)
            yield {
                "data": data,
                "relative_path_name": str(
                    dat_file.relative_to(self.job_configs.input_source)
                ),
                "n_chan": n_chan,
            }

    @staticmethod
    def _get_compressor(compressor_name, **kwargs):
        """
        Retrieve a compressor for a given name and optional kwargs.
        Args:
            compressor_name (str): Matches one of the names Compressors enum
            **kwargs (dict): Options to pass into the Compressor
        Returns:
            An instantiated compressor class.
        """
        if compressor_name == CompressorName.BLOSC.value:
            return Blosc(**kwargs)
        elif compressor_name == CompressorName.WAVPACK.value:
            return WavPack(**kwargs)
        else:
            raise Exception(
                f"Unknown compressor. Please select one of "
                f"{[c for c in CompressorName]}"
            )

    @staticmethod
    def _scale_read_blocks(
        read_blocks,
        num_chunks_per_segment=100,
        chunk_size=10000,
    ):
        """
        Scales a read_block. A read_block is dict of
        {'recording', 'block_index', 'stream_name'}.
        Args:
            read_blocks (iterable): A generator of read_blocks
            num_chunks_per_segment (int):
            chunk_size (int):
        Returns:
            A generated scaled_read_block. A dict of
            {'scaled_recording', 'block_index', 'stream_name'}.

        """
        for read_block in read_blocks:
            # We don't need to scale the NI-DAQ recordings
            # TODO: Convert this to regex matching?
            if RecordingBlockPrefixes.nidaq.value in read_block["stream_name"]:
                rec_to_compress = read_block["recording"]
            else:
                rec_to_compress = spre.correct_lsb(
                    read_block["recording"],
                    num_chunks_per_segment=num_chunks_per_segment,
                    chunk_size=chunk_size,
                )
            yield (
                {
                    "scaled_recording": rec_to_compress,
                    "experiment_name": read_block["experiment_name"],
                    "stream_name": read_block["stream_name"],
                }
            )

    def _copy_and_clip_data(
        self,
        dst_dir: Path,
        stream_gen: Iterator[dict],
    ):
        """
        Copies the raw data to a new directory with the .dat files clipped to
        just a small number of frames. This allows someone to still use the
        spikeinterface api on the clipped data set.
        Parameters
        ----------
        dst_dir : Path
          Desired location for clipped data set
        stream_gen : Iterator[dict]
          An Iterator where each item is a dictionary with shape,
            'data': memmap(dat file),
              'relative_path_name': path name of raw data
                to new dir correctly
              'n_chan': number of channels.
        Returns
        -------
        None
          Moves some directories around.

        """

        # first: copy everything except .dat files
        patterns_to_ignore = ["*.dat"]
        shutil.copytree(
            self.job_configs.source,
            dst_dir,
            ignore=shutil.ignore_patterns(*patterns_to_ignore),
        )
        # second: copy clipped dat files
        for stream in stream_gen:
            data = stream["data"]
            rel_path_name = stream["relative_path_name"]
            n_chan = stream["n_chan"]
            dst_raw_file = dst_dir / rel_path_name
            dst_data = memmap(
                dst_raw_file,
                dtype="int16",
                shape=(self.job_configs.clip_n_frames, n_chan),
                order="C",
                mode="w+",
            )
            dst_data[:] = data[: self.job_configs.clip_n_frames]

    @staticmethod
    def _compress_and_write_block(
        read_blocks: Iterator[dict],
        compressor,
        output_dir,
        job_kwargs: dict,
        max_windows_filename_len: int,
        output_format: str = "zarr",
    ):
        """
        Compress and write read_blocks.
        Args:
            read_blocks (iterable dict):
              Either [{'recording', 'block_index', 'stream_name'}] or
              [{'scale_recording', 'block_index', 'stream_name'}].
            compressor (obj): A compressor class
            output_dir (Path): Output directory to write compressed data
            job_kwargs (dict): Recording save job kwargs.
            output_format (str): Defaults to zarr
            max_windows_filename_len (int): Warn if base file names are larger
              than this.

        Returns:
            Nothing. Writes data to a folder.
        """
        if job_kwargs["n_jobs"] == -1:
            job_kwargs["n_jobs"] = os.cpu_count()

        for read_block in read_blocks:
            if "recording" in read_block:
                rec = read_block["recording"]
            else:
                rec = read_block["scaled_recording"]
            experiment_name = read_block["experiment_name"]
            stream_name = read_block["stream_name"]
            zarr_path = output_dir / f"{experiment_name}_{stream_name}.zarr"
            if (
                platform.system() == "Windows"
                and len(str(zarr_path)) > max_windows_filename_len
            ):
                raise Exception(
                    f"File name for zarr path is too long "
                    f"({len(str(zarr_path))})"
                    f" and might lead to errors. Use a shorter destination "
                    f"path."
                )
            _ = rec.save(
                format=output_format,
                zarr_path=zarr_path,
                compressor=compressor,
                **job_kwargs,
            )

    def _compress_raw_data(self) -> None:
        """If compress data is set to False, the data will be uploaded to s3.
        Otherwise, it will be compressed to zarr, stored in temp_dir, and
        uploaded later."""

        # Correct NP-opto electrode positions:
        # correction is skipped if Neuropix-PXI version > 0.4.0
        # It'd be nice if the original data wasn't modified.
        correct_np_opto_electrode_locations(self.job_configs.source)
        # Clip the data
        logging.info("Clipping source data. This may take a minute.")
        if self.job_configs.number_id is None:
            clipped_data_path = (
                self.job_configs.output_directory / "ecephys_clipped"
            )
        else:
            clipped_data_path = (
                self.job_configs.output_directory
                / f"ecephys_clipped{self.job_configs.number_id}"
            )
        streams_to_clip = self._get_streams_to_clip()
        self._copy_and_clip_data(
            dst_dir=clipped_data_path,
            stream_gen=streams_to_clip,
        )

        logging.info("Finished clipping source data.")

        # Compress the data
        logging.info("Compressing source data.")
        if self.job_configs.number_id is None:
            compressed_data_path = (
                self.job_configs.output_directory / "ecephys_compressed"
            )
        else:
            compressed_data_path = (
                self.job_configs.output_directory
                / f"ecephys_compressed{self.job_configs.number_id}"
            )
        read_blocks = self._get_read_blocks()
        compressor = self._get_compressor(
            self.job_configs.compressor_name.value,
            **self.job_configs.compressor_kwargs,
        )
        scaled_read_blocks = self._scale_read_blocks(
            read_blocks=read_blocks,
            num_chunks_per_segment=(
                self.job_configs.scale_num_chunks_per_segment
            ),
            chunk_size=self.job_configs.scale_chunk_size,
        )
        self._compress_and_write_block(
            read_blocks=scaled_read_blocks,
            compressor=compressor,
            max_windows_filename_len=(
                self.job_configs.compress_max_windows_filename_len
            ),
            output_dir=compressed_data_path,
            output_format=self.job_configs.compress_write_output_format,
            job_kwargs=self.job_configs.compress_job_save_kwargs,
        )
        logging.info("Finished compressing source data.")

        return None

    def run_job(self) -> JobResponse:
        job_start_time = datetime.now()
        self._compress_raw_data()
        job_end_time = datetime.now()
        return JobResponse(
            message=f"Job finished in: {job_end_time-job_start_time}"
        )
