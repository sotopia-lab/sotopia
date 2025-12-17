from typing import AsyncIterator, Literal
from aact import Message, Node, NodeFactory
from aact.messages import Tick, Audio
import numpy as np


def merge_audio_streams(
    streams: list[bytes], sample_width: Literal[1, 2, 4] = 2
) -> bytes:
    # Convert byte streams to numpy arrays of audio samples
    format_str = {1: "B", 2: "h", 4: "i"}[sample_width]
    stream_samples = [
        np.frombuffer(stream, dtype=np.dtype(format_str)) for stream in streams
    ]

    # Make sure both streams are the same length
    stream_length = 0
    for stream in stream_samples:
        assert stream_length == 0 or len(stream) == stream_length
        if not stream_length:
            stream_length = len(stream)

    # Mix audio by adding the samples and avoiding clipping
    # mixed_samples = (stream1_samples.astype(np.int32) + stream2_samples.astype(np.int32)) // 2

    mixed_samples = np.zeros(stream_length, dtype=np.int32)
    for stream in stream_samples:
        mixed_samples += stream
    mixed_samples //= len(stream_samples)

    # Clip the values to ensure they remain within valid range for the bit depth
    mixed_samples = np.clip(
        mixed_samples, np.iinfo(format_str).min, np.iinfo(format_str).max
    )

    # Convert back to byte stream
    return mixed_samples.astype(np.dtype(format_str)).tobytes()


@NodeFactory.register("audio_mixer")
class AudioMixerNode(Node[Tick | Audio, Audio]):
    def __init__(
        self,
        input_channels: list[str],
        tick_input_channel: str,
        output_channel: str,
        redis_url: str,
        buffer_size: int = 1024,
    ):
        super().__init__(
            input_channel_types=[(channel, Audio) for channel in input_channels]
            + [(tick_input_channel, Tick)],
            output_channel_types=[(output_channel, Audio)],
            redis_url=redis_url,
        )
        self.input_channels = input_channels
        self.tick_input_channel = tick_input_channel
        self.output_channel = output_channel
        self.buffers: dict[str, bytes] = {channel: b"" for channel in input_channels}
        self.overflow_buffers: dict[str, bytes] = {
            channel: b"" for channel in input_channels
        }
        self.buffer_size = buffer_size

    async def event_handler(
        self, channel: str, message: Message[Tick | Audio]
    ) -> AsyncIterator[tuple[str, Message[Audio]]]:
        if channel == self.tick_input_channel:
            output_buffers = []

            for audio_channel in self.input_channels:
                output_buffers.append(
                    self.buffers[audio_channel]
                    + b"\x00" * (self.buffer_size - len(self.buffers[audio_channel]))
                )
                self.buffers[audio_channel] = self.overflow_buffers[audio_channel][
                    : self.buffer_size
                ]
                self.overflow_buffers[audio_channel] = self.overflow_buffers[
                    audio_channel
                ][self.buffer_size :]
            output_buffer = merge_audio_streams(output_buffers)
            yield self.output_channel, Message[Audio](data=Audio(audio=output_buffer))

        elif channel in self.input_channels:
            assert isinstance(message.data, Audio)
            if len(self.buffers[channel]) == self.buffer_size:
                self.overflow_buffers[channel] += message.data.audio
            else:
                self.buffers[channel] += message.data.audio
                if len(self.buffers[channel]) >= self.buffer_size:
                    self.overflow_buffers[channel] = self.buffers[channel][
                        self.buffer_size :
                    ]
                    self.buffers[channel] = self.buffers[channel][: self.buffer_size]
        else:
            raise ValueError(f"Unexpected channel: {channel}")
            yield (
                self.output_channel,
                Message(data=Audio(audio=b"")),
            )  # Unreachable code
