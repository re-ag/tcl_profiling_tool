import time
from typing import Callable

from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
import rosbag2_py


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def wait_for(
    condition: Callable[[], bool],
    timeout: Duration,
    sleep_time: float = 0.1,
):
    clock = Clock(clock_type=ClockType.STEADY_TIME)
    start = clock.now()
    while not condition():
        if clock.now() - start > timeout:
            return False
        time.sleep(sleep_time)
        return True