import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

######################################################################
# Noise Models
#  - Implement a noise() method which returns a noise value

class GaussianNoiseModel:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev
    
    def noise(self):
        return np.random.normal(self.mean, self.std_dev)

######################################################################
# Decay Models
#  - Implement a decay(signal, dist) method which returns a value 
#       signal decayed over a distance of dist

class ZeroDecayModel:
    def decay(self, signal, dist):
        return signal


class R3DecayModel:
    def decay(self, signal, dist):
        return signal / dist**3

######################################################################
# Event Models
#  - Implement an amplitude(time) method which returns a signal 
#       amplitude value at the given time

class SimpleEvent:
    def __init__(self, position, start_time, stop_time, max_amp, frequency):
        self.position = position
        self.start_time = start_time
        self.stop_time = stop_time
        self.max_amp = max_amp
        self.frequency = frequency
    
    def amplitude(self, time):
        amp = 0.0

        if self.start_time <= time <= self.stop_time:
            evt_time = time - self.start_time
            # exponential term decays to close to zero by stop_time
            amp = np.sin(2.0 * np.pi * self.frequency * evt_time) * \
                self.max_amp * np.exp(-evt_time**2)
        
        return amp


class SimplestEvent:
    def __init__(self, position, start_time, stop_time, max_amp, frequency):
        self.position = position
        self.start_time = start_time
        self.stop_time = stop_time
        self.max_amp = max_amp
        self.frequency = frequency
    
    def amplitude(self, time):
        amp = 0.0

        if self.start_time <= time <= self.stop_time:
            evt_time = time - self.start_time
            duration = self.stop_time - self.start_time
            amp = self.max_amp * (1 - 2 * evt_time / duration)
        
        return amp


class WavFileEvent:
    def __init__(self, position, start_time, fname):
        self.position = position
        self.start_time = start_time
        self.fname = fname

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        self.sample_rate, self.data = wavfile.read(self.fname)
        self.duration = self.data.shape[0] / self.sample_rate
        self.stop_time = self.start_time + self.duration
    
    def amplitude(self, time):
        amp = 0.0

        if self.start_time <= time <= self.stop_time:
            evt_time = time - self.start_time
            amp = self.data[int(evt_time * self.sample_rate + 0.5)]
        
        return amp

######################################################################
# Converters
#  - Implement a convert(value) method which performs some conversion operation

class AnalogToDigitalConverter:
    # Simply scales a value from the given analog range to the given digital 
    # resolution. Ie. if the analog range is [-1, 1] and the digital 
    # resolution is 1024, then a value of 0.25 will be converted to 256.

    def __init__(self, analog_min, analog_max, digital_resolution=1024):
        self.analog_min = analog_min
        self.analog_max = analog_max
        self.digital_resolution = digital_resolution
    
    def convert(self, value):
        # Returns the digitized value of the given value
        digitized_value = (value - self.analog_min) / (self.analog_max - self.analog_min) * self.digital_resolution
        digitized_value = int(digitized_value)
        digitized_value = max(0, min(self.digital_resolution, digitized_value))
        return digitized_value


class DigitalToAnalogConverter:
    # Simply scales a value from the given digital resolution to the given 
    # analog range. Ie. if the analog range is [-1, 1] and the digital 
    # resolution is 1024, then a value of 256 will be converted to 0.25.
    
    def __init__(self, analog_min, analog_max, digital_resolution=1024):
        self.analog_min = analog_min
        self.analog_max = analog_max
        self.digital_resolution = digital_resolution
    
    def convert(self, value):
        # Returns the analog value of the given digitized value
        analog_value = value * (self.analog_max - self.analog_min) / self.digital_resolution + self.analog_min
        return analog_value

######################################################################

class World:
    # Represents the physical world in which events occur and are measured

    def __init__(self, wave_speed, background_noise_model, decay_model, events):
        self.wave_speed = wave_speed
        self.background_noise_model = background_noise_model
        self.decay_model = decay_model
        self.events = events
    
    def propagate(self, time, position, event):
        # Returns the amplitude of an event at the given time and location
        dist = np.linalg.norm(position - event.position)
        time_delay = dist / self.wave_speed
        amp = event.amplitude(time - time_delay)
        amp = self.decay_model.decay(amp, dist)
        noise = self.background_noise_model.noise()
        return amp + noise
    
    def sample(self, time, position):
        # Returns the total propagated amplitude of all events at the given 
        # time and position
        total_amp = 0.0
        for event in self.events:
            total_amp += self.propagate(time, position, event)
        return total_amp
    
    def get_wave_speed(self):
        return self.wave_speed


class SensorMeasurement:
    # Represents a single measurement from a sensor at a given time and 
    # position. Currently, the position is always the same as the sensor, but 
    # it's included separately in case I want to allow sensors to move over 
    # time.

    def __init__(self, position, time, value, sensor):
        self.position = position
        self.time = time
        self.value = value
        self.sensor = sensor
    
    def copy(self):
        return SensorMeasurement(self.position, self.time, self.value, self.sensor)


class Sensor:
    def __init__(self, world, position, noise_model, converters=[]):
        self.world = world
        self.position = position
        self.noise_model = noise_model

        # Converters are applied in order to each measurement as it is made, 
        # via the converters' convert() methods
        self.converters = converters

    def measure(self, time):
        # Returns a SensorMeasurement from this sensor at the given time

        amp = self.world.sample(time, self.position)
        noise = self.noise_model.noise()
        measurement = amp + noise
        for converter in self.converters:
            measurement = converter.convert(measurement)
        return SensorMeasurement(self.position, time, measurement, self)


class TimeDataBuffer:
    # Represents a buffer of time-domain data, ie. a list of time values and a 
    # corresponding list of data values. The time values must be evenly spaced.

    def __init__(self, times, values, sample_rate=None, original_measurements=None):
        # values is a numpy array.
        # 
        # If times is a single value, it is assumed to be the start time and 
        # the times are generated from it using the sample rate and the length 
        # of the values array. (So sample_rate must be given.)
        # 
        # Otherwise, times may be a numpy array of the same lenth as values. 
        # In this case, sample_rate is optional and if not given, it will be 
        # calculated from the times array. In either case, it must satisfy the 
        # np.allclose() function that the times differ by approx its value.
        # 
        # original_measurements is an optional list of SensorMeasurement 
        # objects which may be used to keep track of where the data came from.

        try:
            assert len(times) == len(values)
        except TypeError:
            times = np.arange(times, times + len(values) * sample_rate, sample_rate)
        
        self.times = times
        self.values = values

        # Cached absolute values of the data, computed on demand
        self.abs_values = None

        self.sample_rate = sample_rate
        if sample_rate is None:
            sample_rate = times[1] - times[0]
        
        assert np.allclose(np.diff(times), sample_rate), \
            "Times must be evenly spaced at the given sample rate"

        self.orig_meas = original_measurements
    
    def abs(self):
        # Returns the absolute values of the data values
        if self.abs_values is None:
            self.abs_values = np.abs(self.values)
        return self.abs_values
    
    def copy(self, times=None, values=None, sample_rate=None, orig_meas=None):
        # Creates a copy of this TimeDataBuffer, optionally with some 
        # parameters changed

        if times is None:
            times = self.times.copy()
        
        if values is None:
            values = self.values.copy()
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if orig_meas is None and self.orig_meas is not None:
            orig_meas = [meas.copy() for meas in self.orig_meas]
        
        return TimeDataBuffer(times, values, sample_rate, orig_meas)

    @classmethod
    def from_measurements(cls, measurements, sample_rate=None):
        # Creates a TimeDataBuffer from a list of SensorMeasurement objects

        times = np.array([measurement.time for measurement in measurements])
        values = np.array([measurement.value for measurement in measurements])
        return cls(times, values, sample_rate, measurements)
    
    def __getitem__(self, index):
        return SensorMeasurement(
            None if self.orig_meas is None else self.orig_meas[index].position, 
            self.times[index], 
            self.values[index], 
            None if self.orig_meas is None else self.orig_meas[index].sensor, 
        )
    
    def __len__(self):
        return len(self.times)



class SimpleSensorController:
    # Basic controller for a set of sensors. It coordinates measurements from 
    # the sensors, keeping track of the time and sensor index as it goes.

    def __init__(self, sensors, sample_rate, inter_buffer_time, buffer_collection_size, start_time=0.0, start_sensor_ind=0):

        # List of sensor objects
        self.sensors = sensors

        # Rate at which measurements are made by the controller
        self.sample_rate = sample_rate

        # Time between buffers (for simulating processing time)
        self.inter_buffer_time = inter_buffer_time

        # Number of measurements collected during each buffer update, spread 
        # evenly across all sensors (assuming it is evenly divisible by the 
        # number of sensors)
        self.buffer_collection_size = buffer_collection_size

        # One list of measurements for each sensor
        self.buffers = [[] for _ in sensors]
        
        # When to start collecting measurements
        self.start_time = start_time

        # Which sensor to start collecting measurements from
        self.start_sensor_ind = start_sensor_ind

        # Internal state variables
        self._time = start_time
        self._sensor_ind = start_sensor_ind
    
    def increment_time(self, inc_time=None):
        # Increments the time by inc_time, or by the sample rate if inc_time 
        # is None
        if inc_time is None:
            self._time += self.sample_rate
        else:
            self._time += inc_time
    
    def increment_sensor_ind(self, inc_sensor_ind=None):
        # Increments the sensor index by inc_sensor_ind, or by 1 if 
        # inc_sensor_ind is None
        if inc_sensor_ind is None:
            self._sensor_ind = int( (self._sensor_ind + 1) % len(self.sensors) )
        else:
            self._sensor_ind = int( (self._sensor_ind + inc_sensor_ind) % len(self.sensors) )
    
    def reset_sensor_ind(self):
        self._sensor_ind = self.start_sensor_ind

    def update_buffers(self):
        # Collects measurements from each sensor and stores them in the 
        # appropriate buffer. A total of buffer_collection_size measurements 
        # are collected, and internal _time and _sensor_ind variables are 
        # updated.

        self.buffers = [[] for _ in self.sensors]

        for _ in range(self.buffer_collection_size):
            self.buffers[self._sensor_ind].append(
                self.sensors[self._sensor_ind].measure(self._time)
            )
            self.increment_time()
            self.increment_sensor_ind()
        
        self.increment_time(self.inter_buffer_time)
        self.reset_sensor_ind()
    
    def get_effective_sample_rate(self):
        # Gets the sample rate of the individual buffers
        return self.sample_rate * len(self.sensors)
    
    def get_buffers(self):
        # Returns the buffers as a list of TimeDataBuffer objects
        return [
            TimeDataBuffer.from_measurements(
                buffer, 
                self.get_effective_sample_rate(), 
            ) 
            for buffer in self.buffers
        ]


class SlidingCFARController:

    def __init__(self, sensors, cfar_detectors, sample_rate, start_time=0.0, start_sensor_ind=0):

        # List of sensor objects and associated CFAR detectors
        self.sensors = sensors
        self.cfar_detectors = cfar_detectors
        assert len(sensors) == len(cfar_detectors)

        # Rate at which measurements are made by the controller
        self.sample_rate = sample_rate
        
        # When to start collecting measurements
        self.start_time = start_time

        # Which sensor to start collecting measurements from
        self.start_sensor_ind = start_sensor_ind

        # Internal state variables
        self._time = start_time
        self._sensor_ind = start_sensor_ind
    
    def increment_time(self, inc_time=None):
        # Increments the time by inc_time, or by the sample rate if inc_time 
        # is None
        if inc_time is None:
            self._time += self.sample_rate
        else:
            self._time += inc_time
    
    def increment_sensor_ind(self, inc_sensor_ind=None):
        # Increments the sensor index by inc_sensor_ind, or by 1 if 
        # inc_sensor_ind is None
        if inc_sensor_ind is None:
            self._sensor_ind = int( (self._sensor_ind + 1) % len(self.sensors) )
        else:
            self._sensor_ind = int( (self._sensor_ind + inc_sensor_ind) % len(self.sensors) )
    
    def reset_sensor_ind(self):
        self._sensor_ind = self.start_sensor_ind

    def update(self):
        measurement = self.sensors[self._sensor_ind].measure(self._time)
        detected = self.cfar_detectors[self._sensor_ind].update(abs(measurement.value))
        self.increment_time()
        self.increment_sensor_ind()
        return detected
    
    def get_effective_sample_rate(self):
        # Gets the sample rate of the individual buffers
        return self.sample_rate * len(self.sensors)
    
    def get_buffers(self):
        # Returns the buffers as a list of TimeDataBuffer objects
        sample_rate = self.get_effective_sample_rate()
        return [
            detector.get_buffer(
                self._time - sample_rate * detector.get_buffer_length(), 
                sample_rate, 
            ) 
            for detector in self.cfar_detectors
        ]


class SlidingCFARShotDetector:
    def __init__(self, num_train, num_guard, num_test, num_done, false_alarm_rate):
        # Newest cells -------------> Time --------------> Oldest cells
        # |--------------|----------|-----------|---------------------|
        # |<- num_done ->|<- test ->|<- guard ->|<- num_train cells ->|
        #                |------------------ Buffer ------------------|
        #                0 1 2 3 4 5 6 7 8 9 10 11... (buffer indices)
        # 
        # Train cells are used to estimate the noise level.
        # Guard cells are ignored to avoid biasing the noise estimate.
        # Reference cells are used to estimate the noise level.
        self.num_train = num_train
        self.num_guard = num_guard
        self.num_test = num_test

        # Number of sequential cells which must fall back under the threshold 
        # before we consider a peak to be done
        self.num_done = num_done
        self._done_counts = num_done  # Delay start of detection

        # Threshold factor
        self.false_alarm_rate = false_alarm_rate
        self._alpha = num_train * (false_alarm_rate**( -1.0/num_train ) - 1)

        self._buffer = np.zeros(num_train + num_guard + num_test)
        self._last_buffer_pos = len(self._buffer) - 1
        
        # _buffer_ind points to the newest cell in the buffer. We could slide 
        # the whole buffer, but for efficiency's sake (especially since this 
        # might be transfered to another language), we just keep track of the 
        # start index. Not going to worry about number of cells filled, since 
        # we'll just use the whole buffer all the time.
        self._buffer_ind = -1

        # These indices each point to the newest cell in their regions
        self._train_ind = num_train - 1
        self._guard_ind = num_train + num_guard - 1

        self._train_mean = 0.0
        self._guard_mean = 0.0
        self._test_mean = 0.0

        # # This is a dynamic list of pairs of indices (in lists) which track 
        # # the start and end of each peak as they are detected. These are 
        # # subtracted out of the train cells. As the end of the buffer 
        # # overwrites these ranges, they are removed from the list.
        # self._nulls = []

        self._detection_inds = []
    
    def _incr_index(self, index):
        if index >= self._last_buffer_pos:
            return 0
        return index + 1
    
    def get_buffer(self, start_time, sample_rate):
        buffer_end = self._incr_index(self._buffer_ind)
        return TimeDataBuffer(
            np.arange(start_time, start_time+sample_rate*len(self._buffer), sample_rate)[:len(self._buffer)], 
            np.concatenate((self._buffer[buffer_end:], self._buffer[:buffer_end])), 
            sample_rate, 
        )
    
    def get_buffer_length(self):
        return len(self._buffer)

    def update(self, value):
        # Note value should probably be passed as the absolute value

        last_ind = self._incr_index(self._buffer_ind)
        last_value = self._buffer[last_ind]

        self._buffer_ind = last_ind
        self._buffer[self._buffer_ind] = value

        if self._detection_inds and self._detection_inds[0] == self._buffer_ind:
            self._detection_inds.pop(0)

        # if self._nulls:
        #     if self._nulls[0][0] == last_ind:
        #         self._train_sum

        self._guard_ind = self._incr_index(self._guard_ind)
        guard_value = self._buffer[self._guard_ind]

        self._train_ind = self._incr_index(self._train_ind)
        train_value = self._buffer[self._train_ind]

        self._test_mean += (value - guard_value) / self.num_test
        self._guard_mean += (guard_value - train_value) / self.num_guard
        self._train_mean += (train_value - last_value) / self.num_train

        threshold = self._alpha * self._train_mean
        detected = False

        if self._done_counts > 0:
            if self._test_mean < threshold:
                self._done_counts -= 1
            else:
                self._done_counts = self.num_done

        elif self._test_mean > threshold:
            self._detection_inds.append(self._buffer_ind)
            detected = True
            self._done_counts = self.num_done
        
        return detected
    
    def get_detection_offsets(self):
        return np.array([
            (
                self._buffer_ind - detection_ind 
                if detection_ind <= self._buffer_ind 
                else self._buffer_ind - detection_ind + self._last_buffer_pos + 1
            )
            for detection_ind in self._detection_inds
        ])


class SimpleFirstShotDetector:

    def __init__(self, basic_snr=1.5):
        self.basic_snr = basic_snr
    
    def detect(self, buffers):
        # Returns a numpy array of indices into each buffer where a shot is 
        # detected, one for each buffer. If no shot is detected in a buffer, 
        # the corresponding index is None.

        # For each buffer, find the first index where the absolute value of
        # the signal is greater than the basic_snr times ...
        # TODO

        return np.array([
            # THIS IS A NONSENSE EQUATION FOR NOW
            int(27 * np.random.rand()) if np.random.rand() < 0.9 else np.nan
            for buffer 
            in buffers
        ])


class SimpleCorrelator:
    def __init__(self, sample_rate, shot_detector):
        self.sample_rate = sample_rate
        self.shot_detector = shot_detector

        self.buffers = []
        self.shot_times = []
        self.first_buffer_ind = None

    def get_buffer_offsets(self, time_data_buffers):

        self.buffers = time_data_buffers
        shot_inds = self.shot_detector.detect(self.buffers)

        if np.isnan(shot_inds).any():
            return None

        self.shot_times = np.array([
            buffer.times[shot_ind] 
            for buffer, shot_ind in zip(self.buffers, shot_inds)
        ])

        # Find the buffer with the first shot time
        self.first_buffer_ind = np.argmin(shot_inds)
        first_buffer = self.buffers[self.first_buffer_ind]

        # Find the offsets of the other buffers relative to the 'first' buffer 
        # by cross-correlating them
        offsets = np.zeros(len(self.buffers))
        for i, buffer in enumerate(self.buffers):
            if i != self.first_buffer_ind:    
                best_ind = np.correlate(buffer.values, first_buffer.values, mode="full").argmax()
                offsets[i] = (buffer.sample_rate / len(self.buffers)) * (best_ind - len(buffer))
        
        return offsets


class SimpleFarFieldDirectionFinder:
    def __init__(self, sensor_controller):
        self.sensor_controller = sensor_controller
    
    def find_direction(self, time_offsets, ref_ind):
        angles = np.zeros(len(time_offsets))
        ref_sensor = self.sensor_controller.sensors[ref_ind]
        for i, time_offset in enumerate(time_offsets):
            if i != ref_ind:
                sensor = self.sensor_controller.sensors[i]
                sensor_pos_diff = sensor.position - ref_sensor.position

                sensor_angle = np.arctan2(sensor_pos_diff[1], sensor_pos_diff[0])
                
                sensor_dist = np.linalg.norm(sensor_pos_diff)
                tdoa_angle = np.arccos(time_offset * ref_sensor.world.get_wave_speed() / sensor_dist)

                angles[i] = np.rad2deg(sensor_angle + tdoa_angle)
            else:
                angles[i] = np.nan
        
        angles += 360.0
        angles_mean = np.nanmean(angles)
        angles_std = np.nanstd(angles)
        if angles_mean > 180.0:
            angles_mean -= 360.0
        
        return angles_mean, angles_std


def plot_buffers(
            ax, buffers, 
            plot_args=("o-",), plot_kwargs={}, 
        ):
    handles = []
    for buffer in buffers:
        handle, = ax.plot(buffer.times, buffer.values, *plot_args, **plot_kwargs)
        handles.append(handle)
    return handles


def plot_buffer_maxes(
            ax, max_times, 
            axvline_args=(), axvline_kwargs={"color":"r"}, 
        ):
    
    handles = []
    for max_time in max_times:
        handle = ax.axvline(x=max_time, *axvline_args, **axvline_kwargs)
        handles.append(handle)
    return handles


def plot_buffer_offsets(
            ax, ref_time, time_offsets, 
            axvline_args=(), axvline_kwargs={"color":"g"}, 
        ):
    
    handles = []
    for time_offset in time_offsets:
        handle = ax.axvline(x=ref_time + time_offset, *axvline_args, **axvline_kwargs)
        handles.append(handle)
    return handles


def main():
    np.random.seed(0)

    events = [
        WavFileEvent(
            position=np.array([1.0, 1000.0]), 
            start_time=-1.0, 
            fname="three_shots_32pcm.wav", 
        ), 
    ]

    world = World(
        wave_speed=1114.0, 
        background_noise_model=GaussianNoiseModel(0.0, 0.0), 
        # background_noise_model=GaussianNoiseModel(0.0, 0.05), 
        # decay_model=R3DecayModel(), 
        decay_model=ZeroDecayModel(), 
        events=events, 
    )

    # sensor_noise = GaussianNoiseModel(0.0, 0.0)
    sensor_noise = GaussianNoiseModel(0.0, 0.03)

    digital_resolution = 1024
    adc = AnalogToDigitalConverter(-1.0, 1.0, digital_resolution)
    dac = DigitalToAnalogConverter(-1.0, 1.0, digital_resolution)
    sensor_converters = [adc, dac]

    sensors = [
        Sensor(
            world=world, 
            position=np.array([10.0, 10.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([1.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
    ]
    cfar_detectors = [
        SlidingCFARShotDetector(
            num_train=150, 
            num_guard=20, 
            num_test=3, 
            num_done=10, 
            false_alarm_rate=1e-3, 
        ), 
        SlidingCFARShotDetector(
            num_train=150, 
            num_guard=20, 
            num_test=3, 
            num_done=10, 
            false_alarm_rate=1e-3, 
        ), 
    ]

    # controller = SimpleSensorController(
    #     sensors=sensors, 
    #     sample_rate=1/5000, 
    #     inter_buffer_time=0.0, 
    #     buffer_collection_size=5000, 
    # )
    controller = SlidingCFARController(
        sensors=sensors, 
        cfar_detectors=cfar_detectors, 
        sample_rate=1/5000, 
    )

    # correlator = SimpleCorrelator(
    #     sample_rate=controller.sample_rate, 
    #     shot_detector=SimpleFirstShotDetector(), 
    # )

    direction_finder = SimpleFarFieldDirectionFinder(controller)

    sim_time = 13.0

    # step = 0
    # while controller._time < sim_time:
    #     controller.update_buffers()
    #     offsets = correlator.get_buffer_offsets(controller.get_buffers())
    #     print("[{}] Time offsets (seconds): {}".format(step, offsets))

    #     if offsets is not None:
    #         angle, angle_std = direction_finder.find_direction(offsets, correlator.first_buffer_ind)
    #         print("    Detected shot at {:.03f} deg (std: {:0.3g})".format(angle, angle_std))
        
    #     fig, ax = plt.subplots()
    #     plot_buffers(ax, correlator.buffers)
    #     if offsets is not None:
    #         plot_buffer_maxes(ax, correlator.shot_times)
    #         plot_buffer_offsets(
    #             ax, 
    #             correlator.shot_times[correlator.first_buffer_ind], 
    #             offsets, 
    #         )
    #     # plt.show()
    #     fig.savefig("sim_{:02d}.png".format(step))
    #     step += 1
    while controller._time < sim_time:
        detected = controller.update()

        if detected:
            print("Detected shot")
            fig, ax = plt.subplots()
            plot_buffers(ax, controller.get_buffers())
            plt.show()
    
    fig, ax = plt.subplots()
    plot_buffers(ax, controller.get_buffers())
    plt.show()


if __name__ == "__main__":
    main()
