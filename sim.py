import matplotlib.pyplot as plt
import numpy as np


class GaussianNoiseModel:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev
    
    def noise(self):
        return np.random.normal(self.mean, self.std_dev)


class R3DecayModel:
    def decay(self, signal, dist):
        return signal / dist**3


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


class World:
    def __init__(self, wave_speed, background_noise_model, decay_model, events):
        self.wave_speed = wave_speed
        self.background_noise_model = background_noise_model
        self.decay_model = decay_model
        self.events = events
    
    def propagate(self, time, dist, event):
        # Returns the amplitude of an event at the given time and distance 
        # from it
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
            dist = np.linalg.norm(position - event.position)
            total_amp += self.propagate(time, dist, event)
        return total_amp


class SensorMeasurement:
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
        self.converters = converters

    def measure(self, time):
        # Returns a SensorMeasurement from this sensor at the given time
        amp = self.world.sample(time, self.position)
        noise = self.noise_model.noise()
        measurement = amp + noise
        for converter in self.converters:
            measurement = converter.convert(measurement)
        return SensorMeasurement(self.position, time, measurement, self)


class AnalogToDigitalConverter:
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
    def __init__(self, analog_min, analog_max, digital_resolution=1024):
        self.analog_min = analog_min
        self.analog_max = analog_max
        self.digital_resolution = digital_resolution
    
    def convert(self, value):
        # Returns the analog value of the given digitized value
        analog_value = value * (self.analog_max - self.analog_min) / self.digital_resolution + self.analog_min
        return analog_value


class TimeDataBuffer:
    def __init__(self, times, values, sample_rate=None, original_measurements=None):
        try:
            assert len(times) == len(values)
        except TypeError:
            times = np.arange(times, times + len(values) * sample_rate, sample_rate)
        
        self.times = times
        self.values = values

        self.sample_rate = sample_rate
        if sample_rate is None:
            sample_rate = times[1] - times[0]
        assert np.allclose(np.diff(times), sample_rate), "Times must be evenly spaced at the given sample rate"

        self.orig_meas = original_measurements
    
    def copy(self, times=None, values=None, sample_rate=None, orig_meas=None):
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
    def __init__(self, sensors, sample_rate, inter_buffer_time, buffer_size, start_time=0.0, start_sensor_ind=0):
        self.sensors = sensors
        self.sample_rate = sample_rate
        self.inter_buffer_time = inter_buffer_time

        self.buffer_size = buffer_size
        self.buffers = [[] for _ in sensors]
        
        self.start_time = start_time
        self.start_sensor_ind = start_sensor_ind

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
        self.buffers = [[] for _ in self.sensors]

        for _ in range(self.buffer_size * len(self.sensors)):
            self.buffers[self._sensor_ind].append(
                self.sensors[self._sensor_ind].measure(self._time)
            )
            self.increment_time()
            self.increment_sensor_ind()
        
        self.increment_time(self.inter_buffer_time)
        self.reset_sensor_ind()
    
    def get_effective_sample_rate(self):
        return self.sample_rate * len(self.sensors)
    
    def get_buffers(self):
        return [
            TimeDataBuffer.from_measurements(
                buffer, 
                self.get_effective_sample_rate(), 
            ) 
            for buffer in self.buffers
        ]


class SimpleFirstShotDetector:
    def __init__(self, basic_snr=1.5):
        self.basic_snr = basic_snr
        self.abs_buffer_values = []
        self.buffer_abs_argmaxes = []
        self.buffer_abs_maxes = []
        self.buffer_abs_medians = []
    
    def detect(self, buffers):
        self.abs_buffer_values = [
            np.abs(buffer.values) 
            for buffer in buffers
        ]
        self.buffer_abs_argmaxes = [
            np.argmax(abs_buffer_values) 
            for abs_buffer_values in self.abs_buffer_values
        ]
        self.buffer_abs_maxes = [
            abs_buffer_values[abs_argmax] 
            for abs_buffer_values, abs_argmax 
            in zip(self.abs_buffer_values, self.buffer_abs_argmaxes)
        ]
        self.buffer_abs_medians = [
            np.median(abs_buffer_values) 
            for abs_buffer_values in self.abs_buffer_values
        ]
        return all([
            abs_max > self.basic_snr * abs_median
            for abs_max, abs_median
            in zip(self.buffer_abs_maxes, self.buffer_abs_medians)
        ])


def normalize(data, largest_result=100.0, data_abs_max=None):

    if data_abs_max is None:
        abs_data = np.abs(data)
        data_abs_max = np.max(abs_data)
    
    data = data * (largest_result / data_abs_max)

    return data


class SimpleCorrelator:
    def __init__(self, sample_rate, shot_detector):
        self.sample_rate = sample_rate
        self.shot_detector = shot_detector

        self.buffers = []
        self.max_times = []
        self.first_buffer_ind = None

    def get_buffer_offsets(self, time_data_buffers):

        self.buffers = time_data_buffers
        
        if not self.shot_detector.detect(self.buffers):
            return None
        
        self.buffers = [
            buffer.copy(values=normalize(buffer.values, data_abs_max=buffer_abs_max))
            for buffer, buffer_abs_max in zip(self.buffers, self.shot_detector.buffer_abs_maxes)
        ]

        # Find the time of the max value in each buffer
        self.max_inds = self.shot_detector.buffer_abs_argmaxes
        self.max_times = np.array([
            buffer.times[max_ind] 
            for buffer, max_ind in zip(self.buffers, self.max_inds)
        ])

        # Find the buffer with the first maximum
        self.first_buffer_ind = np.argmin(self.max_times)
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
                tdoa_angle = np.arccos(time_offset * ref_sensor.world.wave_speed / sensor_dist)

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
        SimplestEvent(
            position=np.array([1.0, 10.0]), 
            start_time=1.0, 
            stop_time=1.5, 
            max_amp=1.0, 
            frequency=127.0, 
        ), 
    ]

    world = World(
        wave_speed=1114.0, 
        # background_noise_model=GaussianNoiseModel(0.0, 0.0), 
        background_noise_model=GaussianNoiseModel(0.0, 0.000001), 
        decay_model=R3DecayModel(), 
        events=events, 
    )

    sensor_noise = GaussianNoiseModel(0.0, 0.0)
    # sensor_noise = GaussianNoiseModel(0.0, 0.00000005)

    digital_resolution = 1024
    adc = AnalogToDigitalConverter(-0.001, 0.001, digital_resolution)
    dac = DigitalToAnalogConverter(-0.001, 0.001, digital_resolution)
    sensor_converters = [adc, dac]

    sensors = [
        Sensor(
            world=world, 
            position=np.array([1.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([3.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([5.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([7.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([9.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([11.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
        Sensor(
            world=world, 
            position=np.array([13.0, 0.0]), 
            noise_model=sensor_noise, 
            converters=sensor_converters, 
        ), 
    ]

    controller = SimpleSensorController(
        sensors=sensors, 
        sample_rate=1/5000.0, 
        inter_buffer_time=0.0, 
        buffer_size=10000, 
    )

    correlator = SimpleCorrelator(
        sample_rate=controller.sample_rate, 
        shot_detector=SimpleFirstShotDetector(), 
    )

    direction_finder = SimpleFarFieldDirectionFinder(controller)

    sim_time = 2.3

    while controller._time < sim_time:
        controller.update_buffers()
        offsets = correlator.get_buffer_offsets(controller.get_buffers())
        print("Time offsets (seconds): {}".format(offsets))

        if offsets is not None:
            angle, angle_std = direction_finder.find_direction(offsets, correlator.first_buffer_ind)
            print("Detected shot at {:.03f} deg (std: {:0.3g})".format(angle, angle_std))
        
        fig, ax = plt.subplots()
        plot_buffers(ax, correlator.buffers)
        plot_buffer_maxes(ax, correlator.max_times)
        plot_buffer_offsets(
            ax, 
            correlator.max_times[correlator.first_buffer_ind], 
            offsets, 
        )
        plt.show()


if __name__ == "__main__":
    main()
