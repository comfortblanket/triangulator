import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


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
    def __init__(self, analog_min, analog_max, digital_scale=1024):
        self.analog_min = analog_min
        self.analog_max = analog_max
        self.digital_scale = digital_scale
    
    def convert(self, value):
        # Returns the digitized value of the given value
        digitized_value = (value - self.analog_min) / (self.analog_max - self.analog_min) * self.digital_scale
        digitized_value = int(digitized_value)
        digitized_value = max(0, min(self.digital_scale, digitized_value))
        return digitized_value


class DigitalToAnalogConverter:
    def __init__(self, analog_min, analog_max, digital_scale=1024):
        self.analog_min = analog_min
        self.analog_max = analog_max
        self.digital_scale = digital_scale
    
    def convert(self, value):
        # Returns the analog value of the given digitized value
        analog_value = value * (self.analog_max - self.analog_min) / self.digital_scale + self.analog_min
        return analog_value


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


class SimpleCorrelator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

        self.clean_buffer_times = []
        self.clean_buffers = []
        self.buffer_maxes = []
        self.first_buffer_ind = None

    def get_buffer_offsets(self, buffers, k=51):

        self.clean_buffers = [
            normalize(np.array([meas.value for meas in buffer])) 
            for buffer in buffers
        ]

        self.clean_buffer_times, self.clean_buffers = upsample_signals(
            [ [meas.time for meas in buffer] for buffer in buffers ], 
            [ np.array([meas.value for meas in buffer]) for buffer in buffers ], 
        )
        self.clean_buffers = [
            normalize(buffer)
            for buffer in self.clean_buffers
        ]

        # abs_buffers = np.array([
        #     np.abs(buffer) 
        #     for buffer in self.clean_buffers
        # ])

        # Find position of the largest absolute sample in each buffer
        # k_largest_inds = np.array([
        #     np.argpartition(buffer, -k)[-k:]
        #     for buffer in abs_buffers
        # ])
        # k_largest = np.array([
        #     buffer[k_largest_inds[i]] 
        #     for i, buffer in enumerate(abs_buffers)
        # ])
        # self.buffer_maxes = np.array([
        #     np.average(largest_inds, weights=largest/np.sum(largest)) 
        #     for largest, largest_inds in zip(k_largest, k_largest_inds)
        # ], dtype=int)
        self.buffer_maxes = np.array([
            np.argmax( np.abs(buffer) ) 
            for buffer in self.clean_buffers
        ])

        # Find the buffer with the first maximum
        self.first_buffer_ind = np.argmin(self.buffer_maxes)
        first_buffer = self.clean_buffers[self.first_buffer_ind]

        # Find the offsets of the other buffers relative to the first buffer 
        # by cross-correlating them
        offsets = np.zeros(len(self.clean_buffers))
        for i, buffer in enumerate(self.clean_buffers):
            if i != self.first_buffer_ind:    
                correlation = np.correlate(buffer, first_buffer, mode="full").argmax()
                offsets[i] = (self.sample_rate / len(self.clean_buffers)) * (correlation - len(buffer))
        
        return offsets


def upsample_simultaneous_signals(signals):
    # signals contains a list of signals which were created by sampling first 
    # the first one, then the second one, etc, and then repeating. This 
    # function slides each signal so that they are all aligned at the first 
    # value, upsamples each by the number of signals in the list, and then 
    # removes the extra values at the ends of each signal.

    new_signals = []
    for i, signal in enumerate(signals):
        # resampled_signal = scipy.signal.resample(signal, len(signal) * len(signals))#, window="hamming")
        
        N = len(signal)
        interp = len(signals)
        resampled_signal = scipy.fft.ifft(N*scipy.fft.fft(signal), interp*N)

        # resampled_signal = resampled_signal[i:i+len(resampled_signal)-len(signals)+1:]
        new_signals.append(resampled_signal)
    return new_signals


def upsample_signals(times, signals):
    resampled_times = []
    for time_buffer in times:
        resampled_times += time_buffer
    resampled_times = sorted( resampled_times )[len(times)-1:]

    resampled_buffers = upsample_simultaneous_signals(signals)

    return resampled_times, resampled_buffers


def reconstruct_signals(signals):
    reconstructed_signals = [
        scipy.signal
    ]

    return reconstructed_signals


def plot_clean_buffers(buffer_times, buffers, maxes=None, offsets=None, first_offset_ind=None):
    if maxes is None:
        for times, buffer in zip(buffer_times, buffers):
            plt.plot(times, buffer, "o-")

    elif offsets is None:
        for buffer, mx in zip(buffers, maxes):
            plt.plot(buffer_times, buffer, "o-")
            # plt.axvline(x=buffer_times[mx].time, color="r")
        
    else:
        ref_buf_time = buffer_times[maxes[first_offset_ind]]
        i = 0
        for buffer, mx, offset in zip(buffers, maxes, offsets):
            plt.plot(buffer_times, buffer, "o-", label=f"Sensor {i}")
            # plt.axvline(x=buffer_times[mx], color="orange", lw=3)
            plt.axvline(x=ref_buf_time + offset, color="g", lw=1)
            i += 1
    
        plt.legend()


def normalize(data):
    
    return data - np.mean(data)


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
        background_noise_model=GaussianNoiseModel(0.0, 0.0), 
        # background_noise_model=GaussianNoiseModel(0.0, 0.000001), 
        decay_model=R3DecayModel(), 
        events=events, 
    )

    sensor_noise = GaussianNoiseModel(0.0, 0.0)#0.00000001)

    adc = AnalogToDigitalConverter(-0.001, 0.001, 16)
    dac = DigitalToAnalogConverter(-0.001, 0.001, 128)
    sensor_converters = [adc]#, dac]

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
        sample_rate=controller.sample_rate*len(sensors), 
    )

    sim_time = 2.3

    while controller._time < sim_time:
        controller.update_buffers()
        offsets = correlator.get_buffer_offsets(controller.buffers)
        print(offsets)
        
        # _times = []
        # for buffer in controller.buffers:
        #     _times += [meas.time for meas in buffer]
        # resampled_times = sorted( _times )[len(sensors)-1:]
        # resampled_buffers = upsample_simultaneous_signals(correlator.clean_buffers)
        # for resampled_buffer in resampled_buffers:
        #     plt.plot(resampled_times, resampled_buffer, "-")
        # for buffer, clean_buffer in zip(controller.buffers, correlator.clean_buffers):
        #     plt.plot([meas.time for meas in buffer], clean_buffer, "o")
        # plt.show()
        # plot_clean_buffers(
        #     [
        #         [meas.time for meas in buffer]
        #         for buffer in controller.buffers
        #     ], 
        #     [
        #         [meas.value for meas in buffer]
        #         for buffer in controller.buffers
        #     ], 
        # )
        plot_clean_buffers(
            correlator.clean_buffer_times, 
            correlator.clean_buffers, 
            correlator.buffer_maxes, 
            offsets, 
            correlator.first_buffer_ind, 
        )
        plt.show()


if __name__ == "__main__":
    main()
