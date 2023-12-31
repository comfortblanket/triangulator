# Allows to set up a scenario with two or more sensors, and then given times 
# of arrival (TOA) at each of them (ie. when a signal is detected at each 
# one), estimates possible locations for the source of the signal.
# 
# In particular, this is intended for estimating the source of a loud sound 
# which was caught on timestamped audio recordings from multiple sensors. 


import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Classes

class World:

    def __init__(self, sensor_locs, wave_speed):
        self.sensor_locs = sensor_locs  # list of (x, y) pairs
        self.wave_speed = wave_speed
    
    def copy(self):
        return World(self.sensor_locs.copy(), self.wave_speed)


class Scenario:

    def __init__(self, world, toas):
        assert len(world.sensor_locs) == len(toas)

        self.world = world
        self.toas = toas


    def copy(self, deep=False):
        if deep:
            return Scenario(self.world.copy(), self.toas.copy())
        return Scenario(self.world, self.toas.copy())


    def solve(
                self, 
                search_x,   # list of regularly-spaced x-values to search over
                search_y,   # list of regularly-spaced y-values to search over
                print_timing=5, 
            ):

        if print_timing is not False:
            start_time = datetime.datetime.now()
            last_time = start_time
            print("Start time: {}".format(start_time))

        # Sensor with the smallest TOA is the reference sensor
        ref_sensor_idx = self.toas.index(min(self.toas))

        loc_map = np.zeros(
            shape=( len(search_y), len(search_x) ), 
            dtype=np.float64, 
        )

        for i,x in enumerate(search_x):
            for j,y in enumerate(search_y):

                if print_timing is not False:
                    cur_time = datetime.datetime.now()
                    if cur_time - last_time > datetime.timedelta(seconds=print_timing):
                        print("({:0.2f}%) [{},{} / {},{}] Current time: {} ({})".format(
                            100.0*(j + i*len(search_y))/(len(search_x)*len(search_y)), 
                            i, j, len(search_x), len(search_y), 
                            cur_time, cur_time - start_time, 
                        ))
                        last_time = cur_time

                test_loc = np.array([x, y])
                est_dist_to_ref = np.linalg.norm(test_loc - self.world.sensor_locs[ref_sensor_idx])

                # Simple negative squared-difference
                # TODO: Make this bayesian
                loc_map[j, i] = sum(
                    (
                        (                       # Estimated time of arrival
                            np.linalg.norm(test_loc - self.world.sensor_locs[k]) - est_dist_to_ref
                        ) / self.world.wave_speed 
                        - (                     # True time of arrival
                            self.toas[k] - self.toas[ref_sensor_idx]
                        )
                    )**2 
                    for k in range(len(self.world.sensor_locs))
                )
        
        return Solution(self.world, self, loc_map, search_x, search_y)


class Solution:

    def __init__(self, world, scenario, solution_map, search_x, search_y):
        self.world = world
        self.scenario = scenario
        self.solution_map = solution_map
        self.search_x = search_x
        self.search_y = search_y

        self._ref_sensor_idx = None
        self._best_loc = None
        self._best_dist = None
        self._best_event_time = None
    

    def ref_sensor_idx(self):
        # Sensor with the smallest TOA is the reference sensor
        if self._ref_sensor_idx is None:
            self._ref_sensor_idx = self.scenario.toas.index(min(self.scenario.toas))
        return self._ref_sensor_idx


    def best_loc(self):
        if self._best_loc is None:
            best_loc = self.solution_map.argmin()     # Index
            self._best_loc = np.array([               # Convert index to (x, y) pair
                self.search_x[best_loc % self.solution_map.shape[1]],
                self.search_y[best_loc // self.solution_map.shape[1]], 
            ])
        return self._best_loc
    
    def best_dist(self):
        if self._best_dist is None:
            self._best_dist = np.linalg.norm(
                self.best_loc() - self.world.sensor_locs[ self.ref_sensor_idx() ]
            )
        return self._best_dist

    def best_event_time(self):
        if self._best_event_time is None:
            self._best_event_time = self.scenario.toas[ self.ref_sensor_idx() ] - self.best_dist() / self.world.wave_speed
        return self._best_event_time


class MapImage:

    def __init__(self, img, a_img, b_img, a_ax=None, b_ax=None):
        self.img = img
        self.a_img = a_img
        self.b_img = b_img
        self.a_ax = a_ax
        self.b_ax = b_ax

    def draw(self, ax, a_ax=None, b_ax=None, **kwargs):
        if a_ax is None:
            a_ax = self.a_ax
        
        if b_ax is None:
            b_ax = self.b_ax

        return draw_map(
            ax,         self.img, 
            self.a_img, self.b_img, 
            a_ax,       b_ax, 
            **kwargs, 
        )

################################################################################
# Functions

def print_solution_info(solution):
    
    best_time_to_ref = solution.best_dist() / solution.world.wave_speed

    print("Best location estimate: {}".format(solution.best_loc()))
    print("Estimated event time: {}".format(solution.best_event_time()))

    for i in range(len(solution.world.sensor_locs)):

        # Time from best-scoring location to this sensor
        time_to_sensor = np.linalg.norm(solution.best_loc() - solution.world.sensor_locs[i]) / solution.world.wave_speed

        # Difference in detection time between reference sensor and this 
        # sensor, assuming the event happened at the best-scoring location
        time_difference_to_sensor = time_to_sensor - best_time_to_ref

        print("[Sensor {}] TOA: {} (Truth: {}), TDOA: {} (Truth: {})".format(
            i, 
            solution.best_event_time() + time_to_sensor, 
            solution.scenario.toas[i], 
            time_difference_to_sensor, 
            solution.scenario.toas[i] - solution.scenario.toas[solution.ref_sensor_idx()], 
        ))


def draw_map(ax, img, a_img, b_img, a_ax, b_ax, **kwargs):
    # Draws an image on a matplotlib axis, with the image extent set such that 
    # the image is scaled correctly. Locations a_ and b_ are two points of 
    # correspondence between the image and the axis coordinate system, 
    # preferably very far apart from each other in both horizontal and 
    # vertical directions, such as the top-left and bottom-right corners of 
    # the image. Each location is given as a pair of coordinates (x, y).

    scale_x = (b_ax[0] - a_ax[0]) / (b_img[0] - a_img[0])
    scale_y = (b_ax[1] - a_ax[1]) / (b_img[1] - a_img[1])

    extent_left   = a_ax[0] - scale_x * a_img[0]
    extent_right  = b_ax[0] + scale_x * (img.shape[1] - b_img[0])
    extent_bottom = a_ax[1] - scale_y * a_img[1]
    extent_top    = b_ax[1] + scale_y * (img.shape[0] - b_img[1])

    return ax.imshow(
        img, 
        extent=[extent_left, extent_right, extent_bottom, extent_top], 
        **kwargs, 
    )


def plot_solution(
            ax, 
            solution, 
            map_images=[],          # list of MapImage objects to draw
            solution_cmap=None,     # colormap to use for plotting solution
            toa_circles={},         # TOA circles centered at sensors: a dict of Circle() kwargs, or False to disable
            toa_circles_best={},    # TOA circles centered at best solution: a dict of Circle() kwargs, or False to disable
            plot_sensors={},        # Sensor points: a dict of plot() kwargs, or False to disable
            annotate_sensors={},    # Sensor annotations: a dict of annotate() kwargs, or False to disable
            plot_best={},           # Best solution point: a dict of plot() kwargs, or False to disable
        ):

    for map_img in map_images:
        map_img.draw(ax)

    # We assume search_x and search_y lists are regularly-spaced
    search_x_diff = solution.search_x[1] - solution.search_x[0]
    search_y_diff = solution.search_y[1] - solution.search_y[0]

    ax.imshow(
        solution.solution_map, 
        extent=[
            solution.search_x.min() - 0.5*search_x_diff, 
            solution.search_x.max() + 0.5*search_x_diff, 
            solution.search_y.min() - 0.5*search_y_diff, 
            solution.search_y.max() + 0.5*search_y_diff, 
        ], 
        interpolation="nearest", 
        cmap=solution_cmap, 
        origin="lower", 
        norm="log", 
    )

    # Plot TOA circles
    if toa_circles is not False:
        circle_dict = {
            "fill" : False, 
            "color" : "k", 
        }
        circle_dict.update(toa_circles)
        
        for i, (loc, t) in enumerate(zip(solution.world.sensor_locs, solution.scenario.toas)):
            ax.add_artist(
                plt.Circle( loc, (t-solution.best_event_time())*solution.world.wave_speed, **circle_dict)
            )

    # Plot TOA circles from estimated location
    if toa_circles_best is not False:
        circle_dict = {
            "fill" : False, 
            "color" : "r", 
        }
        circle_dict.update(toa_circles_best)
        
        for i, (loc, t) in enumerate(zip(solution.world.sensor_locs, solution.scenario.toas)):
            ax.add_artist(
                plt.Circle( solution.best_loc(), (t-solution.best_event_time())*solution.world.wave_speed, **circle_dict)
            )

    # Plot sensor points
    if plot_sensors is not False:
        sensor_dict = {
            "marker" : "s", 
            "markeredgecolor" : "k", 
        }
        sensor_dict.update(plot_sensors)

        for loc in solution.world.sensor_locs:
            ax.plot(loc[0], loc[1], **sensor_dict)

    # Annotate sensors
    if annotate_sensors is not False:
        for i, loc in enumerate(solution.world.sensor_locs):
            ax.annotate(str(i), loc, **annotate_sensors)

    # Plot estimated location
    if plot_best is not False:
        best_dict = {
            "marker" : "x", 
            "color" : "k", 
        }
        best_dict.update(plot_best)
        plt.plot(solution.best_loc()[0], solution.best_loc()[1], **best_dict)


def main(
            ax, 
            scenario, 
            search_x,               # list of regularly-spaced x values to search over
            search_y,               # list of regularly-spaced y values to search over
            map_images=[],          # list of MapImage objects to draw
            solution_cmap=None,     # colormap to use for plotting solution
            toa_circles={},         # TOA circles centered at sensors: a dict of Circle() kwargs, or False to disable
            toa_circles_best={},    # TOA circles centered at best solution: a dict of Circle() kwargs, or False to disable
            plot_sensors={},        # Sensor points: a dict of plot() kwargs, or False to disable
            annotate_sensors={},    # Sensor annotations: a dict of annotate() kwargs, or False to disable
            plot_best={},           # Best solution point: a dict of plot() kwargs, or False to disable
            print_results=True, 
            print_timing=5,         # Print timing information every N seconds, or False to disable
        ):

    # Solve the scenario
    solution = scenario.solve(
        search_x, 
        search_y, 
        print_timing=print_timing, 
    )
    
    if print_results:
        print_solution_info(solution)

    # Plot results
    plot_solution(ax, solution, map_images, solution_cmap, toa_circles, toa_circles_best, plot_sensors, annotate_sensors, plot_best)

################################################################################
################################################################################

if __name__ == "__main__":

    spd_sound = 1114.0  # ft/sec

    # (x, y) of ENU locations in feet
    sensor_locs = [
        np.array([0.0, 0.0]), 
        np.array([-85.0, 2630.0]), 
    ]

    # Timestamps in seconds (most important is that the differences between 
    # them are correct - absolute values don't matter)
    toas = [
        16.5,  # 22:22:16.5
        14.3,  # 22:22:14.3
    ]
    
    world = World(sensor_locs, wave_speed=spd_sound)
    scenario = Scenario(world, toas)

    search_x = np.arange(-1000, 900, 5.0)
    search_y = np.arange(2200, 3500, 5.0)
    
    map_images = [
        # MapImage(
        #     plt.imread("map_without_points.png"), 
        #     np.array([462.0, 902-715.0]), 
        #     np.array([446.0, 902-272.0]), 
        #     world.sensor_locs[0], 
        #     world.sensor_locs[1], 
        # ), 
    ]

    cmap_dict = {
        "red" :   ( (0.0, 1.0, 1.0), (1.0, 1.0, 1.0) ),  # Constant 1.0
        "green" : ( (0.0, 1.0, 1.0), (1.0, 1.0, 1.0) ),  # Constant 1.0
        "blue" : (  (0.0, 0.0, 0.0), (1.0, 0.0, 0.0) ),  # Constant 0.0
        "alpha" : ( (0.0, 1.0, 1.0), (1.0, 0.0, 0.0) ),  # Linear from 1.0 to 0.0
    }
    yellow_alpha_cmap = matplotlib.colors.LinearSegmentedColormap("YellowAlpha", cmap_dict)

    for toa0_offset in [0]:  #[-0.1, 0.0, 0.1]:
        for toa1_offset in [0]:  #[-0.1, 0.0, 0.1]:
            offset_scenario = scenario.copy()
            offset_scenario.toas[0] += toa0_offset
            offset_scenario.toas[1] += toa1_offset
            
            fig, ax = plt.subplots()

            main(
                ax, 
                offset_scenario, 
                search_x, 
                search_y, 
                map_images=map_images, 
                solution_cmap=yellow_alpha_cmap, 
                print_timing=5, 
            )
            
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim([-1000, 900])
            ax.set_ylim([2200, 3500])
            ax.set_title("TOA[0]={:.01f}, TOA[1]={:.01f}".format(offset_scenario.toas[0], offset_scenario.toas[1]))

            plt.show()
            # plt.savefig("triangulation_1_{:.01f}-{:.01f}.png".format(offset_scenario.toas[0], offset_scenario.toas[1]), dpi=300)

            plt.close("all")
