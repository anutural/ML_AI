import numpy as np
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
list_of_loc = list(range(m)) # list of locations


class CabDriver():
    def __init__(self):
        """initialise your state and define your action space and state space"""
        # Initialize environment
        self.initialize_env()


    ## Initializing the environment
    def initialize_env(self):
        self.action_space = [(pickup, drop) for pickup in range(m) for drop in range(m) if pickup != drop or pickup == 0]
        self.state_space = [(pickup, hr_of_day, day_of_week) for pickup in range(m) for hr_of_day in range(t) for day_of_week in range(d)]
        self.state_init = self.state_space[np.random.randint(0, len(self.state_space))] # Pick a random state
        self.state_size = m + t + d
        self.action_size = len(self.action_space)
        self.time_elapsed = 0


    ## Encoding state (or state-action) for NN input
    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format.
        Hint: The vector is of size m + t + d."""
        crnt_loc, hr_of_day, day_of_week = state
        # Converting the state into one-hot encoded vector
        state_enc = np.zeros(self.state_size, dtype=int)
        state_enc[crnt_loc] = 1
        state_enc[m + hr_of_day] = 1
        state_enc[m + t + day_of_week] = 1
        return state_enc


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        lambda_distribution = [2, 12, 4, 7, 8]
        lambda_for_loc = lambda_distribution[list_of_loc.index(location)]
        requests = np.random.poisson(lambda_for_loc)
        if requests > 15:
            requests = 15
        possible_actions_index = random.sample(range(1, ((m - 1) * m) + 1), requests)
        # (0, 0) is not considered as customer request
        possible_actions = [self.action_space[i] for i in possible_actions_index]
        possible_actions.append((0, 0))
        return possible_actions


    ## Calculate reward, and time taken for one trip
    def calc_reward_next_state(self, state, action, time_matrix, is_reward):
        """Takes in state, action and Time-matrix and returns the reward"""
        crnt_loc, hr_of_day, day_of_week = state
        if (action == (0,0)): # Driver is offline
            reward = -C
            time_taken = 1
            next_hr = hr_of_day + time_taken
            if next_hr >= t:
                day_of_week += 1
            next_state = (crnt_loc, next_hr % t, day_of_week % d)
        else:
            # t1 = time required to pickup the passenger from the current location of the driver
            # t2 = time required to drop the passenger from the pickup location
            pickup, drop = action
            inc_day_of_week = False
            if crnt_loc == pickup: # If current location is same as pickup location, t1 is 0
                t1 = 0
            else:
                t1 = time_matrix[crnt_loc, pickup, hr_of_day, day_of_week]
            
            pickup_time = hr_of_day + t1
            # Calculate the reward, drop time, and the time taken to drop
            # If driver accepts the request at 11PM and picks up the passenger at 12AM at the pickup location, it is treated as next day.
            if pickup_time >= t:
                day_of_week += 1
                day_of_week = day_of_week % d
                inc_day_of_week = True
            hr_of_day = pickup_time % t
            t2 = time_matrix[pickup, drop, hr_of_day, day_of_week]
            reward = (R * t2) - C * (t1 + t2)
            drop_time = pickup_time + t2
            time_taken = t1 + t2
            # Update the day of the week in case if the driver picks up the passenger on current day and drops the next day.
            # If already incremented in previous step, its not required to do it again.
            if drop_time >= t and not inc_day_of_week:
                day_of_week += 1
            next_state = (drop, drop_time % t, day_of_week % d)
        if is_reward:
            return reward
        else:
            return next_state, time_taken


    ## Get reward for completing an action
    def get_reward(self, state, action, time_matrix):
        return self.calc_reward_next_state(state, action, time_matrix, True)


    ## Get state transition after taking an action
    def get_next_state(self, state, action, time_matrix):
        """Takes state and action as input and returns next state and the terminal state"""
        next_state, time_taken = self.calc_reward_next_state(state, action, time_matrix, False)
        self.time_elapsed += time_taken
        episode_time = 24 * 30 # 24 hours in a day for 30 days is one episode
        is_terminal = self.time_elapsed >= episode_time
        return next_state, is_terminal


    ## Reset the environment
    def reset(self):
        self.initialize_env()
        return self.action_space, self.state_space, self.state_init
