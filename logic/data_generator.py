# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:40:22 2022

@author: yshapira
"""
import os
import subprocess
import random
import numpy as np
import math
import json
import csv
from copy import deepcopy
from datetime import datetime
import pickle

from configuration.conf import BASE_PATH, TIME_COEFF, COST_COEFF, MP_MAP_PATH, MP_SCRIPT_PATH


def read_from_pickle(path):
    if not path.endswith('.pkl'): raise Exception("Sorry, this is a not pickle file.")
    with open(path, 'rb') as file:
        res = pickle.load(file)
    return res


class Node():
    def __init__(self, node_id, node_type, service_time, x, y):
        self.id = node_id
        self.type = node_type
        self.service_time  = service_time
        self.coordinates = np.array([x, y])
        self.departure_time = None

    def set_departure_time(self, time):
        self.departure_time = time

    def get_euclidean_distance(self, another_node):
        # set min dist to 2
        return max(np.linalg.norm(self.coordinates - another_node.coordinates),2)


class Map():
    
    time_coeff = TIME_COEFF
    cost_coeff = COST_COEFF
    
    def __init__(self, file_name, pt_coeff, map_size, 
                 num_robots, robots_max_travel_time, 
                 num_depots, depot_capacity, depot_st,
                 num_requests, request_st, outsourcing_cost,
                 request_half_time_window, request_max_time,
                 num_service_lines, num_transfers, transfer_capacity, transfer_st
                 ):
        # if the available_depot_capacity is less then num_robots the problem input isn't feasible, 
        # need to fix the depot_capacity
        available_depot_capacity = depot_capacity * num_depots
        if available_depot_capacity < num_robots: depot_capacity = math.ceil(num_robots/num_depots)
        # save all the inputs in the below varibals
        self.file_name_arc_based = f'{file_name}_arc_based'
        self.file_name_path_based = f'{file_name}_path_based'
        self.dat_full_path_arc_based = os.path.join(BASE_PATH, self.file_name_arc_based + ".dat")
        self.dat_full_path_path_based = os.path.join(BASE_PATH, self.file_name_path_based + ".dat")
        self.map_pkl_full_path = os.path.join(BASE_PATH, f'{file_name}_map.pkl')
        self.duals_full_path = os.path.join(BASE_PATH, "duals.json")
        self.paths_data_full_path = os.path.join(BASE_PATH, file_name + "_paths_data.csv")
        self.log_file = os.path.join(BASE_PATH, file_name + "_log.txt")
        self.pt_coeff = pt_coeff
        self.map_size = map_size
        self.length_of_program  = map_size * self.time_coeff * 10
        self.num_robots = num_robots
        self.robots_max_travel_time = robots_max_travel_time * self.time_coeff
        self.num_depots = num_depots
        self.depot_capacity = depot_capacity
        self.depot_st = depot_st
        self.num_requests = num_requests
        self.request_st = request_st
        self.outsourcing_cost = outsourcing_cost
        self.request_half_time_window = request_half_time_window
        self.request_max_time = request_max_time * self.time_coeff
        self.num_service_lines = num_service_lines
        self.num_transfers = num_transfers
        self.transfer_capacity = transfer_capacity
        self.transfer_st = transfer_st
        # range for the possible coordinates on the map
        self.coordinates_list = range(1, map_size + 1)
        # another range for the limited coordinates on the map for transfer nodes
        self.limited_coordinates_list = range(1, max(math.ceil(map_size/10),2))
        # ranges for all the items in the model
        self.robots = range(1, num_robots + 1)
        self.service_lines = range(1, num_service_lines + 1)
        self.all_nodes = range(1, 2 * num_requests + num_transfers + num_depots + 1)
        self.request_nodes = range(1, 2 * num_requests + 1)
        self.pickup_nodes = range(1, num_requests + 1)
        self.drop_off_nodes = range(num_requests + 1, 2 * num_requests + 1)
        self.num_nodes = len(self.all_nodes)
        # list of pairs (pickup,dropoff) of each request
        self.request_pairs = list(map(tuple, zip(self.pickup_nodes, self.drop_off_nodes)))
        self.transfer_nodes = range(2 * num_requests + 1, 2 * num_requests + num_transfers + 1)
        self.depot_nodes = range(2 * num_requests + num_transfers + 1, 2 * num_requests + num_transfers + num_depots + 1)
        # dict of robots origin locations {robot1: "depot_node_id1", robot1: "depot_node_id1"...}
        self.robots_origin_location = dict.fromkeys(self.robots, None)
        # dict with keys of all depot nodes and the values are the robots in this depot
        self.origin_location_dict = dict([(i,set()) for i in self.depot_nodes]) 
        # dict with keys of all depot nodes and the values are the number robots in this depot in the beginning of the program
        self.depots_initial_assignment_dict = dict.fromkeys(self.depot_nodes, None)
        # dict for all nodes info - {node1: Class Node1, node2: Class Node2, ..}
        self.all_nodes_dict = dict.fromkeys(self.all_nodes, None)
        # dict for request nodes time window - {node1: (start,end), node2: (start,end), ..}
        self.request_nodes_dict = dict.fromkeys(self.request_nodes, None)
        # dict for transfer nodes departure time - {node1: departure time, node2: departure time, ..}
        self.transfer_nodes_dict = dict.fromkeys(self.transfer_nodes, None)
        # dict to map transfer nodes by service line - {sl1: [list of tr nodes], sl2: {list of tr nodes}, ..}
        self.service_lines_dict = dict([(i,[]) for i in self.service_lines])  
        # list of mapping of each transfer node with the associated service line
        self.transfer_to_line = dict.fromkeys(self.transfer_nodes, None)
        # dict of tuples, each tuple is arc - (time,cost,is_pt_arc)
        self.arcs_dict = {} 
        # dict of tuples, each tuple is arc - (time,cost,is_pt_arc) when the cost is with the dual cost
        self.arcs_dict_with_dual_cost = {} 
        # dict of dict of tuples, each tuple is set of nodes of this pt arc
        self.pt_arcs_dict = {}
        # index for all paths
        self.paths_index = 0
        # list of tuples of all paths, each tuple is  - {path type, path_id, time, cost, cost_with_dual, robot, list_of_nodes_in_path}
        self.all_paths_list = []
        # set for (start_depot, pickup) that all related paths already added to self.all_paths_list
        self.start_depot_pickup_black_set = set()
        # set for (robot, pickup) that all related paths already added to self.all_paths_list
        self.robot_pickup_black_set = set()
        # dicts with list of paths for each node type
        self.robots_paths_dict = dict([(i,set()) for i in self.robots]) 
        self.request_paths_dict = dict([(i,set()) for i in self.pickup_nodes]) 
        self.transfer_paths_dict = dict([(i,set()) for i in self.transfer_nodes])
        self.start_paths_dict = dict([(i,set()) for i in self.depot_nodes])
        self.end_paths_dict = dict([(i,set()) for i in self.depot_nodes])
        # dictionaries for the dual values
        self.duals_dict = {}
        self.robot_duals_dict = dict.fromkeys(self.robots, 0)
        self.pickup_duals_dict = dict.fromkeys(self.pickup_nodes, 0)
        self.transfer_duals_dict = dict.fromkeys(self.transfer_nodes, 0)
        self.depot_duals_dict = dict.fromkeys(self.depot_nodes, 0)
        # boolean variable to flag if we have the optimal solution for the path based model
        self.path_based_is_optimal_solution = True
        
    def save_map(self, path):
        if not path.endswith('.pkl'): raise Exception("Sorry, this is not a pickle file.") 
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        
    def generate_robot_origin_locations(self):
        depot_capacity_dict = dict.fromkeys(self.depot_nodes, self.depot_capacity)
        depot_capacity_list = list(self.depot_nodes)
        for robot in self.robots_origin_location:
            chosen_depot = np.random.choice(depot_capacity_list)
            self.robots_origin_location[robot] = chosen_depot
            self.origin_location_dict[chosen_depot].add(robot)
            depot_capacity_dict[chosen_depot] -= 1
            if depot_capacity_dict[chosen_depot] < 1: depot_capacity_list.remove(chosen_depot)
        for depot in self.depots_initial_assignment_dict:
            self.depots_initial_assignment_dict[depot] = len(self.origin_location_dict[depot])
    
    def generate_service_lines(self):
        avg_nodes_in_line = int(math.ceil(self.num_transfers/self.num_service_lines))
        transfer_nodes_stack = list(reversed(self.transfer_nodes))
        for line in self.service_lines:
            temp_node_list = []
            for node in range(avg_nodes_in_line):
                if not transfer_nodes_stack: break
                t_node = transfer_nodes_stack.pop()
                temp_node_list.append(t_node)
                # add the associated service line of each transfer node to transfer_to_line mapping
                self.transfer_to_line[t_node] = line 
            self.service_lines_dict[line] = temp_node_list 
            
    def generate_depot_nodes(self):
        for depot in self.depot_nodes:
            depot_node_id = depot
            depot_node_x = random.choice(self.coordinates_list)
            depot_node_y = random.choice(self.coordinates_list)
            depot_node_temp = Node(depot_node_id, 'depot', self.depot_st, depot_node_x, depot_node_y)
            depot_node_temp.set_departure_time(0)
            self.all_nodes_dict[depot] = depot_node_temp
            
    def generate_request_nodes(self):
        for pickup in self.pickup_nodes:
            pickup_node_id = pickup
            pickup_node_x = random.choice(self.coordinates_list)
            pickup_node_y = random.choice(self.coordinates_list)
            pickup_node_temp = Node(pickup_node_id, 'pickup', self.request_st, pickup_node_x, pickup_node_y)
            # set pickup node departure time in the first 1/4 of the program length
            pickup_node_departure_time = random.choice(range(math.ceil((self.length_of_program)/10), math.ceil((self.length_of_program)/4)))
            pickup_node_temp.set_departure_time(pickup_node_departure_time)
            self.all_nodes_dict[pickup] = pickup_node_temp
            pickup_start_time = max(0, pickup_node_departure_time-self.request_half_time_window)
            pickup_end_time = min(self.length_of_program ,pickup_node_departure_time+self.request_half_time_window)
            self.request_nodes_dict[pickup] = (pickup_start_time,pickup_end_time)
            # generating drop_off node info with respect to the associated pickup_node
            drop_off = pickup + self.num_requests
            drop_off_node_id  = drop_off
            drop_off_node_x = random.choice(self.coordinates_list)
            drop_off_node_y = random.choice(self.coordinates_list) 
            drop_off_node_temp = Node(drop_off_node_id, 'drop_off', self.request_st, drop_off_node_x, drop_off_node_y)
            # set drop_off node departure time with appropriate distance from pickup node departure time
            pickup_drop_off_dist = pickup_node_temp.get_euclidean_distance(drop_off_node_temp)
            minimum_time_dist = math.ceil(pickup_node_departure_time + self.request_st + pickup_drop_off_dist * self.time_coeff)
            maximum_time_dist = math.ceil((self.length_of_program)*0.9)
            # check of we need to deliver the package in shorter time
            max_time_of_request = math.ceil(pickup_node_departure_time + self.request_st + self.request_max_time)
            if maximum_time_dist > max_time_of_request: maximum_time_dist = max_time_of_request
            # check if the maximum_time_dist >= minimum_time_dist, if not, need to make it feasible
            if maximum_time_dist <= minimum_time_dist: maximum_time_dist = minimum_time_dist + 1
            drop_off_node_departure_time = random.choice(range(minimum_time_dist, maximum_time_dist))
            drop_off_node_temp.set_departure_time(drop_off_node_departure_time)
            self.all_nodes_dict[drop_off] = drop_off_node_temp
            drop_off_start_time = max(0, drop_off_node_departure_time-self.request_half_time_window)
            drop_off_end_time = min(self.length_of_program ,drop_off_node_departure_time+self.request_half_time_window)
            self.request_nodes_dict[drop_off] = (drop_off_start_time,drop_off_end_time)
            
    def generate_transfer_nodes(self):
        # set minimum_line_length of service line
        minimum_line_length = math.ceil(self.map_size * 0.60)
        for line in self.service_lines:
            if len(self.service_lines_dict[line]) == 0: continue
            # setting first_t_node of line
            first_t_node_id = self.service_lines_dict[line][0]
            fixed_coordinate = random.choice(['x','y'])
            if fixed_coordinate == 'x':
                first_t_node_x = random.choice(self.limited_coordinates_list)
                first_t_node_y = random.choice(self.coordinates_list)
            else:
                first_t_node_x = random.choice(self.coordinates_list)
                first_t_node_y = random.choice(self.limited_coordinates_list)
            first_t_node_temp = Node(first_t_node_id, 'transfer', self.transfer_st, first_t_node_x, first_t_node_y)
            self.all_nodes_dict[first_t_node_id] = first_t_node_temp
            # setting last_t_node of line with respect to minimun distance from first_t_node
            last_t_node_id = self.service_lines_dict[line][-1]
            last_t_node_x = random.choice(self.coordinates_list)
            last_t_node_y = random.choice(self.coordinates_list)
            last_t_node_temp = Node(last_t_node_id, 'transfer', self.transfer_st, last_t_node_x, last_t_node_y)
            fisrt_and_last_dist = first_t_node_temp.get_euclidean_distance(last_t_node_temp)
            # if fisrt_and_last_dist is less then minimum_line_length, need to sample last_t_node coordinates again
            while fisrt_and_last_dist < minimum_line_length:
                last_t_node_x = random.choice(self.coordinates_list)
                last_t_node_y = random.choice(self.coordinates_list)
                last_t_node_temp = Node(last_t_node_id, 'transfer', self.transfer_st, last_t_node_x, last_t_node_y)
                fisrt_and_last_dist = first_t_node_temp.get_euclidean_distance(last_t_node_temp)
            self.all_nodes_dict[last_t_node_id] = last_t_node_temp
            # setting the rest of the nodes in the line
            # check if there are any other nodes in this line, if not, continue to next line
            # calculate the y = m*x + b of the line
            x1, y1 = first_t_node_temp.coordinates
            x2, y2 = last_t_node_temp.coordinates
            m = (y1-y2)/(x1-x2 + np.finfo(np.float32).eps)
            b = (x1*y2 - x2*y1)/(x1-x2 + np.finfo(np.float32).eps)
            if len(self.service_lines_dict[line]) > 1:
                x_dist_between_trensfers = math.floor(abs(x1-x2)/(len(self.service_lines_dict[line])-1))
                y_dist_between_trensfers = math.floor(abs(y1-y2)/(len(self.service_lines_dict[line])-1))
            else:
                x_dist_between_trensfers = 0
                y_dist_between_trensfers = 0
            if x1 > x2: x_dist_between_trensfers *= -1
            if y1 > y2: y_dist_between_trensfers *= -1
            previous_x, previous_y = x1, y1
            for transfer in self.service_lines_dict[line][1:-1]:
                transfer_node_id = transfer
                if x1 == x2:
                    transfer_node_x = previous_x
                    transfer_node_y = int(previous_y + y_dist_between_trensfers)
                else:
                    transfer_node_x = int(previous_x + x_dist_between_trensfers)
                    transfer_node_y = int(m * transfer_node_x + b)
                transfer_node_temp = Node(transfer_node_id, 'transfer', self.transfer_st, transfer_node_x, transfer_node_y)
                self.all_nodes_dict[transfer] = transfer_node_temp
                previous_x, previous_y = transfer_node_x, transfer_node_y          
            # calculate the total travel time of the line in order to set departture times
            previous_t_node = self.all_nodes_dict[first_t_node_id]
            total_travel_time = self.transfer_st 
            for transfer in self.service_lines_dict[line][1:]:
                if previous_t_node == last_t_node_id: break
                current_t_node = self.all_nodes_dict[transfer]
                total_travel_time += (previous_t_node.get_euclidean_distance(current_t_node) + self.transfer_st) * self.time_coeff
                previous_t_node = current_t_node
            # set departure time of the first node in the line
            latest_departure_time_of_first_t_node = math.ceil((self.length_of_program) * 0.9 - total_travel_time)
            first_node_departure_time = random.choice(range(0, latest_departure_time_of_first_t_node))
            self.all_nodes_dict[first_t_node_id].set_departure_time(first_node_departure_time)
            self.transfer_nodes_dict[first_t_node_id] = first_node_departure_time
            # set departure time for the rest of the nodes in the line
            previous_t_node = self.all_nodes_dict[first_t_node_id]
            for transfer in self.service_lines_dict[line][1:]:
                current_t_node = self.all_nodes_dict[transfer]
                travel_time = (previous_t_node.get_euclidean_distance(current_t_node)) * self.time_coeff * max(self.pt_coeff, 0.1)
                current_node_departure_time = math.ceil(previous_t_node.departure_time + travel_time + self.transfer_st)
                self.all_nodes_dict[transfer].set_departure_time(current_node_departure_time)
                self.transfer_nodes_dict[transfer] = current_node_departure_time
                previous_t_node = current_t_node
                
    def arc_feasibility_check(self, origin_node_id, dest_node_id):
        origin_node = self.all_nodes_dict[origin_node_id]
        dest_node = self.all_nodes_dict[dest_node_id]
        dist = origin_node.get_euclidean_distance(dest_node)     
        time = dist * self.time_coeff
        cost = dist * self.cost_coeff
        # check if it is public transportation arc from the same service line
        is_pt_arc = False
        if origin_node.type == 'transfer' and dest_node.type == 'transfer':
            # check if these 2 transfer nodes in the same line
            if self.transfer_to_line[origin_node_id] == self.transfer_to_line[dest_node_id]:
                is_pt_arc = True
        if is_pt_arc:
            # add to arc time the service times of the transfer nodes between origin_node and dest_node (dest node excluded)
            time = (time + ((dest_node_id - origin_node_id -1) * self.transfer_st)) * max(self.pt_coeff, 0.1)           
            cost *= self.pt_coeff
            # reducing cost of directed pt arcs
            cost *= (100.0 - (dest_node_id - origin_node_id)+ 1)/100.0
            set_of_transfer_nodes = set()
            for transfer_node in range(origin_node_id, dest_node_id):
                set_of_transfer_nodes.add(transfer_node) 
            self.pt_arcs_dict[(origin_node_id,dest_node_id)] = set_of_transfer_nodes
            
        # boolean variable to flag if the arc is feasibile
        is_feasible_arc = True
        # no need to create self arc
        if origin_node_id == dest_node_id:
            is_feasible_arc = False
        # robots_max_travel_time check
        if time > self.robots_max_travel_time and not is_pt_arc:
            is_feasible_arc = False
        # from trnasfer node to pickup/dropoff node
        if origin_node.type == 'transfer' and dest_node.type in ('pickup','drop_off') and origin_node.departure_time + time + dest_node.service_time > self.request_nodes_dict[dest_node_id][1]:
            is_feasible_arc = False
        # from pickup/dropoff node to trnasfer node
        elif origin_node.type in ('pickup','drop_off') and dest_node.type == 'transfer' and self.request_nodes_dict[origin_node_id][0] + time + dest_node.service_time > dest_node.departure_time:
            is_feasible_arc = False 
        # from pickup node to dropoff node
        elif origin_node.type == 'pickup' and dest_node.type == 'drop_off' and (self.request_nodes_dict[origin_node_id][0] + time + dest_node.service_time > self.request_nodes_dict[dest_node_id][1] or origin_node_id + self.num_requests != dest_node_id):
            is_feasible_arc = False
        # from dropoff node to pickup/drop_off node
        elif origin_node.type == 'drop_off' and dest_node.type in ('pickup', 'drop_off'):
            is_feasible_arc = False
        # from pickup node to pickup/depot node
        elif origin_node.type == 'pickup' and dest_node.type in ('pickup', 'depot'):
            is_feasible_arc = False
        # from depot node to pickup node
        elif origin_node.type == 'depot' and dest_node.type == 'pickup' and 0 + time + dest_node.service_time > self.request_nodes_dict[dest_node_id][1]:
            is_feasible_arc = False
        # from depot node to dropoff node
        elif origin_node.type == 'depot' and dest_node.type == 'drop_off':
            is_feasible_arc = False
        # from depot node to trnasfer node
        elif origin_node.type == 'depot' and dest_node.type == 'transfer' and 0 + time + dest_node.service_time > dest_node.departure_time:
            is_feasible_arc = False
        # from trnasfer node to trnasfer node from a diffrent service line
        elif origin_node.type == 'transfer' and dest_node.type == 'transfer' and self.transfer_to_line[origin_node_id] != self.transfer_to_line[dest_node_id]:
            is_feasible_arc = False
        # from  trnasfer node to trnasfer node from the same service line but in the wrong direction
        elif origin_node.type == 'transfer' and dest_node.type == 'transfer' and (origin_node_id > dest_node_id or origin_node.departure_time + time + dest_node.service_time > dest_node.departure_time):
            is_feasible_arc = False
        return is_feasible_arc, time, cost, is_pt_arc
    
    def generate_arcs(self):
        for origin_node_id in self.all_nodes:
            for dest_node_id in self.all_nodes:
                is_feasible_arc, time, cost, is_pt_arc = self.arc_feasibility_check(origin_node_id, dest_node_id)
                if is_feasible_arc:
                    self.arcs_dict[(origin_node_id,dest_node_id)] = (time, cost, is_pt_arc)

                    
    def path_feasibility_check(self, full_path):
        is_feasible_path = True
        # check if all time constraints are met (request time window and trnasfer nodes departure time)
        departure_time = 0
        battery_left = self.robots_max_travel_time
        for index, current_node_id in enumerate(full_path[:-1]):
            current_node = self.all_nodes_dict[current_node_id]
            next_node_id = full_path[index+1]
            next_node = self.all_nodes_dict[next_node_id]     
            is_feasible_arc, arc_time, arc_cost, is_pt_arc = self.arcs_dict[current_node_id,next_node_id,:][1:]
            
            if not is_pt_arc: battery_left -= arc_time
            if battery_left < 0: 
                is_feasible_path = False
            earliest_time = departure_time + arc_time + next_node.service_time
            if current_node.type == 'depot' and next_node.type == 'transfer':
                if earliest_time > next_node.departure_time:
                    is_feasible_path = False
                    break
                else:
                    departure_time = next_node.departure_time
            elif current_node.type in ['depot', 'transfer'] and next_node.type in ['pickup', 'drop_off']:
                if earliest_time > self.request_nodes_dict[next_node_id][1]:
                    is_feasible_path = False
                    break
                else:
                    departure_time = max(earliest_time, self.request_nodes_dict[next_node_id][0] + next_node.service_time)
            elif current_node.type in ['pickup', 'drop_off', 'transfer'] and next_node.type == 'transfer':
                if earliest_time > next_node.departure_time:
                    is_feasible_path = False
                    #self.log_str +=  f'not feasible - pickup drop_off transfer - transfer, {current_node_id}, {next_node_id}, {full_path}, \n'
                    break
                else:
                    departure_time = next_node.departure_time 
            elif current_node.type in ['pickup', 'drop_off'] and next_node.type in ['pickup', 'drop_off']:
                if earliest_time > self.request_nodes_dict[next_node_id][1]:
                    is_feasible_path = False
                    break
                else:
                    departure_time = max(earliest_time, self.request_nodes_dict[next_node_id][0] + next_node.service_time)
        return is_feasible_path

    def get_shortest_path_with_constriants(self, robot, pickup):
        start_depot = self.robots_origin_location[robot]
        dropoff = pickup + self.num_requests
        results_dict = {} 
        pt_arcs_list = [arc for arc, arc_info in self.arcs_dict_with_dual_cost.items() if arc_info[2]]
        
        def get_arc(origin, dest):
            if (origin,dest) in self.arcs_dict_with_dual_cost:
                return self.arcs_dict_with_dual_cost[(origin,dest)]                
            else:
                return float('inf'), float('inf'), False
        
        step_0_pt_arcs_list = pt_arcs_list + [(start_depot,pickup)]
        step_1_pt_arcs_list = pt_arcs_list + [(pickup,dropoff)]
        step_2_pt_arcs_list = pt_arcs_list + [(dropoff,dropoff)]
        
        def get_node(step, node):
            if step == 2:
                return node
            else:
                return 0
            
        def arcs_list(step):
            if step == 0:
                return step_0_pt_arcs_list
            elif step == 1:
                return step_1_pt_arcs_list
            elif step == 2:
                return step_2_pt_arcs_list
            elif step == 3:
                return self.depot_nodes

        def get_cost_time_energy(step, arc, time, energy):
            cost, updated_time, updated_energy = 0, time, energy
            if step == 0:
                if arc == (start_depot, pickup):
                    depot_pickup_time, depot_pickup_cost = get_arc(start_depot,pickup)[0:2]
                    depot_pickup_energy = depot_pickup_time
                    if depot_pickup_time > self.request_nodes_dict[pickup][1]:
                        return float('inf'), 0, 0
                    else:
                        cost = depot_pickup_cost
                        updated_time = max(self.request_nodes_dict[pickup][0], depot_pickup_time) + self.request_st
                        updated_energy = energy - depot_pickup_energy      
                else:
                    depot_transfer_time, depot_transfer_cost = get_arc(start_depot,arc[0])[0:2]
                    transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                    transfer_pickup_time, transfer_pickup_cost = get_arc(arc[1],pickup)[0:2]
                    depot_transfer_energy, transfer_pickup_energy = depot_transfer_time, transfer_pickup_time
                    if depot_transfer_time > self.transfer_nodes_dict[arc[0]]: 
                        return float('inf'), 0, 0
                    # multply transfer_st by zero due to diffrence between arc based and path based, need to investigate why this way is ok
                    transfer_pickup_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st * 1 + transfer_pickup_time
                    if transfer_pickup_time > self.request_nodes_dict[pickup][1]:
                        return float('inf'), 0, 0
                    else:
                        cost = depot_transfer_cost + transfer_transfer_cost + transfer_pickup_cost
                        updated_time =  max(transfer_pickup_time, self.request_nodes_dict[pickup][0]) + self.request_st
                        updated_energy = energy - (depot_transfer_energy + transfer_pickup_energy)
            elif step == 1:
              if arc == (pickup, dropoff):
                  pickup_dropoff_time, pickup_dropoff_cost = get_arc(pickup,dropoff)[0:2]
                  pickup_dropoff_energy = pickup_dropoff_time
                  if time + pickup_dropoff_time > self.request_nodes_dict[dropoff][1]:
                      return float('inf'), 0, 0
                  else:
                      cost = pickup_dropoff_cost
                      updated_time = max(time + pickup_dropoff_time, self.request_nodes_dict[dropoff][0]) + self.request_st
                      updated_energy = energy - pickup_dropoff_energy
              else:
                  pickup_transfer_time, pickup_transfer_cost = get_arc(pickup,arc[0])[0:2]
                  transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                  transfer_dropoff_time, transfer_dropoff_cost = get_arc(arc[1],dropoff)[0:2]
                  pickup_transfer_energy, transfer_dropoff_energy = pickup_transfer_time, transfer_dropoff_time
                  if time + pickup_transfer_time > self.transfer_nodes_dict[arc[0]]:
                      return float('inf'), 0, 0
                  # multply transfer_st by zero due to diffrence between arc based and path based, need to investigate why this way is ok
                  transfer_dropoff_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st * 1 + transfer_dropoff_time
                  if transfer_dropoff_time > self.request_nodes_dict[dropoff][1]:
                      return float('inf'), 0, 0
                  else:
                      cost = pickup_transfer_cost + transfer_transfer_cost + transfer_dropoff_cost
                      updated_time = max(transfer_dropoff_time, self.request_nodes_dict[dropoff][0]) + self.request_st
                      updated_energy = energy - (pickup_transfer_energy + transfer_dropoff_energy)
            elif step == 2:
                if arc == (dropoff, dropoff):
                    cost, updated_time, updated_energy = 0, time, energy
                else:
                    dropoff_transfer_time, dropoff_transfer_cost = get_arc(dropoff,arc[0])[0:2]
                    dropoff_transfer_energy = dropoff_transfer_time
                    if time + dropoff_transfer_time > self.transfer_nodes_dict[arc[0]]: 
                        return float('inf'), float('inf'), 0
                    transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                    cost = dropoff_transfer_cost + transfer_transfer_cost
                    updated_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st
                    updated_energy = energy - dropoff_transfer_energy
            return cost, updated_time, updated_energy
            
        def get_path(step, arc):
            path = []
            if step == 0:
                if arc == (start_depot,pickup):
                    path = [start_depot,pickup]
                else:
                    path = [start_depot,arc[0],arc[1],pickup]
            elif step == 1:
                if arc == (pickup,dropoff):
                    path = [dropoff]
                else:
                    path = [arc[0],arc[1],dropoff]
            elif step == 2:
                if arc == (dropoff,dropoff):
                    path = []
                else:
                    path = [arc[0],arc[1]] 
            elif step == 3:
               path = [arc[1]]
            return path     
        
        def find_path(step, time, energy, node):
            if (step, time, energy, node) in results_dict:
                result = results_dict[(step, time, energy, node)]
                return result[0], result[1]
            min_cost = float('inf')
            best_path = []
            path = []
            if energy < 0:
                return float('inf'), []
            elif step <= 2:
                for transfer_arc in arcs_list(step):
                    cost, updated_time, updated_energy = get_cost_time_energy(step, transfer_arc, time, energy)
                    path = get_path(step, transfer_arc)
                    if math.isinf(cost):
                        total_cost = cost
                    else:
                        future_cost, future_path = find_path(step + 1, updated_time, updated_energy, get_node(step, transfer_arc[1]))
                        total_cost = cost + future_cost
                        total_path = path + future_path
                    if total_cost < min_cost: 
                        min_cost = total_cost
                        best_path = total_path 
            elif step == 3:
                for end_depot in self.depot_nodes:
                    node_depot_time, node_depot_cost = get_arc(node,end_depot)[0:2]
                    cost = node_depot_cost
                    updated_time = time + node_depot_time
                    updated_energy = energy - node_depot_time
                    path = get_path(step, (end_depot,end_depot))
                    if math.isinf(cost): 
                        total_cost = cost
                    else:
                        future_cost, future_path = find_path(step + 1, updated_time, updated_energy, get_node(step, end_depot))
                        total_cost = cost + future_cost
                        total_path = path + future_path
                    if total_cost < min_cost: 
                        min_cost = total_cost
                        best_path = total_path 
            else:
                updated_time = time
                updated_energy = energy
                min_cost, best_path = 0, []
            results_dict[(step, time, energy, node)] = (min_cost, best_path)
            return min_cost, best_path
        min_cost, best_path = find_path(0, 0, self.robots_max_travel_time, get_node(0,start_depot))
        return min_cost, best_path
    
    def get_shortest_path_with_constriants_iterable(self, robot_pickup_iterable , return_dict):
        for robot, pickup in robot_pickup_iterable:
            min_cost, best_path = self.get_shortest_path_with_constriants(robot, pickup)
            return_dict[(robot, pickup)] = min_cost, best_path
    
    def get_path_cost(self, path):
            cost = 0
            for index, node in enumerate(path[:-1]):
                next_node = path[index+1]
                arc_cost = self.arcs_dict[(node,next_node)][1]
                cost += arc_cost
            return cost

    def generate_paths_sp(self):
        # single process implementation #
        result = {}
        for pickup in self.pickup_nodes:
            for robot in self.robots:
                min_cost, best_path = self.get_shortest_path_with_constriants(robot, pickup)
                result[(robot, pickup)] = (min_cost, best_path)
        return result

    def generate_paths_mp(self):
        # multi processing implementation #
        self.save_map(MP_MAP_PATH)
        process = subprocess.Popen(['python', MP_SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        result = eval(out.decode( "utf-8" ))
        return result

    def insert_paths_to_map_atrs(self, result, is_first_iteration = True):
        if not is_first_iteration: self.path_based_is_optimal_solution = True
        for robot_pickup, cost_path in result.items():
            robot, pickup = robot_pickup
            min_cost, best_path = cost_path
            if not math.isinf(min_cost) and best_path:
                is_negative_path_cost = min_cost - (self.pickup_duals_dict[pickup] + self.robot_duals_dict[robot]) < -0.001
                if is_negative_path_cost or is_first_iteration:
                    original_cost = self.get_path_cost(best_path)
                    if not is_first_iteration: self.path_based_is_optimal_solution = False
                    self.paths_index += 1
                    current_index = self.paths_index
                    initial_set_or_cr = 'initial_set' if is_first_iteration else 'cr'
                    self.all_paths_list.append((initial_set_or_cr, current_index, 'na', original_cost, original_cost, robot, best_path))
                    self.robots_paths_dict[robot].add(current_index)
                    self.start_paths_dict[best_path[0]].add(current_index)
                    self.end_paths_dict[best_path[-1]].add(current_index)
                    self.request_paths_dict[pickup].add(current_index)
                    for index, node in enumerate(best_path[1:-1]):
                        previous_node = best_path[index]
                        if node in self.transfer_nodes and previous_node in self.transfer_nodes:
                            if self.transfer_to_line[node] == self.transfer_to_line[previous_node]:
                                for transfer_node in range(previous_node, node):
                                    self.transfer_paths_dict[transfer_node].add(current_index)
     
    def generate_dummy_paths(self):  
        # adding dummy path for every robot
        for robot in self.robots:
            self.paths_index += 1
            self.all_paths_list.append(('robot dummy', self.paths_index, 0, 0, 0, robot, [self.robots_origin_location[robot]]))
            self.robots_paths_dict[robot].add(self.paths_index)
        # adding outsourcing path for every pickup node
        for pickup in self.pickup_nodes:
            self.paths_index += 1
            self.all_paths_list.append(('outsourcing', self.paths_index, 0, self.outsourcing_cost, self.outsourcing_cost, 0, [pickup]))
            self.request_paths_dict[pickup].add(self.paths_index)
                                
    def get_duals_values(self):
        # load duals data from dual json file
        with open(self.duals_full_path, 'r') as j:
            duals_values = j.read().replace(", ]", "]")
        self.duals_dict = json.loads(duals_values)
        # load duals data from self.duals_dict to the relevant dual dict
        for index, robot in enumerate(self.robots):
            self.robot_duals_dict[robot] = self.duals_dict['ct_robots'][index]
        for index, pickup in enumerate(self.pickup_nodes):
            self.pickup_duals_dict[pickup] = self.duals_dict['ct_pickup'][index]
        for index, transfer in enumerate(self.transfer_nodes):
            self.transfer_duals_dict[transfer] = self.duals_dict['ct_transfer'][index]
        for index, depot in enumerate(self.depot_nodes):
            self.depot_duals_dict[depot] = self.duals_dict['ct_depot'][index]
        
    def update_arcs_dict_with_dual_cost(self):
        updated_arcs_dict = {}
        for arc, arc_info in self.arcs_dict.items():
            arc_time, arc_cost, is_pt_arc = arc_info
            dual_cost = 0
            origin_node_type = self.all_nodes_dict[arc[0]].type
            dest_node_type = self.all_nodes_dict[arc[1]].type
            if is_pt_arc:
                transfer_dual = sum(self.transfer_duals_dict[transfer_node] for transfer_node in self.pt_arcs_dict[arc])
                dual_cost -= transfer_dual
            elif origin_node_type == 'depot':
                dual_cost += self.depot_duals_dict[arc[0]]
            elif dest_node_type == 'depot':
                dual_cost -= self.depot_duals_dict[arc[1]]
            updated_arcs_dict[arc] = (arc_time, arc_cost + dual_cost, is_pt_arc)
        self.arcs_dict_with_dual_cost = deepcopy(updated_arcs_dict)#.copy()       
                                                 
    def generate_map(self):
        self.generate_robot_origin_locations()
        self.generate_service_lines()
        self.generate_depot_nodes()
        self.generate_request_nodes()
        self.generate_transfer_nodes()
        self.generate_arcs()
        self.arcs_dict_with_dual_cost = deepcopy(self.arcs_dict)
          
    def generate_paths(self, single_process = True, is_first_iteration = True):
        if is_first_iteration:
            self.generate_dummy_paths()
        else:    
            self.update_arcs_dict_with_dual_cost()
            
        if single_process:
            result = self.generate_paths_sp()
        else:
            result = self.generate_paths_mp()
        self.insert_paths_to_map_atrs(result, is_first_iteration = is_first_iteration)
        
    def export_to_dat_arc_based(self):
        with open(self.dat_full_path_arc_based, "w+") as dat_file:
            dat_file.truncate(0)
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            dat_file.write(f"""
/*********************************************
 * OPL 20.1.0.0 Data
 * Author: yshapira (by data_generator.py)
 * Creation Date: {now}
 *********************************************/
                          \n""")
            dat_file.write(f'num_robots = {self.num_robots};\n')
            dat_file.write(f'num_service_lines = {self.num_service_lines};\n')
            dat_file.write(f'num_requests = {self.num_requests};\n')
            dat_file.write(f'num_transfers = {self.num_transfers};\n')
            dat_file.write(f'num_depots = {self.num_depots};\n')
            dat_file.write('M = 100000000;\n')
            dat_file.write('//robots info\n')
            robots_origin_location_dat = ",".join([str(o) for o in self.robots_origin_location.values()])
            dat_file.write(f'robots_origin_location = [{robots_origin_location_dat}];\n')
            dat_file.write(f'robots_max_travel_time = {self.robots_max_travel_time};\n')
            dat_file.write('//transfers nodes info\n')
            transfers_nodes_departure_time_dat = []
            for transfer in self.transfer_nodes:
                transfers_nodes_departure_time_dat.append(int(self.all_nodes_dict[transfer].departure_time))
            dat_file.write(f'transfers_nodes_departure_time = {transfers_nodes_departure_time_dat};\n')
            dat_file.write(f'transfers_nodes_capacity = {self.transfer_capacity};\n')
            dat_file.write(f'transfers_nodes_service_time = {self.transfer_st};\n')
            transfers_nodes_of_line_dat = []
            for line, related_trnsfers in self.service_lines_dict.items():
                list_of_transfers = ",".join([str(t) for t in related_trnsfers])
                list_of_transfers = '{'+list_of_transfers+'}'
                transfers_nodes_of_line_dat.append(list_of_transfers)
            transfers_nodes_of_line_dat = ",".join(transfers_nodes_of_line_dat)
            dat_file.write(f'transfers_nodes_of_line = [{transfers_nodes_of_line_dat}];\n')
            transfer_to_line_dat = ",".join(str(self.transfer_to_line[x]) for x in sorted(self.transfer_to_line))
            dat_file.write(f'transfer_to_line = [{transfer_to_line_dat}];\n')
            dat_file.write('//requests nodes info\n')
            dat_file.write(f'requests_nodes_service_time = {self.request_st};\n')
            requests_start_time_window_dat = []
            for request in self.request_nodes:
                requests_start_time_window_dat.append(self.request_nodes_dict[request][0])
            dat_file.write(f'requests_start_time_window = {requests_start_time_window_dat};\n')
            requests_end_time_window_dat = []
            for request in self.request_nodes:
                requests_end_time_window_dat.append(self.request_nodes_dict[request][1])
            dat_file.write(f'requests_end_time_window = {requests_end_time_window_dat};\n')
            outsourcing_cost_dat = [self.outsourcing_cost] * self.num_requests
            dat_file.write(f'requests_outsourcing_cost = {outsourcing_cost_dat};\n')
            dat_file.write('//depots nodes info\n')
            dat_file.write(f'depots_nodes_service_time = {self.depot_st};\n')
            dat_file.write(f'depots_nodes_capacity = {self.depot_capacity};\n')
            dat_file.write('//arcs info\n')
            arcs_list_dat = []
            for arc, arc_info in self.arcs_dict.items():
                arc_str = '<'+str(arc[0])+','+str(arc[1])+','+str(arc_info[0])+','+str(arc_info[1])+'>'
                arcs_list_dat.append(arc_str)
            arcs_list_dat = ",\n".join(arcs_list_dat)
            arcs_list_dat = '{' + arcs_list_dat + '}'
            dat_file.write(f'all_arcs = \n{arcs_list_dat}\n;\n')
            print("Arc based .dat file created")

    def export_to_dat_path_based(self):
        with open(self.dat_full_path_path_based, "w+") as dat_file:
            dat_file.truncate(0)
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            dat_file.write(f"""
/*********************************************
 * OPL 20.1.0.0 Data
 * Author: yshapira (by data_generator.py)
 * Creation Date: {now}
 *********************************************/
                          \n""")
            dat_file.write(f'num_robots = {self.num_robots};\n')
            dat_file.write(f'num_requests = {self.num_requests};\n')
            dat_file.write(f'num_transfers = {self.num_transfers};\n')
            dat_file.write(f'num_depots = {self.num_depots};\n')
            dat_file.write(f'transfers_nodes_capacity = {self.transfer_capacity};\n')
            dat_file.write(f'depots_nodes_capacity = {self.depot_capacity};\n')
            all_paths_dat = ",".join([str(o[1]) for o in self.all_paths_list])
            dat_file.write('all_paths = {'+all_paths_dat+'};\n')
            path_cost_dat = ",".join([str(o[3]) for o in self.all_paths_list])
            dat_file.write(f'path_cost = [{path_cost_dat}];\n')
            depots_initial_assignment = ",".join([str(o) for o in self.depots_initial_assignment_dict.values()])
            dat_file.write(f'depots_initial_assignment = [{depots_initial_assignment}];\n')
            robots_paths_dat = []
            for robot, related_paths in self.robots_paths_dict.items():
                list_of_paths = ",".join([str(p) for p in related_paths])
                list_of_paths = '{'+list_of_paths+'}'
                robots_paths_dat.append(list_of_paths)
            robots_paths_dat = ",".join(robots_paths_dat)
            dat_file.write(f'robot_paths = [{robots_paths_dat}];\n')
            requests_paths_dat = []
            for request, related_paths in self.request_paths_dict.items():
                list_of_paths = ",".join([str(p) for p in related_paths])
                list_of_paths = '{'+list_of_paths+'}'
                requests_paths_dat.append(list_of_paths)
            requests_paths_dat = ",".join(requests_paths_dat)
            dat_file.write(f'request_paths = [{requests_paths_dat}];\n')
            transfers_paths_dat = []
            for transfer, related_paths in self.transfer_paths_dict.items():
                list_of_paths = ",".join([str(p) for p in related_paths])
                list_of_paths = '{'+list_of_paths+'}'
                transfers_paths_dat.append(list_of_paths)
            transfers_paths_dat = ",".join(transfers_paths_dat)
            dat_file.write(f'transfer_paths = [{transfers_paths_dat}];\n')
            depos_start_paths_dat = []
            for depo, related_paths in self.start_paths_dict.items():
                list_of_paths = ",".join([str(p) for p in related_paths])
                list_of_paths = '{'+list_of_paths+'}'
                depos_start_paths_dat.append(list_of_paths)
            depos_start_paths_dat = ",".join(depos_start_paths_dat)
            dat_file.write(f'depot_start_paths = [{depos_start_paths_dat}];\n')
            depos_end_paths_dat = []
            for depo, related_paths in self.end_paths_dict.items():
                list_of_paths = ",".join([str(p) for p in related_paths])
                list_of_paths = '{'+list_of_paths+'}'
                depos_end_paths_dat.append(list_of_paths)
            depos_end_paths_dat = ",".join(depos_end_paths_dat)
            dat_file.write(f'depot_end_paths = [{depos_end_paths_dat}];\n')
            print("Path based .dat file created")
            # save paths data to csv file
            csv_headers = ['path_type','path_id', 'time', 'cost', 'cost_with_dual', 'robot', 'list_of_nodes_in_path']
            with open(self.paths_data_full_path ,'w', newline="") as rfile:
                write = csv.writer(rfile)
                write.writerow(csv_headers)
                write.writerows(self.all_paths_list)
                
    def get_map_summary(self):
        summary = 'request_pairs: ' + str(self.request_pairs) + ' '
        summary += 'robots_origin_location: ' + str(self.origin_location_dict) + ' '
        summary += 'service_lines: ' + str(self.service_lines_dict) + ' '
        return summary.replace(',', ' ')
    
    def get_map_details(self):
        summary = '' 
        summary += 'All relevant information about the map is below:\n\n' 
        summary += f'Total # of nodes: {self.num_nodes}\n'
        summary += f'# of robots: {self.num_robots}, robots_origin_location: {str(self.origin_location_dict)}\n'
        summary += f'# of requests: {self.num_requests}, request_pairs: {str(self.request_pairs)}\n'
        summary += f'# of depots: {self.num_depots}, depot nodes: {str(list(self.depot_nodes))}\n'
        summary += f'# of service lines: {self.num_service_lines}\n'
        summary += f'# of num transfers: {self.num_transfers}, service lines mapping: {str(self.service_lines_dict)}\n'
        summary += f'public transportation factor: {self.pt_coeff}\n'
        summary += f'map_size: {self.map_size}\n'
        summary += f'outsourcing cost: {self.outsourcing_cost}\n'
        summary += f'robots max travel time: {self.robots_max_travel_time}\n'

        summary += f'request service time: {self.request_st}\n'
        summary += f'request half time window: {self.request_half_time_window}\n'
        summary += f'request max time: {self.request_max_time}\n'
        summary += f'depot capacity: {self.depot_capacity}\n'
        summary += f'depot service time: {self.depot_st}\n'
        summary += f'transfer capacity: {self.transfer_capacity}\n'
        summary += f'transfer service time: {self.transfer_st}\n'
        summary += f'request nodes time windows: {str(self.request_nodes_dict)}\n'
        summary += f'transfer nodes departure time: {str(self.transfer_nodes_dict)}\n'
        summary += f'arcs info: --> (start node, end node): (time, cost, is public transportation arc? T/F)\n'
        for arc, arc_info in self.arcs_dict.items():
            summary += f'   {arc}: {arc_info}\n'
        return summary
    
    # updated definition of generate_paths_sp Map method
    def get_shortest_path_with_constriants_iterable_v2(self, depot_pickup_iterable , return_dict):
        for depot, pickup in depot_pickup_iterable:
            min_cost, best_path = self.get_shortest_path_with_constriants(depot, pickup)
            return_dict[(depot, pickup)] = min_cost, best_path
        
    def generate_paths_sp_v2(self):
        # single process implementation #
        result = {}
        for pickup in self.pickup_nodes:
            for depot in self.depot_nodes:
                min_cost, best_path = self.get_shortest_path_with_constriants(depot, pickup)
                result[(depot, pickup)] = (min_cost, best_path)
        return result

    # updated definition of get_shortest_path_with_constriants_v2 Map method
    def get_shortest_path_with_constriants_v2(self, depot, pickup):
        start_depot = depot
        dropoff = pickup + self.num_requests
        results_dict = {} 
        pt_arcs_list = [arc for arc, arc_info in self.arcs_dict_with_dual_cost.items() if arc_info[2]]
        
        def get_arc(origin, dest):
            if (origin,dest) in self.arcs_dict_with_dual_cost:
                return self.arcs_dict_with_dual_cost[(origin,dest)]                
            else:
                return float('inf'), float('inf'), False
        
        step_0_pt_arcs_list = pt_arcs_list + [(start_depot,pickup)]
        step_1_pt_arcs_list = pt_arcs_list + [(pickup,dropoff)]
        step_2_pt_arcs_list = pt_arcs_list + [(dropoff,dropoff)]
        
        def get_node(step, node):
            if step == 2:
                return node
            else:
                return 0
            
        def arcs_list(step):
            if step == 0:
                return step_0_pt_arcs_list
            elif step == 1:
                return step_1_pt_arcs_list
            elif step == 2:
                return step_2_pt_arcs_list
            elif step == 3:
                return self.depot_nodes
    
        def get_cost_time_energy(step, arc, time, energy):
            cost, updated_time, updated_energy = 0, time, energy
            if step == 0:
                if arc == (start_depot, pickup):
                    depot_pickup_time, depot_pickup_cost = get_arc(start_depot,pickup)[0:2]
                    depot_pickup_energy = depot_pickup_time
                    if depot_pickup_time > self.request_nodes_dict[pickup][1]:
                        return float('inf'), 0, 0
                    else:
                        cost = depot_pickup_cost
                        updated_time = max(self.request_nodes_dict[pickup][0], depot_pickup_time) + self.request_st
                        updated_energy = energy - depot_pickup_energy      
                else:
                    depot_transfer_time, depot_transfer_cost = get_arc(start_depot,arc[0])[0:2]
                    transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                    transfer_pickup_time, transfer_pickup_cost = get_arc(arc[1],pickup)[0:2]
                    depot_transfer_energy, transfer_pickup_energy = depot_transfer_time, transfer_pickup_time
                    if depot_transfer_time > self.transfer_nodes_dict[arc[0]]: 
                        return float('inf'), 0, 0
                    # multply transfer_st by zero due to diffrence between arc based and path based, need to investigate why this way is ok
                    transfer_pickup_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st * 1 + transfer_pickup_time
                    if transfer_pickup_time > self.request_nodes_dict[pickup][1]:
                        return float('inf'), 0, 0
                    else:
                        cost = depot_transfer_cost + transfer_transfer_cost + transfer_pickup_cost
                        updated_time =  max(transfer_pickup_time, self.request_nodes_dict[pickup][0]) + self.request_st
                        updated_energy = energy - (depot_transfer_energy + transfer_pickup_energy)
            elif step == 1:
              if arc == (pickup, dropoff):
                  pickup_dropoff_time, pickup_dropoff_cost = get_arc(pickup,dropoff)[0:2]
                  pickup_dropoff_energy = pickup_dropoff_time
                  if time + pickup_dropoff_time > self.request_nodes_dict[dropoff][1]:
                      return float('inf'), 0, 0
                  else:
                      cost = pickup_dropoff_cost
                      updated_time = max(time + pickup_dropoff_time, self.request_nodes_dict[dropoff][0]) + self.request_st
                      updated_energy = energy - pickup_dropoff_energy
              else:
                  pickup_transfer_time, pickup_transfer_cost = get_arc(pickup,arc[0])[0:2]
                  transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                  transfer_dropoff_time, transfer_dropoff_cost = get_arc(arc[1],dropoff)[0:2]
                  pickup_transfer_energy, transfer_dropoff_energy = pickup_transfer_time, transfer_dropoff_time
                  if time + pickup_transfer_time > self.transfer_nodes_dict[arc[0]]:
                      return float('inf'), 0, 0
                  # multply transfer_st by zero due to diffrence between arc based and path based, need to investigate why this way is ok
                  transfer_dropoff_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st * 1 + transfer_dropoff_time
                  if transfer_dropoff_time > self.request_nodes_dict[dropoff][1]:
                      return float('inf'), 0, 0
                  else:
                      cost = pickup_transfer_cost + transfer_transfer_cost + transfer_dropoff_cost
                      updated_time = max(transfer_dropoff_time, self.request_nodes_dict[dropoff][0]) + self.request_st
                      updated_energy = energy - (pickup_transfer_energy + transfer_dropoff_energy)
            elif step == 2:
                if arc == (dropoff, dropoff):
                    cost, updated_time, updated_energy = 0, time, energy
                else:
                    dropoff_transfer_time, dropoff_transfer_cost = get_arc(dropoff,arc[0])[0:2]
                    dropoff_transfer_energy = dropoff_transfer_time
                    if time + dropoff_transfer_time > self.transfer_nodes_dict[arc[0]]: 
                        return float('inf'), float('inf'), 0
                    transfer_transfer_time, transfer_transfer_cost = get_arc(arc[0],arc[1])[0:2]
                    cost = dropoff_transfer_cost + transfer_transfer_cost
                    updated_time = self.transfer_nodes_dict[arc[0]] + transfer_transfer_time + self.transfer_st
                    updated_energy = energy - dropoff_transfer_energy
            return cost, updated_time, updated_energy
            
        def get_path(step, arc):
            path = []
            if step == 0:
                if arc == (start_depot,pickup):
                    path = [start_depot,pickup]
                else:
                    path = [start_depot,arc[0],arc[1],pickup]
            elif step == 1:
                if arc == (pickup,dropoff):
                    path = [dropoff]
                else:
                    path = [arc[0],arc[1],dropoff]
            elif step == 2:
                if arc == (dropoff,dropoff):
                    path = []
                else:
                    path = [arc[0],arc[1]] 
            elif step == 3:
               path = [arc[1]]
            return path     
        
        def find_path(step, time, energy, node):
            if (step, time, energy, node) in results_dict:
                result = results_dict[(step, time, energy, node)]
                return result[0], result[1]
            min_cost = float('inf')
            best_path = []
            path = []
            if energy < 0:
                return float('inf'), []
            elif step <= 2:
                for transfer_arc in arcs_list(step):
                    cost, updated_time, updated_energy = get_cost_time_energy(step, transfer_arc, time, energy)
                    path = get_path(step, transfer_arc)
                    if math.isinf(cost):
                        total_cost = cost
                    else:
                        future_cost, future_path = find_path(step + 1, updated_time, updated_energy, get_node(step, transfer_arc[1]))
                        total_cost = cost + future_cost
                        total_path = path + future_path
                    if total_cost < min_cost: 
                        min_cost = total_cost
                        best_path = total_path 
            elif step == 3:
                for end_depot in self.depot_nodes:
                    node_depot_time, node_depot_cost = get_arc(node,end_depot)[0:2]
                    cost = node_depot_cost
                    updated_time = time + node_depot_time
                    updated_energy = energy - node_depot_time
                    path = get_path(step, (end_depot,end_depot))
                    if math.isinf(cost): 
                        total_cost = cost
                    else:
                        future_cost, future_path = find_path(step + 1, updated_time, updated_energy, get_node(step, end_depot))
                        total_cost = cost + future_cost
                        total_path = path + future_path
                    if total_cost < min_cost: 
                        min_cost = total_cost
                        best_path = total_path 
            else:
                updated_time = time
                updated_energy = energy
                min_cost, best_path = 0, []
            results_dict[(step, time, energy, node)] = (min_cost, best_path)
            return min_cost, best_path
        min_cost, best_path = find_path(0, 0, self.robots_max_travel_time, get_node(0,start_depot))
        return min_cost, best_path

    # updated definition of insert_paths_to_map_atrs_v2 Map method
    def insert_paths_to_map_atrs_v2(self, result, is_first_iteration = True):
        if not is_first_iteration: self.path_based_is_optimal_solution = True
        for depot_pickup, cost_path in result.items():
            depot, pickup = depot_pickup
            min_cost, best_path = cost_path
            if not math.isinf(min_cost) and best_path:
                for robot in self.origin_location_dict[depot]:
                    is_negative_path_cost = min_cost - (self.pickup_duals_dict[pickup] + self.robot_duals_dict[robot]) < -0.001
                    if is_negative_path_cost or is_first_iteration:
                        original_cost = self.get_path_cost(best_path)
                        if not is_first_iteration: self.path_based_is_optimal_solution = False
                        self.paths_index += 1
                        current_index = self.paths_index
                        initial_set_or_cr = 'initial_set' if is_first_iteration else 'cr'
                        self.all_paths_list.append((initial_set_or_cr, current_index, 'na', original_cost, original_cost, robot, best_path))
                        self.robots_paths_dict[robot].add(current_index)
                        self.start_paths_dict[best_path[0]].add(current_index)
                        self.end_paths_dict[best_path[-1]].add(current_index)
                        self.request_paths_dict[pickup].add(current_index)
                        for index, node in enumerate(best_path[1:-1]):
                            previous_node = best_path[index]
                            if node in self.transfer_nodes and previous_node in self.transfer_nodes:
                                if self.transfer_to_line[node] == self.transfer_to_line[previous_node]:
                                    for transfer_node in range(previous_node, node):
                                        self.transfer_paths_dict[transfer_node].add(current_index)


    


