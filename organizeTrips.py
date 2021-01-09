# organize chicago data for team project
import csv
import pickle
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.cElementTree as eTree
from scipy.optimize import linear_sum_assignment


class Task:
    def __init__(self, task_id, depart_time, arrive_time, depart_lat, depart_long, arrive_lat, arrive_long, trip_miles):
        self.task_id = task_id
        self.arrive_long = arrive_long
        self.arrive_lat = arrive_lat
        self.depart_long = depart_long
        self.depart_lat = depart_lat
        self.arrive_time = arrive_time
        self.depart_time = depart_time
        self.trip_miles = trip_miles


class Vehicle:
    def __init__(self, v_id, first_task, schedule):
        self.vehicle_id = v_id
        self.tasks = []
        self.current_task = first_task
        self.work_start_time = first_task.depart_time
        self.schedule = schedule
        self.schedule_index = 0


def distance(v, t):
    return (v.pos_lat - t.depart_lat) ** 2 + (v.pos_long - t.depart_long) ** 2


def min_math(w_vehicles, r_vehicles, w_tasks):
    dist_matrix = []
    for w_vehicle in w_vehicles:
        row_list = []
        for waiting_task in w_tasks:
            row_list.append(distance(w_vehicle, waiting_task))
        dist_matrix.append(row_list)
    cost = np.array(dist_matrix)
    w_vehicles_size = len(w_vehicles)
    w_tasks_size = len(w_tasks)

    if w_vehicles_size <= w_tasks_size:
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(w_vehicles_size):
            w_vehicles[i].current_task = w_tasks[col_ind[i]]
            w_tasks[col_ind[i]] = None
            r_vehicles.append(w_vehicles[i])
            w_vehicles[i] = None
    else:
        cost = np.transpose(cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(w_tasks_size):
            w_vehicles[col_ind[i]].current_task = w_tasks[i]
            w_tasks[i] = None
            r_vehicles.append(w_vehicles[col_ind[i]])
            w_vehicles[col_ind[i]] = None
    w_vehicles = list(filter(lambda v: v is not None, w_vehicles))
    w_tasks = list(filter(lambda v: v is not None, w_tasks))
    return w_vehicles, r_vehicles, w_tasks


def index_convert(dt):
    h = dt.hour - 3 if dt.hour >= 3 else dt.hour + 21
    return h * 4 + int(dt.minute / 15)


starttime = datetime.now()
# schedule prepare divided into 24 lists
schedule_list = [[] for i in range(24 * 4)]
persons_intervals = pickle.load(open("persons_with_interval.pY", "rb"))#?
for schedule in persons_intervals:
    schedule = list(map(lambda s: s + timedelta(days=698), schedule))#?698
    hour = schedule[0].hour - 3
    hour = hour if hour >= 0 else hour + 24#?24
    count = hour * 3600 + schedule[0].minute * 60 + schedule[0].second
    index = int(count / (15 * 60))
    schedule_list[index].append(schedule)

trip_list = []
with open('trips_complete.csv', 'r', encoding='utf-8') as file:
    counter = 0
    reader = csv.reader(file, dialect='excel')
    for item in reader:
        item.append(counter)
        counter = counter + 1
        trip_list.append(item)
    trip_list.remove(trip_list[0])
    trip_size = len(trip_list)
    random.shuffle(trip_list)
    trip_list = trip_list[0:trip_size]
    trip_list.sort(key=lambda trip: trip[-1])

tasks_list = [[] for i in range(24 * 4)]
for item in trip_list:
    if len(item[0]) == 0 or len(item[1]) == 0 or \
            len(item[2]) == 0 or len(item[15]) == 0 or \
            len(item[16]) == 0 or len(item[18]) == 0 or len(item[19]) == 0:
        continue
    task_id = item[0]
    depart_time = datetime.strptime(item[1][:-4], "%Y-%m-%dT%H:%M:%S")#removed last 4chars
    arrive_time = datetime.strptime(item[2][:-4], "%Y-%m-%dT%H:%M:%S")
    arrive_time = arrive_time + timedelta(minutes=15) if arrive_time == depart_time else arrive_time
    depart_latitude = float(item[15])
    depart_longitude = float(item[16])
    arrive_latitude = float(item[18])
    arrive_longitude = float(item[19])
    trip_miles = float(item[4])
    task = Task(task_id, depart_time, arrive_time, depart_latitude,
                depart_longitude, arrive_latitude, arrive_longitude, trip_miles)
    ind = index_convert(task.depart_time)#?
    tasks_list[ind].append(task)#load objects in tasks_list

pic_total_driver = [0 for i in range(24 * 4)]
pic_available_driver = [0 for i in range(24 * 4)]
pic_trips_to_match = [0 for i in range(24 * 4)]
pic_ongoing_trips = [0 for i in range(24 * 4)]

running_vehicles = []
waiting_vehicles = []
counter = 0
vehicle_id = 0
for time_index in range(24 * 4):
    print(time_index)
    pic_trips_to_match[time_index] = len(tasks_list[time_index])

    # every 15 minute update the vehicle position and task status
    for i in range(len(running_vehicles)):
        vehicle = running_vehicles[i]
        task = vehicle.current_task
        if time_index >= index_convert(task.arrive_time):
            waiting_vehicles.append(vehicle)#bcoz, that running vehicle alerady finishes its trip
            running_vehicles[i] = None
            vehicle.tasks.append(task)
            vehicle.pos_lat = task.arrive_lat
            vehicle.pos_long = task.arrive_long
    running_vehicles = list(filter(lambda v: v is not None, running_vehicles))

    for vehicle in list(filter(lambda v: v.schedule_index % 2 == 0, waiting_vehicles)):
        next_time = vehicle.schedule[vehicle.schedule_index + 1]
        if time_index >= index_convert(next_time):
            vehicle.schedule_index += 1
    for vehicle in list(
            filter(lambda v: v.schedule_index % 2 == 1 and v.schedule_index != len(v.schedule) - 1, waiting_vehicles)):
        next_time = vehicle.schedule[vehicle.schedule_index + 1]
        if time_index >= index_convert(next_time):
            vehicle.schedule_index += 1

    # waiting tasks dispensing
    on_working_vehicle = list(filter(lambda v: v.schedule_index % 2 == 0, waiting_vehicles))
    waiting_vehicles = [x for x in waiting_vehicles if x not in on_working_vehicle]
    if len(on_working_vehicle) > 0 and len(tasks_list[time_index]) > 0:
        on_working_vehicle, running_vehicles, tasks_list[time_index] = \
            min_math(on_working_vehicle, running_vehicles, tasks_list[time_index])
        waiting_vehicles.extend(on_working_vehicle)

    # add new vehicle
   
    for task in tasks_list[time_index]:  # get new tasks for this time
        schedules = schedule_list[time_index]#schedule_list for time index 0 has tasks for several vehicles
        for i in range(len(schedules)):
            if len(schedules[i])==2:
                working_hr=((schedules[i][1]-schedules[i][0]).total_seconds())/3600
                schedules[i].append(working_hr)
            elif len(schedules[i])==4:
                working_hr_1=((schedules[i][1]-schedules[i][0]).total_seconds())/3600
                working_hr_2=((schedules[i][3]-schedules[i][2]).total_seconds())/3600
                working_hr=working_hr_1+working_hr_2   
                schedules[i].append(working_hr)
            elif len(schedules[i])==6:
                working_hr_1=((schedules[i][1]-schedules[i][0]).total_seconds())/3600
                working_hr_2=((schedules[i][3]-schedules[i][2]).total_seconds())/3600
                working_hr_3=((schedules[i][5]-schedules[i][4]).total_seconds())/3600
                working_hr=working_hr_1+working_hr_2+working_hr_3
                schedules[i].append(working_hr)
            elif len(schedules[i])==8:
                working_hr_1=((schedules[i][1]-schedules[i][0]).total_seconds())/3600
                working_hr_2=((schedules[i][3]-schedules[i][2]).total_seconds())/3600
                working_hr_3=((schedules[i][5]-schedules[i][4]).total_seconds())/3600
                working_hr_4=((schedules[i][7]-schedules[i][6]).total_seconds())/3600
                working_hr=working_hr_1+working_hr_2+working_hr_3+working_hr_4
                schedules[i].append(working_hr)
            elif len(schedules[i])==10:
                working_hr_1=((schedules[i][1]-schedules[i][0]).total_seconds())/3600
                working_hr_2=((schedules[i][3]-schedules[i][2]).total_seconds())/3600
                working_hr_3=((schedules[i][5]-schedules[i][4]).total_seconds())/3600
                working_hr_4=((schedules[i][7]-schedules[i][6]).total_seconds())/3600
                working_hr_5=((schedules[i][9]-schedules[i][8]).total_seconds())/3600
                working_hr=working_hr_1+working_hr_2+working_hr_3+working_hr_4+working_hr_5
                schedules[i].append(working_hr)          
                
        mu=3.37 #see pdf_workinghours.py, line 137-161 to see why normal distribution chosen and how its parameters are estimated
        sigma=2.93                   
        schedules_f_rand_dist=list(filter(lambda v:v[-1] <= np.random.normal(mu,sigma,1)\
                                           ,schedules))
            
# =============================================================================
#        #Alternative:
#         rand_w_hours=np.random.normal(mu,sigma,1)
#         schedules_f_rand_dist=[]
#         for m in range(len(schedules)):
#             if schedules[m][-1]<=rand_w_hours:
#                 schedules_f_rand_dist.append(schedules[m]) 
# =============================================================================
                
        schedule = random.sample(schedules_f_rand_dist, 1)[0]#randomly choose one driver and first available time
        for i in range(len(schedules)):
            del schedules[i][-1]
        del schedule[-1]
        new_vehicle = Vehicle(str(vehicle_id), task, schedule)
        vehicle_id = vehicle_id + 1
        running_vehicles.append(new_vehicle)
    pic_ongoing_trips[time_index] = len(running_vehicles) - pic_trips_to_match[time_index]
    pic_total_driver[time_index] = len(on_working_vehicle) + len(running_vehicles)

x = [i * 0.25 for i in range(24 * 4)]

plt.plot(x, pic_trips_to_match, label='trips to match')
plt.plot(x, pic_ongoing_trips, label='occupied drivers')
plt.plot(x, pic_total_driver, label='total drivers')

local_font = {'size': 10}
plt.xlim((0, 26))
plt.ylim((0, 10000))
my_x_ticks = np.arange(0, 26, 4)
my_y_ticks = np.arange(0, 10000, 1000)
plt.xticks(my_x_ticks, fontsize=13)
plt.yticks(my_y_ticks, fontsize=13)
plt.xlabel("Time (Hour)", local_font)
plt.ylabel("Counts", local_font)

endtime = datetime.now()
print((endtime - starttime).seconds)

for vehicle in running_vehicles:
    vehicle.tasks.append(vehicle.current_task)
vehicles = []
vehicles.extend(waiting_vehicles)
vehicles.extend(running_vehicles)
vehicles.extend(on_working_vehicle)

persons = eTree.Element("persons")
for vehicle in vehicles:
    entry = eTree.SubElement(persons, "entry")
    eTree.SubElement(entry, "ID").text = vehicle.vehicle_id
    person = eTree.SubElement(entry, "person")
    trip_miles_sum = 0
    for task in vehicle.tasks:
        task_xml = eTree.SubElement(person, "task", {"taskID": task.task_id, "departTime": task.depart_time.strftime('%Y-%m-%d %H:%M'),
                                                     "arriveTime": task.arrive_time.strftime('%Y-%m-%d %H:%M')})
        eTree.SubElement(task_xml, "departCoord",
                         {"longitude": str(task.depart_long), "latitude": str(task.depart_lat)})
        eTree.SubElement(task_xml, "arriveCoord",
                         {"longitude": str(task.arrive_long), "latitude": str(task.arrive_lat)})
        trip_miles_sum = trip_miles_sum + task.trip_miles
    person.attrib["trip_miles_sum"] = str(trip_miles_sum)
tree = eTree.ElementTree(persons)
tree.write('persons.xml', encoding="utf-8", xml_declaration=True)

plt.legend()
plt.show()
