import numpy as np
import csv 
import math 
import random
import time
import pandas as pd

class TimeExceed(Exception):
    pass

##### IMPORTATION DE INSTANCE

#f = open(r"C:\Users\mrvnb\Downloads\taillard\taillard\tai01.txt") 
f = open(r"C:\Users\mrvnb\Downloads\taillard\taillard\tai03.txt")

data = csv.reader(f,delimiter =" ")

data1 =list(data)
for i in data1:
    for j in range(len(i)-1):
        i[j]=float(i[j])
    


#lines = f.readlines()

f.close()

del data1[0]

release_time = [int(data1[i][0]) for i in range(len(data1))]

for i in data1:
    i.pop()
    del i[0:4]




#On créé le fichier jobs avec i ligne correspond a i jobs (donc jobs[i]), et 
#opérations sont liés à tps de cycle (op,tps de cycle)
#donc jobs[i][j] donc pour le jobs i => (machine de l'op j, tps de cycle j)
#ainsi machine = 0, tps de cycle = 1, jobs[i][j][0] pour avoir la machine de l'op j 

jobs = [[(int(machine), int(time)) for machine, time in zip(*[iter(line)]*2)] for line in data1]


######### CREATION DE FONCTIONS 


#Créattion d'une fonction coût pour voir le coût de la fonction qui prend un tableau normalisé

def cost_partial(jobs, partial_schedule):
    return cost(jobs,normalize_schedule(partial_schedule))

#on crée ce tableau normalisé, enfin cet edt normalisé

def normalize_schedule(jobs,partial_schedule):
    j = len(jobs)
    m = len(jobs[0])
    
    
    occurences = [0]*j #liste proportionnelle aux nombres de jobs
    normalized_schedule = []
    
    #Normalized schedule est donc le nombre de machines 
    
    for t in partial_schedule:
        
        if occurences[t] < m: #si l'occurences de mon jobs t est plus petit que le nombre de machines
            normalized_schedule.append(t) #on rentre la valeur de t dans notre liste d'edt normalisé
            occurrences[t] +=1 #on ajoute +1 à l'occurences
        else:
            pass
    
    for t, count in enumerate(occurences):
        if count < m:
            normalized_schedule.extend([t]*(m - count))
    return normalizedSchedule

###  PRINT SCHEDULE On peut print le schedule mais plus tard 


#### FONCTIONS 


#On compare la somme des temps d'un job et la somme des temps sur une machine

def lowerBound(jobs):
    
    #FOCUS JOBS
    # Regarde qui a le temps le plus long entre tous jobs (on add les tps de cycle) donc ici JOBS
    def lower0():
        return max(sum(time for _,time in job) for job in jobs)
    
    #FOCUS MACHINE
    #Objectif est de calculer le temps total par machine donc M1 tous les temps de cycles de tous les jobs
    def lower1():
        mtimes1 = [0]* num_machine(jobs) #on crée une liste prop aux nbrs de machines, on va rentrer les tps machines dedans
        
        for job in jobs :
            for i in range(len(job)):
                mtimes1[0]+= job[i][1]
                
        return max(mtimes1)
    
    #puis on regarde le max des deux
    return max(lower0(),lower1())

#Caractéristiques de notre instance

def num_machine(jobs):
    return len(jobs[0])    

def num_jobs(jobs):
    return len(jobs)

def shuffle(x,start=0, stop=None):
    
    #création d'une fonction shuffle, mélange
    
    #on prend la valeur de x qui est donné
    if stop is None or stop > len(x):
        stop = len(x)
    
    for i in reversed(range(start+1, stop)):
        
        #j est un entier au hasard entre 0 et i, donc 0 et 1 puis 0 et 2 etc ...
        j = random.randint(start,i)
        #on échange
        x[i], x[j] = x[j], x[i]
        
        
##### Algorithme Print schedule

def print_schedule(jobs, schedule): 
    def format_job(time, jobnr):
        if time==1:
            return '#'
        if time ==2:
            return '[]'
        
        js =str(jobnr)
        
        if 2+len(js) <= time :
            return ('[{:^' + str(time-2) +'}]').format(jobnr)
        return '#'*time 
    
    j= len(jobs)
    m =len(jobs[0])
    
    tj = [0]*j
    tm = [0]*m
    
    ij = [0]*j
    
    final = np.zeros((j,m))
    
    output = [""]*m
    
    for i in range(len(schedule)):
        x = schedule[i]
        machine =jobs[x][ij[x]][0] -1
        #print(machine)
        time =jobs[x][ij[x]][1]
        ij[x] +=1
        
        start = max(tj[x], tm[machine])
        space = start - tm[machine]
        end = start + time
        tj[x] = end
        tm[machine] = end
        output[machine]+=' '*space + format_job(time,i)
        final[x][machine] =start
        
    
    print("")
   #print("Optimal Schedule: ")
    #[print("Machine ", idx, ":", machine_schedule) for idx, machine_schedule in enumerate(output)]
    print("")
    print("Optimal Schedule Length: ", max(tm))
    print(tj)
    print(tm)
    print("makespan :", max(tj))
    print(final)
    
    pd.DataFrame(final).to_csv('sample.csv')
    
    
    


#### ALGORITHME RECUIT SQUELETTE


#On commence par mélanger la liste
#Ca crée une liste de la liste des opérations avec un décompte de 1 à 15, 15 fois rangé de façon aléatoire
def random_schedule(j,m) :
    schedule = [i for i in list(range(j)) for _ in range(m)]
    random.shuffle(schedule)
    return schedule
    
j = len(jobs)
m = len(jobs[0])


#### ON CALCULE LE MAKESPAN 
def cout(jobs,schedule):
    j = len(jobs)
    m = len(jobs[0])
    
    tj= [0]*j #temps des j jobs
    tm = [0]*m #temps des m jobs
    
    ij = [0]*j #donc liste des 15 jobs
    
    for i in schedule: #du coup on prend dans la liste de 255 mais quand on prend i on prend sa valeur
    #sa valeur est bien entre 0 et 14 !! (on retrouve nos 15 jobs)
    
        machine = jobs[i][ij[i]][0]-1
        time = jobs[i][ij[i]][1]
        #on va prendre itération après itérations après la boucle de notre schedule 255, toutes les opérations
    
        start = max(tj[i],tm[machine])
        end = start + time
        tj[i] = end
        tm[machine] = end
    return max(tm)
    

### A retravailler mais on obtient une matrice de 255 sur 254, enfin 254 listes de longueur 255 (avec toutes les opé)
def get_neigbors(state, mode="normal"):
    neighbors = []
    
    for i in range(len(state)-1):  
        n = state[:]
        if mode == "normal":
            swap_index = i+1
        elif mode == "random":
            swap_index = random.randrange(len(state))
            
        n[i], n[swap_index] = n[swap_index], n[i]
        neighbors.append(n)
        
    return neighbors


def simulated_annealing(jobs, T, termination, halting, mode, decrease):
    
    total_jobs = len(jobs)
    total_machines = len(jobs[0])
    
    state = random_schedule(total_jobs, total_machines)
    
    for i in range(halting):
        T = decrease * float(T)
        
        for k in range(termination):
            actual_cost = cout(jobs, state)
            
            for n in get_neigbors(state, mode):
                
                n_cost = cout(jobs,n)
                
                if n_cost < actual_cost :
                    state =n
                    actual_cost = n_cost
                    
                else:
                    probability = math.exp(-n_cost/T)
                    
                    if random.random() < probability:
                        actual_cost = n_cost
                        
    return actual_cost, state



### Fonction de recherche du recuit simulé

def simulated_annealing_search(jobs, max_time=None, T=200, termination=10, halting =10,mode ="random", decrease =0.8):
    
    num_experiments = 1
    
    solutions = []
    best = 10000000
    
    t0 = time.time()
    total_experiments = 0
    
    j = len(jobs)
    m = len(jobs[0])
    rs = random_schedule(j,m)
    
    while True:
        try:
            start = time.time()
            
            for i in range(num_experiments): 
                cost, schedule = simulated_annealing(jobs, T=T, termination=termination, halting = halting, mode = "random", decrease = decrease)
                
                if cost < best :
                    best =cost
                    solutions.append((cost, schedule))
            
                    
            total_experiments+= num_experiments 
            
            if max_time and time.time() - t0 > max_time:
                raise TimeExceed("Time is over")
                
            t = time.time() - start
            
            
            if t> 0:
                print("Best:", best, "({:1f} Experiments/s, {:.1f} s)".format(num_experiments/t, time.time() - t0))
                
            if t> 4: 
                num_experiments //=2
                num_experiments = max(num_experiments,1)
            elif t <1.5:
                num_experiments*=2
                
        except(KeyboardInterrupt, TimeExceed) as e:
            print()
            print("==================================================")
            print("Best Solution:")
            print(solutions[-1][-1])
            #print("Found in {:1.f} experiments in {:1.f}s".format(total_experiments, time.time()-t0))
            
            return solutions[-1]
                

cost, solution = simulated_annealing_search(jobs, max_time=20, T =int(200), termination =int(10), halting = int(10), mode ="random", decrease= float(0.8))

def modif(schedule):
    for i in schedule:
        i = i-1
    return schedule
        
#solution1 = modif(solution)           
print_schedule(jobs, solution)                 
                    
                    
                    













    



