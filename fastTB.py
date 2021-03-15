import random
from features import compute_fitness
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from preprocess import word_frequencies
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
import re
from preprocess import preprocess_for_article
from preprocess import preprocess_numberOfNNP
import time
import os
from rouge import Rouge
from shutil import copyfile
import pandas as pd 
import math
import multiprocessing

class Summerizer(object):
    def __init__(self, title, sentences, raw_sentences, population_size, max_generation, crossover_rate, mutation_rate, num_picked_sents, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params, MinLT, MaxLT, scheme):
        self.title = title
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.num_objects = len(sentences)
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_picked_sents = num_picked_sents
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns
        self.order_params = order_params
        self.MinLT = MinLT
        self.MaxLT = MaxLT
        self.scheme = scheme


    def generate_population(self, amount):
        # print("Generating population...")
        population = []
        typeA = []
        typeB = []

        
        # for i in range(int(amount/2)):
        for i in range(amount):
            #creat type A
            chromosome1 = np.zeros(self.num_objects)
            chromosome1[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome1 =  chromosome1.tolist()
            fitness1 = compute_fitness(self.title, self.sentences, chromosome1, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            
            chromosome2 = np.zeros(self.num_objects)
            chromosome2[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome2 =  chromosome2.tolist()
            fitness2 = compute_fitness(self.title, self.sentences, chromosome2, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            fitness = max(fitness1, fitness2)
            life_time = 0
            age = 0
            
            typeA.append((chromosome1, chromosome2, fitness, life_time, age))
            
            
            #creat type B
            chromosome3 = np.zeros(self.num_objects)
            chromosome3[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome3 =  chromosome3.tolist()
            fitness3 = compute_fitness(self.title, self.sentences, chromosome3, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            chromosome4 = []
            typeB.append((chromosome3, chromosome4, fitness3, life_time, age))


        population.append(typeA)
        population.append(typeB)
        return population 



    def words_count(self, sentences):
        words = nltk.word_tokenize(sentences)
        return len(words)


    def roulette_select(self, total_fitness, population):
        fitness_slice = np.random.rand() * total_fitness
        fitness_so_far = 0.0
        for phenotype in population:
            fitness_so_far += phenotype[2]
            if fitness_so_far >= fitness_slice:
                return phenotype
        return None


    def rank_select(self, population):
        ps = len(population)
        if ps == 0:
            return None
        population = sorted(population, key=lambda x: x[2], reverse=True)
        fitness_value = []
        for individual in population:
            fitness_value.append(individual[2])
        if len(fitness_value) == 0:
            return None
        fittest_individual = max(fitness_value)
        medium_individual = sta.median(fitness_value)
        selective_pressure = fittest_individual - medium_individual
        j_value = 1
        a_value = np.random.rand()   
        for agent in population:
            if ps == 0:
                return None
            elif ps == 1:
                return agent
            else:
                range_value = selective_pressure - (2*(selective_pressure - 1)*(j_value - 1))/( ps - 1) 
                prb = range_value/ps
                if prb > a_value:
                    return agent
            j_value +=1

    def reduce_no_memes(self, agent, max_sent):
        sum_sent_in_summary = sum(agent)
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent[remove_point] == 1:
                    agent[remove_point] = 0
                    sum_sent_in_summary -=1    
        return agent

    def crossover(self, individual_1, individual_2, max_sent):

        # check tỷ lệ crossover
        if self.num_objects < 2 or random.random() >= self.crossover_rate:
            return individual_1[:], individual_2[:]

        individual_2 = random.choice(individual_2[:2])


        if len(individual_2) == 0:
            fitness1 = compute_fitness(self.title, self.sentences, individual_1[0], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            child1 = (individual_1[0], individual_2, fitness1, 0, 0)
            fitness2 = compute_fitness(self.title, self.sentences, individual_1[1], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            child2 = (individual_1[1], individual_2, fitness2, 0, 0)
            return child1, child2



        individual_1 = random.choice(individual_1[:2])
        
        #tìm điểm chéo 1
        crossover_point = 1 + random.randint(0, self.num_objects - 2)
        agent_1a = individual_1[:crossover_point] + individual_2[crossover_point:]
        agent_1a = self.reduce_no_memes(agent_1a, max_sent)
        fitness_1a = compute_fitness(self.title, self.sentences, agent_1a, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        
        agent_1b = individual_2[:crossover_point] + individual_1[crossover_point:]
        agent_1b = self.reduce_no_memes(agent_1b, max_sent)
        fitness_1b = compute_fitness(self.title, self.sentences, agent_1b, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        
        if fitness_1a > fitness_1b:
            child_1 = (agent_1a, agent_1b, fitness_1a, 0, 0)
        else:
            child_1 = (agent_1a, agent_1b, fitness_1b, 0, 0)

        #tìm điểm chéo 2
        crossover_point_2 = 1 + random.randint(0, self.num_objects - 2)
        
        agent_2a = individual_1[:crossover_point_2] + individual_2[crossover_point_2:]
        agent_2a = self.reduce_no_memes(agent_2a, max_sent)
        fitness_2a = compute_fitness(self.title, self.sentences, agent_2a, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
        
        agent_2b = individual_2[:crossover_point_2] + individual_1[crossover_point_2:]
        agent_2b = self.reduce_no_memes(agent_2b, max_sent)
        fitness_2b = compute_fitness(self.title, self.sentences, agent_2b, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
        
        if fitness_2a > fitness_2b:
            child_2 = (agent_2a, agent_2b, fitness_2a, 0, 0)
        else:
            child_2 = (agent_2a, agent_2b, fitness_2b, 0, 0)        
        
        return child_1, child_2
    

    def mutate(self, individual, max_sent):
        sum_sent_in_summary = sum(individual[0])
        sum_sent_in_summary2 =sum(individual[1])
        if len(individual[1]) == 0:
            self.mutation_rate = 2/self.num_objects
            chromosome = individual[0][:]
            for i in range(len(chromosome)):
                if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                    if chromosome[i] == 0 :
                        chromosome[i] = 1
                        sum_sent_in_summary +=1
                    # else:
                    #     chromosome[i] = 0
                    #     sum_sent_in_summary -=1     
            fitness = compute_fitness(self.title, self.sentences, chromosome , self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            return (chromosome, individual[1], fitness, 0, 0)

        if random.random() < 0.05 :
            chromosome  = random.choice(individual[:2])
            null_chromosome = []
            fitness = compute_fitness(self.title, self.sentences, chromosome , self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            return(chromosome, null_chromosome, fitness, 0, 0) 


        chromosome1 = individual[0][:]
        chromosome2 = individual[1][:]
        self.mutation_rate = 1/self.num_objects

        for i in range(len(chromosome1)):
            if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                if chromosome1[i] == 0 :
                    chromosome1[i] = 1
                    sum_sent_in_summary +=1
                # else:
                #     chromosome1[i] = 0
                #     sum_sent_in_summary -=1 
        
        
        for i in range(len(chromosome2)):
            if random.random() < self.mutation_rate and sum_sent_in_summary2 < max_sent :
                if chromosome2[i] == 0 :
                    chromosome2[i] = 1
                    sum_sent_in_summary2 +=1
                else:
                    chromosome2[i] = 0
                    sum_sent_in_summary2 -=1 
        
        
        fitness1 = compute_fitness(self.title, self.sentences, chromosome1, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        fitness2 = compute_fitness(self.title, self.sentences, chromosome2, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        fitness = max(fitness1, fitness2)
        return (chromosome1, chromosome2, fitness, 0, 0)

    def compare(self, lst1, lst2):
        for i in range(self.num_objects):
            if lst1[i] != lst2[i]:
                return False
        return True

    def survivor_selection (self, individual , population, check, max_sent ):
        if len(population) > 4 :
            competing = random.sample(population, 4)
            lowest_individual = min(competing , key = lambda x: x[2])
            if individual[2] > lowest_individual[2]:
                check = 1
                return individual, check
            elif sum(lowest_individual[0]) <= max_sent:
                check = 1
                return lowest_individual, check
        return individual, check


    def calculate_lifetime(self, fitness, avg_fitness, max_fitness, min_fitness, scheme):
        eta = 1/2*(self.MaxLT - self.MinLT)

        if scheme == 0:
            life_time = min(self.MinLT + int(eta*(fitness/avg_fitness)), self.MaxLT)
        elif scheme == 1:
            try:
                life_time = self.MinLT + int(2*eta*(fitness - min_fitness)/(max_fitness - min_fitness))
            except:
                life_time = 2
        else:
            try:
                if fitness <= avg_fitness:
                    life_time = self.MinLT + int(eta*(fitness - min_fitness)/(avg_fitness - min_fitness))
                else:
                    life_time = int(0.5*(self.MinLT + self.MaxLT) + eta*(fitness - avg_fitness)/(max_fitness - avg_fitness))
            except:
                life_time = 2
        return life_time


    def evaluate_age(self, population, scheme):
        
        fitness_value = []
        for individual in population[0]:
            fitness_value.append(individual[2])
        for individual in population[1]:
            fitness_value.append(individual[2])   
        try:
            avg_fitness = sta.mean(fitness_value)
            max_fitness = max(fitness_value)
            min_fitness = min(fitness_value)
        except: 
            print("bug")
            import pdb; pdb.set_trace()


        new_typeA = []
        new_typeB = []
        #life_time
        for individual in population[0]:
            indiv = list(individual)
            indiv[3] =  self.calculate_lifetime(indiv[2], avg_fitness, max_fitness, min_fitness, scheme)
            indiv[4] += 1
            new_typeA.append(tuple(indiv))

        for individual in population[1]:
            indiv = list(individual)
            indiv[3] =  self.calculate_lifetime(indiv[2], avg_fitness, max_fitness, min_fitness, scheme)
            indiv[4] += 1
            new_typeB.append(tuple(indiv))

            
        population[0] = new_typeA
        population[1] = new_typeB
        return population 

    def check_timelife(self, population):
        count = 0
        new_typeA = []
        new_typeB = []
        for individual in population[0]:
            if individual[3] == individual[4]:
                count +=1
            else:
                new_typeA.append(individual)

        for individual in population[1]:
            if individual[3] == individual[4]:
                count +=1
            else:
                new_typeB.append(individual)

        return count, new_typeA, new_typeB


    def selection(self, population, popsize):
        population = self.evaluate_age(population, self.scheme)
        max_sent = 4
        if len(self.sentences) < 4:
            max_sent = len(self.sentences)
        new_population = []
        new_typeA = []
        new_typeB = []

                
        
        population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
        population[1] = sorted(population[1], key=lambda x: x[2], reverse=True)




        # chosen_agents = int(0.65*len(population))
        chosen_agents_A = int(0.1*len(population[0]))
        chosen_agents_B = int(0.1*len(population[1]))
        
        elitismA = population[0][ : chosen_agents_A ]
        # new_typeA = elitismA

        elitismB = population[1][ : chosen_agents_B]
        # new_typeB = elitismB



        population[0] = population[0][ chosen_agents_A :]
        population[1] = population[1][ chosen_agents_B :]

        
        total_fitness = 0
        for indivi in population[1]:
            total_fitness = total_fitness + indivi[2]  
        
        population_size = popsize
        cpop = 0.0



        #chọn cá thể  loại A bằng rank_selection, cá thể loại B bằng RW
        while cpop <= population_size:
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
            parent_1 = None

            check_time_1 = time.time()
            while parent_1 == None:
                parent_1 = self.rank_select(population[0])
                if parent_1 == None and (time.time() - check_time_1) > 100:
                    try:
                        parent_1 = random.choice(population[0])
                    except:
                        return self.generate_population(population_size), self.population_size
            parent_2 = None

            check_time_2 = time.time()
            while parent_2 == None :
                parent_2 = self.roulette_select(total_fitness, population[1])
                if parent_2 == None and (time.time() - check_time_2) > 100:
                    try:
                        parent_2 =  random.choice(population[1])
                    except:
                        return self.generate_population(population_size), self.population_size
                if parent_2 != None:
                    if self.compare(parent_2[0], parent_1[0]) or self.compare(parent_2[0], parent_1[1]):
                        parent_2 = self.roulette_select(total_fitness, population[1])

            parent_1, parent_2 = copy(parent_1), copy(parent_2)
            child_1, child_2 = self.crossover(parent_1, parent_2, max_sent)

            check1 = 0
            check2 = 0

            # child_1
            individual_X = self.mutate(child_1, max_sent)

            #Nếu X loại B:
            if len(individual_X[1]) == 0:
                individual_X , check1 = self.survivor_selection(individual_X, population[1], check1, max_sent)
                if check1 == 1:
                    new_typeB.append(individual_X)
            else:
                individual_X , check1 = self.survivor_selection(individual_X, population[0], check1, max_sent)
                if check1 == 1:
                    new_typeA.append(individual_X)

            # child_2
            individual_Y = self.mutate(child_2, max_sent)

            #Nếu Y loại B:
            if len(individual_Y[1]) == 0:
                individual_Y , check1 = self.survivor_selection(individual_Y, population[1], check2, max_sent)
                if check2 == 1:
                    new_typeB.append(individual_Y)
            else:
                individual_Y , check1 = self.survivor_selection(individual_Y, population[0], check2, max_sent)
                if check2 == 1:
                    new_typeA.append(individual_Y)

            if check1 + check2 == 0:
                cpop += 0.01
            else:
                cpop += check1 + check2

        # try:
        #     new_size = min(len(new_typeA), len(new_typeB))
        #     new_population.append(new_typeA)
        #     new_population.append(new_typeB)
        #     self.evaluate_age(new_population)
        # except: 
        #     new_size = 0


        new_size = min(len(new_typeA), len(new_typeB))

        if new_size != 0 :
            new_population.append(new_typeA)
            new_population.append(new_typeB)
            new_population = self.evaluate_age(new_population, self.scheme)
        else:
            new_population = self.generate_population(self.population_size)
            new_population = self.evaluate_age(new_population, self.scheme)

        Dsize , typeA_current_population, typeB_current_population = self.check_timelife(population)

        new_population[0].extend(elitismA)
        new_population[0].extend(typeA_current_population)

        new_population[1].extend(elitismB)
        new_population[1].extend(typeB_current_population)


        new_popsize = popsize + new_size - Dsize
        if new_size > 100:
            new_size = 100        

        if len(new_population[0]) == 0 or len(new_population[1]) == 0:
            print("bug")
            import pdb; pdb.set_trace()
            print("wait...")


        fitness_value = []
        for individual in new_population[0]:
            fitness_value.append(individual[2])
        for individual in new_population[1]:
            fitness_value.append(individual[2])   

        try:
            avg_fitness = sta.mean(fitness_value)
        except: 
            return self.generate_population(population_size), self.population_size

        agents_in_Ev = [] 

        for agent in new_population[0]:
            if (agent[2] > 0.95*avg_fitness) and (agent[2] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)
        for agent in new_population[1]:
            if (agent[2] > 0.95*avg_fitness) and (agent[2] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)


        if len(agents_in_Ev) >= new_popsize*0.9 :
            new_popsize = self.population_size
            widen_population = self.generate_population(int(new_popsize*0.7))
            chosen = new_popsize - int(new_popsize*0.7)

            type_A = widen_population[0]
            type_B = widen_population[1]

            new_typeA = sorted(new_population[0], key=lambda x: x[2], reverse=True)
            new_typeB = sorted(new_population[1], key=lambda x: x[2], reverse=True)
            new_typeA = new_typeA[ : chosen]
            new_typeB = new_typeB[ : chosen]


            for x in new_typeA:
                type_A.append(x)
            for y in new_typeB:
                type_B.append(y)
            new_population = []
            new_population.append(type_A)
            new_population.append(type_B)


        return new_population, new_popsize
    

    def find_best_individual(self, population):
        if len(population) == 0:
            population = self.generate_population(self.generate_population)
        

        try:
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
            population[1] = sorted(population[1], key=lambda x: x[2], reverse=True)
            best_type_A = population[0][0]
            best_type_B = population[1][0]
        except:
            population = self.generate_population(self.population_size)
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
            population[1] = sorted(population[1], key=lambda x: x[2], reverse=True)
            best_type_A = population[0][0]
            best_type_B = population[1][0]

            

        if best_type_A[2] > best_type_B[2]:
            fitness1 = compute_fitness(self.title, self.sentences, best_type_A[0], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns,self.order_params)
            fitness2 = compute_fitness(self.title, self.sentences, best_type_A[1], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            if fitness1 >= fitness2:
                return (best_type_A[0], fitness1)
            else:
                return (best_type_A[1], fitness2)
        return (best_type_B[0], best_type_B[2])
 

   #MASingleDocSum    
    def solve(self):
        population = self.generate_population(self.population_size)
        popsize = self.population_size
        for i in tqdm(range(self.max_generation)):
            population, popsize = self.selection(population, popsize)
        return self.find_best_individual(population)
    
    
    def show(self, individual,  file):
        index = individual[0]
        f = open(file,'w', encoding='utf-8')
        for i in range(len(index)):
            if index[i] == 1:
                f.write(self.raw_sentences[i] + '\n')
        f.close()

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs


def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 5:
        return 'None'
    return cleaned


def start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, order_params, scheme):
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split(" . ", example[0]) 
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue
       
        # print('raw', len(raw_sentences), stories.index(example))
        title_raw = raw_sentences[0]

        sentences = []
        sentences_for_NNP = []
        for raw_sent in raw_sentences:
            sent = preprocess_raw_sent(raw_sent)
            # sent_tmp = preprocess_numberOfNNP(raw_sent)
            
            sent_tmp = preprocess_raw_sent(raw_sent, True)
            if len(sent.split(' ')) < 2:
                raw_sentences.remove(raw_sent)
            else:
                sentences.append(sent)
                sentences_for_NNP.append(sent_tmp)
        title = preprocess_raw_sent(title_raw)
        list_sentences_frequencies = word_frequencies(sentences, title)
        number_of_nouns = count_noun(sentences_for_NNP)
        simWithTitle = sim_with_title(list_sentences_frequencies)
        sim2sents = sim_2_sent(list_sentences_frequencies)
        simWithDoc = []
        # for sent in sentences:
        for i in range(len(sentences)):
            simWithDoc.append(sim_with_doc(list_sentences_frequencies, index_sentence=i))



        # POPU_SIZE = 40
        if len(sentences) < 20:
            MAX_GEN = 20
        elif len(sentences) < 50:
            MAX_GEN = 50
        else:
            MAX_GEN = 80


        print("POPULATION SIZE: {}".format(POPU_SIZE))
        print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
          
        print("Done preprocessing!")
        # DONE!
        print('time for processing', time.time() - start_time)
        if len(sentences) < 4:
            NUM_PICKED_SENTS = len(sentences)
        else:
            NUM_PICKED_SENTS = 4
        
        MinLT = 1
        MaxLT = 7 
            
        Solver = Summerizer(title, sentences, raw_sentences, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, NUM_PICKED_SENTS, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params, MinLT, MaxLT, scheme)
        best_individual = Solver.solve()
        file_name = os.path.join(save_path, example[1] )         

        if best_individual is None:
            print ('No solution.')
        else:
            print(file_name)
            print(best_individual)
            Solver.show(best_individual, file_name)


def a_process_do(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, next_part):
        rouge_score = []
        for i in range(3): 
            #chạy từng bộ
            start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, 0,  i) 
            #tính rouge của từng bộ
            rouge1, rouge2, rougeL = evaluate_rouge(save_path)
            rouge_score.append((i,rouge1, rouge2, rougeL))
        scheme_had_max_value = max(rouge_score, key = lambda i : i[1])[0]

        result_file = '{}.{}'.format(processID, 'txt')
        fp = open(result_file,'w', encoding='utf-8')
        fp.write('\n'.join('{} , {} , {} , {} '.format(x[0],x[1], x[2], x[3]) for x in rouge_score))
       

        save_path_for_valid = 'hyp'
        start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, next_part, save_path_for_valid, 0, scheme_had_max_value)
       

def multiprocess(num_process, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path, order_params, scheme ):
    processes = []

    # num_docs_per_loop = math.floor(len(stories)/5)
    n = math.floor(len(stories)/5)
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 

    for index, sub_stories in enumerate(set_of_docs):
        p = multiprocessing.Process(target=start_run, args=(
            index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[index], order_params, scheme))
        processes.append(p)
        p.start()

        
    for p in processes:
        p.join()


def evaluate_rouge(hyp_path):
    hyp = hyp_path
    raw_ref = 'abstracts'
    FJoin = os.path.join
    files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(hyp)]
    
    f_hyp = []
    f_raw_ref = []
    print("number of document: ", len(files_hyp))
    for file in files_hyp:
        f = open(file)
        f_hyp.append(f.read())
        f.close()
    for file in files_raw_ref:
        f = open(file)
        f_raw_ref.append(f.read())
        f.close()
        
    rouge_1_tmp = []
    rouge_2_tmp = []
    rouge_L_tmp = []
    for hyp, ref in zip(f_hyp, f_raw_ref):
        try:
            rouge = Rouge()
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge_1 = scores["rouge-1"]["r"]
            rouge_2 = scores["rouge-2"]["r"]
            rouge_L = scores["rouge-l"]["r"]
            rouge_1_tmp.append(rouge_1)
            rouge_2_tmp.append(rouge_2)
            rouge_L_tmp.append(rouge_L)
        except Exception:
            pass
        # print(scores)
    rouge_1_avg = sta.mean(rouge_1_tmp)
    rouge_2_avg = sta.mean(rouge_2_tmp)
    rouge_L_avg = sta.mean(rouge_L_tmp)
    print('Rouge-1: ', rouge_1_avg)
    print('Rouge-2: ',rouge_2_avg )
    print('Rouge-L: ', rouge_L_avg)

    for path in os.listdir(hyp_path):
        full_path = os.path.join(hyp_path, path)
        os.remove(full_path)

    return rouge_1_avg, rouge_2_avg, rouge_L_avg        
            


def main():
    # Setting Variables
    POPU_SIZE = 40
    MAX_GEN = 20
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4
    #NUM_PICKED_SENTS = 4

    directory = 'stories'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5', 'hyp6']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')
    # if not os.path.exists('hyp6'):
    #     os.makedirs('hyp6')


    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()

    
    order_params = 0 #chon bo tham so feature
    scheme = 0 #chon option 1 2 3 de thay doi
    
    multiprocess(5, POPU_SIZE, MAX_GEN, CROSS_RATE,
                 MUTATE_RATE, stories, save_path, order_params, scheme)
    # start_run(1, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path[0], 0, 0)


    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()              
     
     
    


    
    
    
    
        
            
            
         
