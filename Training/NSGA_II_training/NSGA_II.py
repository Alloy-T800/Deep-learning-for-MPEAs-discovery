from deap import base, creator, tools, algorithms
from src.GAN_model.GAN_achi import GANLoader
from src.AFN_model.AFN_model_archi import ModelLoader
from src.data_processing.minmax_scaler import MinMax
from src.genetic_algorithm.composition_check import NormalizeAndCorrect
from src.material_descriptor.calculate_descriptors import DescriptorsCalculator
import numpy as np
import pandas as pd
import torch
import json
import random
import os

# Load Configuration File
with open('config.json', 'r') as f:
    config = json.load(f)

save_path = config["moga_results"]

# Checking GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load normalization parameters
scaler = MinMax()
feature_scaler = scaler.transform_X
target_scaler = scaler.transform_y

# Load the trained model
loader = ModelLoader()
loader.load_model("AFN.pth")
AFN_model = loader.get_model()
columns = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Al', 'Cu', 'Ti', 'Zr',
           'Nb', 'V', 'Mo', 'Hf', 'Ta', 'Si', 'W', 'Target_1', 'Target_2']

# Load GAN
gan = GANLoader(config["GAN_generator.pth"], config['GAN_discriminator.pth'])

# Load Descriptor Calculation
desc_cal = DescriptorsCalculator()

# Creating multi-objective optimization fitness classes
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize the correction function
correct = NormalizeAndCorrect()

# crossover
def crossover(ind1, ind2):
    tools.cxBlend(ind1, ind2, alpha=0.5)
    # Applying the correction function
    ind1 = correct.adjust(ind1)
    ind2 = correct.adjust(ind2)
    return ind1, ind2

# mutate
def mutate(individual):
    mutation_point = random.randint(0, len(individual) - 1)
    individual[mutation_point] = random.uniform(0, 30)
    # Applying the correction function
    individual = correct.adjust(individual)
    return individual,

# print function
def print_status(population, generation):
    for i, ind in enumerate(population):
        features = ind
        ga_pre1, ga_pre2 = fitness_function(ind)
        print(f"features: {features}")
        print(f"No.{generation},YS_fitness：{ga_pre1:.2f},Elo_fitness：{ga_pre2:.2f}")

# Define functions to compute descriptors and splices
def compute_and_concatenate(individual, desc_calculator):
    # Compute descriptor
    descriptors_dict = desc_calculator.compute_descriptors(columns[:16], individual)

    # Extracting Descriptor Values from a Dictionary
    descriptors_values = list(descriptors_dict.values())

    # Splicing individual and descriptor
    concatenated = individual + descriptors_values

    return concatenated

# Define the fitness function
def fitness_function(individual):

    # Splicing features and descriptors
    feature_with_descri = compute_and_concatenate(individual, desc_cal)
    # normalized characteristic
    feature_with_descri_reshaped = np.array(feature_with_descri).reshape(1, -1)
    normalized_features = feature_scaler(feature_with_descri_reshaped)
    # Converting data to PyTorch tensor
    normalized_features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    # Prediction using model
    normalized_predictions = AFN_model(normalized_features_tensor)

    # Use the discriminator to calculate the probability that the input data is true
    discriminator_output = gan.discriminate(normalized_features_tensor[:, :16]).item()

    # Converting predictions to numpy arrays and back-normalizing them
    normalized_predictions_np = normalized_predictions.detach().numpy()
    normalized_predictions_reverse = scaler.inverse_transform_nor_y(normalized_predictions_np)
    prediction1 = normalized_predictions_reverse[0, 0]
    prediction2 = normalized_predictions_reverse[0, 1]

    weighted_prediction1 = prediction1 * discriminator_output
    weighted_prediction2 = prediction2 * discriminator_output

    return weighted_prediction1, weighted_prediction2


# Functions to run NSGA-II
def run_nsga2(population_size, n_generations):

    # Toolbox Setup
    toolbox = base.Toolbox()

    # GAN generates the initial population
    generated_data = gan.generate_data(population_size) * 100
    generated_data = generated_data.detach().numpy()
    data_iterator = iter(generated_data)
    def create_individual_from_gan():
        ind = next(data_iterator)
        ind = creator.Individual(ind)
        ind = correct.adjust(ind)
        return ind
    num_gan_data = population_size
    generated_data = []
    if num_gan_data > 0:
        generated_data = gan.generate_data(num_gan_data) * 100
        generated_data = generated_data.detach().numpy().tolist()

    data_iterator = iter(generated_data)

    # Registering new individual creation functions to the toolbox
    toolbox.register("individual", create_individual_from_gan)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", lambda ind: fitness_function(ind)[:2])
    population = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    all_results = []  # Used to preserve all individuals of all populations

    # Running Algorithms
    for generation in range(n_generations):
        algorithms.eaMuPlusLambda(population, toolbox, mu=population_size,
                                  lambda_=population_size, cxpb=0.85, mutpb=0.15,
                                  ngen=1, stats=stats, halloffame=None, verbose=True)

        print_status(population, generation + 1)  # Print current status

        # Add results to all_results
        for i, ind in enumerate(population):
            pred1, pred2 = fitness_function(ind)
            all_results.append(ind + [pred1, pred2])

        # Updating the DataFrame
        all_results_df = pd.DataFrame(all_results, columns=columns)

        # Save data
        with pd.ExcelWriter('results.xlsx', engine='openpyxl') as excel_writer:
            all_results_df.to_excel(excel_writer, index=False)


    # Save the Pareto-optimal solution
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_results = []

    for ind in pareto_front:
        pred1, pred2 = fitness_function(ind)
        pareto_results.append(ind + [pred1, pred2])

    return all_results, pareto_results

# Running
population_size = config["population_size"]
n_generations = config["n_generations"]
all_results, pareto_results = run_nsga2(population_size, n_generations)

# Save all results
all_results_excel_path = os.path.join(save_path, 'All_results.xlsx')
all_results_df = pd.DataFrame(all_results, columns=columns)
all_results_df.to_excel(all_results_excel_path, index=False)

# Save the Pareto-optimization solution
pareto_results_excel_path = os.path.join(save_path, 'pareto_results.xlsx')
pareto_results_df = pd.DataFrame(pareto_results, columns=columns)
pareto_results_df.to_excel(pareto_results_excel_path, index=False)




