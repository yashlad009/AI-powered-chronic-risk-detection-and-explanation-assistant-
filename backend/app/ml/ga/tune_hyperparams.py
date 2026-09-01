"""
Genetic Algorithm (GA) Hyperparameter Optimization.
Optimizes neural network parameters (units, dropout, learning rate) using DEAP.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# DEAP imports for evolutionary algorithm components
from deap import base, creator, tools, algorithms

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Global variables for caching data to avoid reloading for every evaluation
_X_train_scaled = None
_X_val_scaled = None
_y_train = None
_y_val = None

def load_and_prepare_data():
    """
    Loads the BRFSS dataset, performs stratified split, and scales features.
    Caches the results globally to optimize execution speed during GA runs.
    """
    global _X_train_scaled, _X_val_scaled, _y_train, _y_val
    if _X_train_scaled is not None:
        return _X_train_scaled, _X_val_scaled, _y_train, _y_val

    # Resolve dataset path relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    dataset_path = os.path.join(base_dir, "ml-training", "new_dataset", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"BRFSS dataset not found at: {dataset_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # Stratified split to preserve class distribution (80% train / 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    _X_train_scaled = scaler.fit_transform(X_train)
    _X_val_scaled = scaler.transform(X_val)
    _y_train = y_train.values
    _y_val = y_val.values

    return _X_train_scaled, _X_val_scaled, _y_train, _y_val

def evaluate_hyperparameters(dense_units_1, dense_units_2, dropout_rate, learning_rate, epochs=5, batch_size=64):
    """
    Fitness Function:
    Builds a small neural network using the specified hyperparameters,
    trains it briefly, and evaluates performance using Validation ROC-AUC.
    
    Parameters:
    - dense_units_1 (int): Units in the first hidden layer.
    - dense_units_2 (int): Units in the second hidden layer.
    - dropout_rate (float): Dropout probability for regularization.
    - learning_rate (float): Learning rate for the Adam optimizer.
    - epochs (int): Number of training epochs (brief training for speed).
    
    Returns:
    - float: Validation ROC-AUC score (fitness score).
    """
    X_train, X_val, y_train, y_val = load_and_prepare_data()

    # Build model using keras Sequential API with the given gene parameters
    model = Sequential([
        Dense(units=dense_units_1, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate) if dropout_rate > 0 else Dropout(0.0),
        Dense(units=dense_units_2, activation="relu"),
        Dropout(dropout_rate) if dropout_rate > 0 else Dropout(0.0),
        Dense(units=1, activation="sigmoid")
    ])

    # Compile model with the custom learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["AUC"]
    )

    # Brief training with EarlyStopping to optimize speed
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )

    # Predict probabilities on the validation set
    y_pred_probs = model.predict(X_val, verbose=0).flatten()

    # Calculate ROC-AUC score
    val_auc = roc_auc_score(y_val, y_pred_probs)
    return val_auc


# --- DEAP SETUP AND EVOLUTIONARY LOOP ---

# 1. Define Fitness and Individual templates.
# We maximize ROC-AUC, hence weights=(1.0,)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate_individual(individual):
    """
    Evaluates an individual chromosome by decoding its genes and returning
    its fitness score as a tuple (required by DEAP).
    """
    dense_units_1 = int(individual[0])
    dense_units_2 = int(individual[1])
    dropout_rate = float(individual[2])
    learning_rate = float(individual[3])

    # Evaluate using our validated fitness function (5 epochs for speedy evaluation)
    auc = evaluate_hyperparameters(
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        epochs=5
    )
    return (auc,)

def mutate_individual(individual, indpb=0.25):
    """
    Custom mutation operator that respects the specific data types and discrete/continuous
    boundaries of each gene:
    - dense_units_1: randomly pick from [8, 16, 32, 64]
    - dense_units_2: randomly pick from [4, 8, 16, 32]
    - dropout_rate: apply small Gaussian step clipped within [0.0, 0.5]
    - learning_rate: randomly pick from [0.0001, 0.001, 0.01, 0.1]
    """
    if random.random() < indpb:
        individual[0] = random.choice([8, 16, 32, 64])
    if random.random() < indpb:
        individual[1] = random.choice([4, 8, 16, 32])
    if random.random() < indpb:
        # Small continuous change for dropout rate, clipping to valid bounds
        new_val = individual[2] + random.normalvariate(0, 0.1)
        individual[2] = float(np.clip(new_val, 0.0, 0.5))
    if random.random() < indpb:
        individual[3] = random.choice([0.0001, 0.001, 0.01, 0.1])
    return (individual,)

class GeneticOptimizer:
    """Genetic Algorithm runner for ANN hyperparameter optimization."""

    def __init__(self):
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        # Define gene generators
        self.toolbox.register("attr_units_1", random.choice, [8, 16, 32, 64])
        self.toolbox.register("attr_units_2", random.choice, [4, 8, 16, 32])
        self.toolbox.register("attr_dropout", random.uniform, 0.0, 0.5)
        self.toolbox.register("attr_lr", random.choice, [0.0001, 0.001, 0.01, 0.1])

        # Register chromosome structures
        self.toolbox.register(
            "individual", 
            tools.initCycle, 
            creator.Individual, 
            (self.toolbox.attr_units_1, self.toolbox.attr_units_2, self.toolbox.attr_dropout, self.toolbox.attr_lr), 
            n=1
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("evaluate", evaluate_individual)
        # Uniform crossover allows swapping discrete parameter fields effectively
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", mutate_individual)
        # Tournament selection maintains pressure while preventing premature convergence
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize(self, pop_size=10, generations=5, cxpb=0.5, mutpb=0.2):
        """
        Runs the DEAP evolutionary algorithm loop.
        
        Parameters:
        - pop_size (int): Size of the candidate population.
        - generations (int): Number of generations to evolve.
        - cxpb (float): Probability of mating/crossover.
        - mutpb (float): Probability of mutating.
        
        Returns:
        - dict: Best individual parameters found and corresponding fitness score.
        """
        print(f"Initializing population of size {pop_size}...")
        pop = self.toolbox.population(n=pop_size)
        
        # Evaluate the initial population
        print("Evaluating initial population...")
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Keep track of the best individual seen
        hof = tools.HallOfFame(1)
        hof.update(pop)

        # Evolution loop
        for gen in range(1, generations + 1):
            print(f"\n--- Generation {gen} / {generations} ---")
            
            # Select next generation of parents
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover (mating)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring that have changed
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the population with the offspring
            pop[:] = offspring
            hof.update(pop)

            # Print statistics for this generation
            fits = [ind.fitness.values[0] for ind in pop]
            print(f"  Min Fitness: {min(fits):.4f}")
            print(f"  Max Fitness: {max(fits):.4f}")
            print(f"  Avg Fitness: {np.mean(fits):.4f}")
            print(f"  Best individual so far: {hof[0]} (Score: {hof[0].fitness.values[0]:.4f})")

        best_ind = hof[0]
        return {
            "dense_units_1": int(best_ind[0]),
            "dense_units_2": int(best_ind[1]),
            "dropout_rate": float(best_ind[2]),
            "learning_rate": float(best_ind[3]),
            "best_fitness": float(best_ind.fitness.values[0])
        }
