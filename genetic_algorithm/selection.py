# genetic_algorithm/selection.py

"""
selection.py — GA selection operator

Implementing the tournament selection for the Genetic Algorithm in the Student-Project Allocation (SPA) problem.
Uses DEAP’s built-in `selTournament` to choose individuals for the next generation,
balancing exploitation of the best solutions and exploration of diversity.
"""


from deap import tools

def tournament_selection(population, k=3):
    """
    Applies tournament selection to a population.
    
    Selecting the individuals by running tournaments of size k and choosing the best individual
    in each group. This helps in balancing the exploitation (best solutions) and exploration.
    
    Parameters:
        population (list): List of individuals (chromosomes) to select from.
        k (int): Tournament size (default = 3).
    
    Returns:
        list: Selected individuals for the next generation.
    """
    # Use DEAP's built-in tournament selection
    return tools.selTournament(population, len(population), tournsize=k)
