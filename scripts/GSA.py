import numpy as np

class GSA:
    def __init__(self, problem_size, domain_range, objective_function, epoch=100, pop_size=50, initial_agent=None, *args):
        self.problem_size = problem_size
        self.domain_range = domain_range
        self.objective_function = objective_function
        self.epoch = epoch
        self.pop_size = pop_size
        self.pop = np.random.randint(self.domain_range[0], self.domain_range[1] + 1, (self.pop_size, self.problem_size))  # Initialize with integers
        if initial_agent is not None:
            self.pop[0] = initial_agent  # Replace the first agent with the initial agent
        self.fitness = np.zeros(self.pop_size)
        self.mass = np.zeros(self.pop_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.args = args  # Store additional arguments

    def calculate_fitness(self):
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_function(self.pop[i], *self.args)  # Pass the additional arguments
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.pop[i]

    def update_mass(self):
        worst = np.max(self.fitness)
        best = np.min(self.fitness)
        for i in range(self.pop_size):
            self.mass[i] = (self.fitness[i] - worst) / (best - worst + 1e-5)

    def move_agents(self):
        G = 6.67430e-11  # Gravitational constant
        for i in range(self.pop_size):
            force = np.zeros(self.problem_size, dtype=float)
            for j in range(self.pop_size):
                if i != j:
                    distance = np.linalg.norm(self.pop[i] - self.pop[j])
                    force += G * self.mass[i] * self.mass[j] * (self.pop[j] - self.pop[i]) / (distance + 1e-5)
            self.pop[i] = self.pop[i] + force.astype(int)  # Add force and cast to int
            self.pop[i] = np.clip(self.pop[i], self.domain_range[0], self.domain_range[1]).astype(int)  # Ensure within bounds and cast to int

    def solve(self):
        for _ in range(self.epoch):
            self.calculate_fitness()
            self.update_mass()
            self.move_agents()
        return self.best_solution, self.best_fitness
