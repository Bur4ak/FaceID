import pygame
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

# === PARAMETRELER ===
WIDTH, HEIGHT = 800, 600
POP_SIZE = 150
LIFESPAN = 300
MUTATION_RATE = 0.02

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🧬 Genetik Algoritma – Roketler")
clock = pygame.time.Clock()

# === HEDEF VE ENGEL ===
target = pygame.Vector2(WIDTH // 2, 50)
obstacle = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2, 200, 20)
dragging = False
offset_x = 0
offset_y = 0

# === ROKET SINIFI ===
class Rocket:
    def __init__(self, dna=None):
        self.pos = pygame.Vector2(WIDTH / 2, HEIGHT - 10)
        self.vel = pygame.Vector2()
        self.acc = pygame.Vector2()
        self.completed = False
        self.crashed = False
        self.step = 0
        self.fitness = 0
        self.dna = dna or [pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(LIFESPAN)]

    def apply_force(self, force):
        self.acc += force

    def update(self, step):
        if not self.completed and not self.crashed:
            if step < LIFESPAN:
                self.apply_force(self.dna[step])
                self.step = step
            self.vel += self.acc
            self.pos += self.vel
            self.acc *= 0

            if (target - self.pos).length() < 10:
                self.completed = True
                self.pos = target
            if obstacle.collidepoint(self.pos.x, self.pos.y):
                self.crashed = True
            if not (0 < self.pos.x < WIDTH and 0 < self.pos.y < HEIGHT):
                self.crashed = True

    def calc_fitness(self):
        d = (target - self.pos).length()
        self.fitness = 1 / (d + 1)
        if self.completed:
            self.fitness *= 10 + (LIFESPAN - self.step) / 10
        if self.crashed:
            self.fitness /= 10

    def show(self, screen):
        color = (0, 255, 0) if self.completed else (255, 0, 0) if self.crashed else (255, 255, 255)
        pygame.draw.circle(screen, color, (int(self.pos.x), int(self.pos.y)), 3)

# === POPÜLASYON SINIFI ===
class Population:
    def __init__(self):
        self.rockets = [Rocket() for _ in range(POP_SIZE)]
        self.mating_pool = []
        self.generations = 0
        self.max_fitness = 0
        self.avg_fitness = 0

    def evaluate(self):
        max_fit = 0
        total_fit = 0
        for rocket in self.rockets:
            rocket.calc_fitness()
            total_fit += rocket.fitness
            if rocket.fitness > max_fit:
                max_fit = rocket.fitness
        self.max_fitness = max_fit
        self.avg_fitness = total_fit / POP_SIZE
        self.mating_pool = []
        for rocket in self.rockets:
            n = int((rocket.fitness / max_fit) * 100)
            self.mating_pool += [rocket] * n

    def selection(self):
        new_rockets = []
        for _ in range(POP_SIZE):
            parent_a = random.choice(self.mating_pool).dna
            parent_b = random.choice(self.mating_pool).dna
            mid = random.randint(0, LIFESPAN - 1)
            child_dna = parent_a[:mid] + parent_b[mid:]
            # Mutasyon
            for i in range(LIFESPAN):
                if random.random() < MUTATION_RATE:
                    child_dna[i] = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            new_rockets.append(Rocket(child_dna))
        self.rockets = new_rockets
        self.generations += 1

# === GRAFİK ===
fitness_data = {"gen": [], "max": [], "avg": []}
def plot_thread():
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        if fitness_data["gen"]:
            ax.clear()
            ax.plot(fitness_data["gen"], fitness_data["max"], label="Max Fitness")
            ax.plot(fitness_data["gen"], fitness_data["avg"], label="Avg Fitness")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.legend()
            plt.pause(0.1)

Thread(target=plot_thread, daemon=True).start()

# === ANA DÖNGÜ ===
population = Population()
count = 0
running = True

while running:
    clock.tick(60)
    screen.fill((20, 20, 30))

    # === EVENTLER ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if obstacle.collidepoint(event.pos):
                dragging = True
                offset_x = obstacle.x - event.pos[0]
                offset_y = obstacle.y - event.pos[1]

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                obstacle.x = event.pos[0] + offset_x
                obstacle.y = event.pos[1] + offset_y

    # === ROKETLERİ GÜNCELLE ===
    for rocket in population.rockets:
        rocket.update(count)
        rocket.show(screen)

    # === HEDEF VE ENGEL ===
    pygame.draw.circle(screen, (0, 255, 0), (int(target.x), int(target.y)), 10)
    pygame.draw.rect(screen, (255, 255, 0), obstacle)

    count += 1
    if count == LIFESPAN:
        population.evaluate()
        fitness_data["gen"].append(population.generations)
        fitness_data["max"].append(population.max_fitness)
        fitness_data["avg"].append(population.avg_fitness)
        population.selection()
        count = 0

    pygame.display.flip()

pygame.quit()
