import random
import math
from typing import List, Tuple, Set, Dict
import numpy as np

class LKH:
    def __init__(self, distances: List[List[float]]):
        self.distances = np.array(distances)
        self.n = len(distances)
        self.tour = list(range(self.n))
        random.shuffle(self.tour)
        self.alpha = 0.3  # Parameter for candidate set
        self.candidates = self._generate_candidate_sets()
        self.dont_look_bits = [False] * self.n
        self.pi = self._calculate_pi()

    def _calculate_pi(self) -> List[float]:
        pi = [0] * self.n
        for i in range(self.n):
            pi[i] = max(self.distances[i]) / 2
        return pi

    def _generate_candidate_sets(self) -> List[Set[int]]:
        candidates = [set() for _ in range(self.n)]
        for i in range(self.n):
            sorted_distances = sorted([(j, self.distances[i][j]) for j in range(self.n) if i != j], 
                                      key=lambda x: x[1])
            candidates[i] = set(j for j, _ in sorted_distances[:int(self.alpha * self.n)])
        return candidates

    def _alpha_nearness(self, i: int, j: int) -> float:
        return self.distances[i][j] - self.pi[i] - self.pi[j]

    def total_distance(self, tour: List[int]) -> float:
        return sum(self.distances[tour[i-1]][tour[i]] for i in range(self.n))

    def _gain(self, t1: int, t2: int, t3: int, t4: int) -> float:
        return (self.distances[t1][t2] + self.distances[t3][t4] - 
                self.distances[t1][t3] - self.distances[t2][t4])

    def _find_best_move(self, tour: List[int]) -> Tuple[int, int, float]:
        best_gain = 0
        best_i, best_j = -1, -1
        for i in range(self.n):
            if self.dont_look_bits[tour[i]]:
                continue
            t1, t2 = tour[i-1], tour[i]
            found_improvement = False
            for j in self.candidates[t1]:
                if j == t2 or j == tour[i-2]:
                    continue
                t3, t4 = j, tour[tour.index(j)-1]
                gain = self._gain(t1, t2, t3, t4)
                if gain > best_gain:
                    best_gain = gain
                    best_i, best_j = i, tour.index(j)
                    found_improvement = True
            if not found_improvement:
                self.dont_look_bits[tour[i]] = True
        return best_i, best_j, best_gain

    def _double_bridge_kick(self, tour: List[int]) -> List[int]:
        n = len(tour)
        i, j, k = sorted(random.sample(range(1, n), 3))
        return tour[:i] + tour[j:k] + tour[i:j] + tour[k:]

    def _lin_kernighan_step(self, tour: List[int]) -> List[int]:
        initial_length = self.total_distance(tour)
        current_tour = tour[:]
        
        improvements = 0
        while improvements < 50:  # Limit the number of improvements
            i, j, gain = self._find_best_move(current_tour)
            if gain <= 0:
                break
            if i < j:
                current_tour[i:j] = reversed(current_tour[i:j])
            else:
                current_tour[j:] = reversed(current_tour[j:])
                current_tour[:i] = reversed(current_tour[:i])
            improvements += 1
        
        if self.total_distance(current_tour) < initial_length:
            return current_tour
        return tour

    def _update_pi(self, tour: List[int]):
        for i in range(self.n):
            self.pi[tour[i]] = (self.distances[tour[i-1]][tour[i]] + 
                                self.distances[tour[i]][tour[(i+1)%self.n]]) / 2

    def run(self, max_iterations: int = 100, max_no_improve: int = 100) -> Tuple[List[int], float]:
        best_tour = self.tour[:]
        best_distance = self.total_distance(best_tour)
        no_improve = 0

        for iteration in range(max_iterations):
            self.dont_look_bits = [False] * self.n
            current_tour = self._lin_kernighan_step(self.tour[:])
            current_distance = self.total_distance(current_tour)

            if current_distance < best_distance:
                best_tour = current_tour[:]
                best_distance = current_distance
                no_improve = 0
                self._update_pi(best_tour)
            else:
                no_improve += 1

            if no_improve >= max_no_improve:
                break

            self.tour = self._double_bridge_kick(self.tour)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best distance = {best_distance}")

        return best_tour, best_distance

def generate_random_graph(n: int) -> List[List[float]]:
    return [[random.uniform(1, 100) if i != j else 0 for j in range(n)] for i in range(n)]

# Example usage
n = 50  # number of cities
distances = generate_random_graph(n)

lkh = LKH(distances)
best_tour, best_distance = lkh.run(max_iterations=100000, max_no_improve=10000)

print(f"Best tour length: {best_distance}")
print(f"First 20 cities in the best tour: {best_tour[:20]}...")