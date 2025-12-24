import os
import glob
import numpy as np
import random
import time
import pandas as pd

# ==========================================
# 1. SOP Instance Class
# ==========================================
class SOPInstance:
    def __init__(self, filepath):
        self.filename = os.path.basename(filepath)
        self.dimension = 0
        self.matrix = []
        self.constraints = {} 
        self.dependents = {}  
        self.valid = True
        try:
            self.load_from_file(filepath)
        except Exception as e:
            print(f"Error loading {self.filename}: {e}")
            self.valid = False

    def load_from_file(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            
        lines = content.strip().split('\n')
        matrix_data = []
        reading_weights = False
        
        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith("DIMENSION"):
                parts = line.replace(':', ' ').split()
                try:
                    self.dimension = int(parts[1])
                except IndexError:
                    self.dimension = int(parts[0].split()[-1])
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                reading_weights = True
            elif reading_weights:
                if line.startswith("EOF"):
                    break
                parts = line.split()
                for part in parts:
                    try:
                        matrix_data.append(int(part))
                    except ValueError: pass

        # Handle TSPLIB "Dimension repeated" quirk
        expected_len = self.dimension * self.dimension
        if len(matrix_data) == expected_len + 1:
            matrix_data.pop(0)

        if len(matrix_data) != expected_len:
            if len(matrix_data) > expected_len:
                matrix_data = matrix_data[:expected_len]
            else:
                raise ValueError(f"Data length mismatch: Expected {expected_len}, got {len(matrix_data)}")

        self.matrix = np.array(matrix_data).reshape((self.dimension, self.dimension))
        
        # Initialize sets
        for r in range(self.dimension):
            self.constraints[r] = set()
            self.dependents[r] = set()

        # Constraints Logic
        count_constraints = 0
        for u in range(self.dimension):
            for v in range(self.dimension):
                if self.matrix[u][v] == -1:
                    self.constraints[u].add(v)      
                    self.dependents[v].add(u)       
                    count_constraints += 1
        self.num_constraints = count_constraints

    def get_cost(self, u, v):
        val = self.matrix[u][v]
        return float('inf') if val == -1 else val

# ==========================================
# 2. Algorithm Components
# ==========================================
def generate_initial_population(sop, pop_size):
    """Generates random feasible solutions using Topological Sort."""
    population = []
    attempts = 0
    max_attempts = pop_size * 500 
    
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        
        current_constraints = {n: len(sop.constraints[n]) for n in range(sop.dimension)}
        available = [n for n, c in current_constraints.items() if c == 0]
        tour = []
        
        while available:
            node = random.choice(available)
            available.remove(node)
            tour.append(node)
            
            for dep in sop.dependents[node]:
                current_constraints[dep] -= 1
                if current_constraints[dep] == 0:
                    available.append(dep)
        
        if len(tour) == sop.dimension:
            population.append(tour)
            
    return population

def mpo_crossover(parent1, parent2, sop):
    """Maximum Partial Order / Arbitrary Insertion Crossover."""
    size = sop.dimension
    p2_pos = {node: i for i, node in enumerate(parent2)}
    
    # --- MPO Step ---
    dp = {u: 1 for u in range(size)}
    predecessor = {u: None for u in range(size)}
    
    for i in range(size):
        u = parent1[i]
        for j in range(i):
            v = parent1[j]
            if p2_pos[v] < p2_pos[u]:
                if dp[v] + 1 > dp[u]:
                    dp[u] = dp[v] + 1
                    predecessor[u] = v
                    
    if not dp: return parent1 
    
    curr = max(dp, key=dp.get)
    mpo_tour = []
    while curr is not None:
        mpo_tour.append(curr)
        curr = predecessor[curr]
    mpo_tour.reverse()
    
    # --- AI Step ---
    current_set = set(mpo_tour)
    missing = [n for n in range(size) if n not in current_set]
    random.shuffle(missing)
    final_tour = list(mpo_tour)
    
    for node in missing:
        best_inc = float('inf')
        best_pos = -1
        
        min_idx = 0
        max_idx = len(final_tour)
        
        for c in sop.constraints[node]:
            if c in current_set:
                try: min_idx = max(min_idx, final_tour.index(c) + 1)
                except: pass
        for d in sop.dependents[node]:
            if d in current_set:
                try: max_idx = min(max_idx, final_tour.index(d))
                except: pass
        
        if min_idx <= max_idx:
            for i in range(min_idx, max_idx + 1):
                pred = final_tour[i-1] if i > 0 else None
                succ = final_tour[i] if i < len(final_tour) else None
                
                increase = 0
                possible = True
                
                if pred is not None:
                    c = sop.get_cost(pred, node)
                    if c == float('inf'): possible = False
                    else: increase += c
                
                if succ is not None:
                    c = sop.get_cost(node, succ)
                    if c == float('inf'): possible = False
                    else: increase += c
                    
                if pred is not None and succ is not None:
                    c = sop.get_cost(pred, succ)
                    if c != float('inf'): increase -= c
                
                if possible and increase < best_inc:
                    best_inc = increase
                    best_pos = i
        
        if best_pos != -1:
            final_tour.insert(best_pos, node)
        else:
            final_tour.insert(min_idx, node)
        current_set.add(node)
        
    return final_tour

def calculate_fitness(tour, sop):
    cost = 0
    for i in range(len(tour) - 1):
        c = sop.get_cost(tour[i], tour[i+1])
        if c == float('inf'): return float('inf')
        cost += c
    return cost

# ==========================================
# 3. Main Solver Logic (Updated Output)
# ==========================================
def process_instance(filepath, runs=3):
    sop = SOPInstance(filepath)
    if not sop.valid: return None
    
    print(f"Processing {sop.filename}...", end="", flush=True)
    
    mpo_bests = []
    run_times = []
    
    for r in range(runs):
        start_time = time.time() # Start Timer
        
        # 1. Baseline Generation
        initial_pop = generate_initial_population(sop, pop_size=50)
        
        if not initial_pop:
            print(" [Fail] ", end="", flush=True)
            mpo_bests.append(float('inf'))
            run_times.append(time.time() - start_time)
            continue
            
        best_tour = min(initial_pop, key=lambda t: calculate_fitness(t, sop))
        best_score = calculate_fitness(best_tour, sop)
        
        # 2. GA Optimization
        pop = initial_pop[:]
        
        generations = 30 
        no_improv = 0
        
        for g in range(generations):
            if no_improv > 8: break 
            
            new_pop = [best_tour]
            while len(new_pop) < 50:
                candidates = random.sample(pop, min(5, len(pop)))
                p1 = min(candidates, key=lambda t: calculate_fitness(t, sop))
                candidates = random.sample(pop, min(5, len(pop)))
                p2 = min(candidates, key=lambda t: calculate_fitness(t, sop))
                
                child = mpo_crossover(p1, p2, sop)
                new_pop.append(child)
            
            pop = new_pop
            current_best = min(pop, key=lambda t: calculate_fitness(t, sop))
            current_score = calculate_fitness(current_best, sop)
            
            if current_score < best_score:
                best_score = current_score
                best_tour = current_best
                no_improv = 0
            else:
                no_improv += 1
        
        end_time = time.time() # End Timer
        run_duration = end_time - start_time
        
        mpo_bests.append(best_score)
        run_times.append(run_duration)
        print(".", end="", flush=True)

    # --- Statistics Calculation ---
    valid_scores = [s for s in mpo_bests if s != float('inf')]
    
    if valid_scores:
        best_result = np.min(valid_scores)
        avg_result = np.mean(valid_scores)
        std_dev = np.std(valid_scores)
    else:
        best_result = float('inf')
        avg_result = float('inf')
        std_dev = 0.0
        
    avg_time = np.mean(run_times) if run_times else 0.0
    
    print(f" Done. Best: {int(best_result) if best_result != float('inf') else 'Inf'}")
    
    # Return dictionary matching the table columns
    return {
        "PROB": sop.filename.replace('.sop', ''),
        "Best Result": int(best_result) if best_result != float('inf') else "Inf",
        "Avg. Result": round(avg_result, 1) if avg_result != float('inf') else "Inf",
        "Std.Dev.": round(std_dev, 1),
        "Avg Time (sec)": round(avg_time, 1)
    }

# ==========================================
# 4. Entry Point
# ==========================================
if __name__ == "__main__":
    folder_path = r"D:\Master_BK\AdvanceALGO\archives\problems\sop"
    files = glob.glob(os.path.join(folder_path, "*.sop"))
    
    results = []
    print(f"Found {len(files)} instances in {folder_path}")
    print("-" * 60)
    
    # Sort files to match typical ordering (optional but good for display)
    files.sort()

    for f in files:
        res = process_instance(f, runs=3) # Runs=3 as placeholder, increase for better stats
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns explicitly to match the image table
        cols = ["PROB", "Best Result", "Avg. Result", "Std.Dev.", "Avg Time (sec)"]
        df = df[cols]
        
        print("\n" + "="*80)
        print("MPO/AI+LS RESULTS") # Matching the super-header style
        print("="*80)
        # to_string helps print the whole table nicely
        print(df.to_string(index=False, col_space=12))
        print("="*80)
        
        output_file = "sop_final_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nReport saved to {output_file}")
    else:
        print("\nNo results generated.")