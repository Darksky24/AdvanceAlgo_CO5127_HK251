import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import copy
import re
import matplotlib.cm as cm
import pandas as pd
import contextlib
import sys
import time
# ==========================================
# 1. SOP Instance (Fixed Parsing & Logic)
# ==========================================
class SOPInstance:
    def __init__(self, filepath):
        print(f"Loading problem from {filepath}...")
        self.problem = tsplib95.load(filepath)
        
        # Dimension
        # tsplib95 might return nodes as 1-based. We standardize to 0..N-1.
        self.nodes = list(self.problem.get_nodes())
        print(self.nodes)
        self.n = len(self.nodes)
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        self.rev_map = {i: node for i, node in enumerate(self.nodes)}

        # Matrices
        self.dist_matrix = np.zeros((self.n, self.n), dtype=float)
        self.prec_matrix = np.zeros((self.n, self.n), dtype=bool)
        
        self._load_weights_robust()
        self._compute_transitive_closure()
        self._build_adjacency()
  
    def _load_weights_robust(self):
        """
        Robustly loads weights handling potential artifacts in SOP files
        (like dimension number appearing in weight section).
        """
        # Try to get explicit weights first
        edge_weights = self.problem.edge_weights
        flat_weights = []
        
        if edge_weights:
            # Flatten list of lists if necessary
            if isinstance(edge_weights[0], list):
                for row in edge_weights: flat_weights.extend(row)
            else:
                flat_weights = edge_weights
                
            # FIX: Check for artifact (First number == Dimension)
            # Some SOP files start with the dimension in the data block
            if len(flat_weights) == (self.n * self.n) + 1:
                if flat_weights[0] == self.n:
                    print("Detected dimension header in weights. Dropping first element.")
                    flat_weights = flat_weights[1:]
            
            if len(flat_weights) != self.n * self.n:
                print(f"Warning: Weight count {len(flat_weights)} does not match N*N ({self.n*self.n}). Parsing might be incorrect.")

            # Populate Matrices
            for i in range(self.n):
                for j in range(self.n):
                    idx = i * self.n + j
                    if idx < len(flat_weights):
                        w = flat_weights[idx]
                        self._process_weight(i, j, w)
        else:
            # Fallback to get_weight
            for i in range(self.n):
                for j in range(self.n):
                    u, v = self.rev_map[i], self.rev_map[j]
                    try:
                        w = self.problem.get_weight(u, v)
                        self._process_weight(i, j, w)
                    except:
                        self.dist_matrix[i][j] = 999999

    def _process_weight(self, i, j, w):
        if w == -1:
            self.dist_matrix[i][j] = float('inf')
            # FIX: SOP Standard: Matrix[i][j] = -1 implies j must precede i
            # i.e., You cannot go i -> j directly because j must be done first.
            # So Precedence: j < i.
            self.prec_matrix[j][i] = True 
        else:
            self.dist_matrix[i][j] = w

    def _compute_transitive_closure(self):
        # Floyd-Warshall for precedence
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.prec_matrix[i][k] and self.prec_matrix[k][j]:
                        self.prec_matrix[i][j] = True
        
    def _build_adjacency(self):
        self.successors = [[] for _ in range(self.n)]
        self.predecessors = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if self.prec_matrix[i][j]: # i precedes j
                    self.successors[i].append(j)
                    self.predecessors[j].append(i)
    
class Solution:
    def __init__(self, sequence, length=0):
        self.sequence = sequence
        self.length = length

# ==========================================
# 2. Local Search (SOP-3-Exchange)
# ==========================================
class SOP3Exchange:
    def __init__(self, instance):
        self.instance = instance
        self.n = instance.n
        self.f_mark = np.full(self.n, -1, dtype=int)
        self.count_h = 0
        self.stack = []
        self.in_stack = np.full(self.n, False, dtype=bool)

    def push(self, node):
        if not self.in_stack[node]:
            self.stack.append(node)
            self.in_stack[node] = True

    def pop(self):
        if not self.stack: return None
        node = self.stack.pop()
        self.in_stack[node] = False
        return node

    def run(self, solution):
        # Safety check for invalid solutions (like the one causing your error)
        if len(solution.sequence) < 3:
            return solution

        self.stack = []
        self.in_stack[:] = False
        for node in solution.sequence: self.push(node)
        
        seq = np.array(solution.sequence, dtype=int)
        pos = np.zeros(self.n, dtype=int)
        for i, node in enumerate(seq): pos[node] = i
        best_len = solution.length
        
        while self.stack:
            h_node = self.pop()
            if h_node is None: break
            
            # Ensure h_node is in valid range for a 3-exchange
            if pos[h_node] >= self.n - 3: continue
            
            idx_h = pos[h_node]
            self.count_h += 1
            improved = False
            
            for idx_i in range(idx_h + 1, self.n - 1):
                node_i = seq[idx_i]
                # Label successors of path_left
                for succ in self.instance.successors[node_i]:
                    self.f_mark[succ] = self.count_h
                
                for idx_j in range(idx_i + 1, self.n - 1):
                    node_j = seq[idx_j]
                    # Feasibility Check
                    if self.f_mark[node_j] == self.count_h: break 
                    
                    n_h, n_h1 = seq[idx_h], seq[idx_h+1]
                    n_i, n_i1 = seq[idx_i], seq[idx_i+1]
                    n_j, n_j1 = seq[idx_j], seq[idx_j+1]
                    
                    d = self.instance.dist_matrix
                    gain = (d[n_h, n_h1] + d[n_i, n_i1] + d[n_j, n_j1]) - \
                           (d[n_h, n_i1] + d[n_j, n_h1] + d[n_i, n_j1])
                    
                    if gain > 1e-6:
                        # Perform Swap
                        seg_left = seq[idx_h+1 : idx_i+1].copy()
                        seg_right = seq[idx_i+1 : idx_j+1].copy()
                        len_r = len(seg_right)
                        seq[idx_h+1 : idx_h+1+len_r] = seg_right
                        seq[idx_h+1+len_r : idx_j+1] = seg_left
                        
                        best_len -= gain
                        improved = True
                        # Update positions
                        for k in range(idx_h + 1, idx_j + 1): pos[seq[k]] = k
                        # Push boundary nodes to stack
                        for node in [n_h, n_h1, n_i, n_i1, n_j, n_j1]: self.push(node)
                        break
                if improved: break
        
        solution.sequence = seq.tolist()
        solution.length = best_len
        return solution

# ==========================================
# 3. HAS-SOP Solver
# ==========================================
class HASSOP:
    def __init__(self, instance, n_ants=10):
        self.instance = instance
        self.n_ants = n_ants
        self.local_search = SOP3Exchange(self.instance)
        self.beta = 2.0
        self.rho = 0.1
        self.phi = 0.1
        self.q0 = 0.9
        
        # Initial Pheromone
        init_sol = self.greedy_solution()
        if len(init_sol.sequence) < self.instance.n:
            print("Warning: Greedy solution failed to find full path. Pheromone init might be poor.")
            self.L_nn = 100000
        else:
            self.L_nn = init_sol.length
            
        self.tau0 = 1.0 / (self.instance.n * self.L_nn)
        self.tau = np.full((self.instance.n, self.instance.n), self.tau0)
        self.best_solution = None

    def greedy_solution(self):
        # Basic construction without pheromone
        return self._build_ant(use_pheromone=False)

    def construct_solution(self):
        return self._build_ant(use_pheromone=True)

    def _build_ant(self, use_pheromone=True):
        n = self.instance.n
        visited = np.full(n, False, dtype=bool)
        # SOP usually implies 0 is start. 
        # If 0 depends on others, this will fail. We assume valid Topo Sort starts at 0.
        sequence = [0] 
        visited[0] = True
        current_len = 0
        current_node = 0
        
        # We need to fill N-1 more nodes
        for _ in range(n - 1):
            # Identify candidates (Unvisited nodes whose predecessors are ALL visited)
            candidates = []
            for cand in range(n):
                if not visited[cand]:
                    preds = self.instance.predecessors[cand]
                    if not preds: # No constraints
                        candidates.append(cand)
                    else:
                        # Check if all preds are visited
                        if all(visited[p] for p in preds):
                            candidates.append(cand)
            
            if not candidates:
                # Deadlock detected
                break
            
            # Selection Rule
            next_node = -1
            if not use_pheromone:
                # Pure Greedy (Nearest Neighbor)
                # Filter out infinite distances
                valid_cands = [c for c in candidates if self.instance.dist_matrix[current_node][c] < float('inf')]
                if not valid_cands: valid_cands = candidates # Fallback
                next_node = min(valid_cands, key=lambda x: self.instance.dist_matrix[current_node][x])
            else:
                # ACS Rule
                heuristic = lambda i, j: 1.0 / self.instance.dist_matrix[i][j] if self.instance.dist_matrix[i][j] > 1e-4 else 10000
                
                vals = []
                for c in candidates:
                    tau = self.tau[current_node][c]
                    eta = heuristic(current_node, c)
                    vals.append(tau * (eta ** self.beta))
                
                if np.random.random() < self.q0:
                    next_node = candidates[np.argmax(vals)]
                else:
                    s_val = sum(vals)
                    if s_val == 0: 
                        probs = [1.0/len(candidates)] * len(candidates)
                    else:
                        probs = [v/s_val for v in vals]
                    next_node = np.random.choice(candidates, p=probs)
            
            # Local Pheromone Update
            if use_pheromone:
                self.tau[current_node][next_node] = (1-self.phi)*self.tau[current_node][next_node] + self.phi*self.tau0
            
            current_len += self.instance.dist_matrix[current_node][next_node]
            sequence.append(next_node)
            visited[next_node] = True
            current_node = next_node
            
        return Solution(sequence, current_len)

    def global_update(self, sol):
        reward = 1.0 / sol.length
        seq = sol.sequence
        for k in range(len(seq)-1):
            i, j = seq[k], seq[k+1]
            self.tau[i][j] = (1-self.rho)*self.tau[i][j] + self.rho*reward
    
    def solve(self, max_iterations=50, early_stop_rounds=15):
        no_improve = 0
        
        print(f"--- Iteration Logs ---")
        
        for it in range(max_iterations):
            ants_sols = []
            for _ in range(self.n_ants):
                sol = self.construct_solution()
                # Safety check: Only optimize valid solutions
                if len(sol.sequence) == self.instance.n:
                    ants_sols.append(sol)
            
            if not ants_sols:
                print(f"Iter {it}: All ants deadlocked. Checking constraints...")
                continue

            # Local Search
            for sol in ants_sols:
                self.local_search.run(sol)
            
            ants_sols.sort(key=lambda x: x.length)
            current_best = ants_sols[0]
            
            if self.best_solution is None or current_best.length < self.best_solution.length:
                self.best_solution = copy.deepcopy(current_best)
                no_improve = 0
            else:
                no_improve += 1
            
            self.global_update(self.best_solution)
            
            if it == 0 or it % 5 == 0 or it == max_iterations - 1:
                 print(f"Iteration-{it}: Best Distance = {self.best_solution.length:.4f}")
            
            if no_improve >= early_stop_rounds:
                print(f"Terminating early at iteration {it} due to no improvement.")
                break
                
        return self.best_solution



def print_run_sequence(instance, solution, run_index):
    """
    Prints the formatted sequence of nodes for a specific run.
    Uses rev_map to show original Node IDs (e.g. 1-based IDs from file).
    """
    if not solution:
        print(f"Run {run_index} did not return a valid solution.")
        return

    # Convert internal indices (0..N-1) back to original file IDs
    # This is crucial because SOP files usually use 1-based indexing
    original_node_ids = [str(instance.rev_map[node_idx]) for node_idx in solution.sequence]
    
    # Create a string with arrows
    sequence_str = " -> ".join(original_node_ids)
    
    print(f"\n" + "="*40)
    print(f"RUN {run_index} COMPLETED")
    print(f"Final Cost: {solution.length:.4f}")
    print(f"Sequence ({len(original_node_ids)} nodes):")
    print(sequence_str)
    print("="*40 + "\n")

def parse_sop_file(filepath):
    """
    Parses a SOP file to extract the dimension and the weight matrix.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Find Dimension
    dim_match = re.search(r'DIMENSION\s*:\s*(\d+)', content)
    if not dim_match:
        raise ValueError("Could not find DIMENSION in file.")
    dimension = int(dim_match.group(1))

    # Find Edge Weight Section
    # The section usually starts after 'EDGE_WEIGHT_SECTION'
    if 'EDGE_WEIGHT_SECTION' not in content:
         raise ValueError("Could not find EDGE_WEIGHT_SECTION in file.")
         
    header, data_section = content.split('EDGE_WEIGHT_SECTION', 1)
    
    # Clean up data section to get all numbers
    # We treat any whitespace/newlines as separators
    tokens = data_section.split()
    
    # Filter out non-integer tokens (like 'EOF' often found at the end)
    values = []
    for t in tokens:
        try:
            values.append(int(t))
        except ValueError:
            continue 

    # Handling the dimension number if it appears at the start of the data block
    # (Common in SOPLIB files like ESC11.sop where the first number after SECTION is '13')
    if len(values) == dimension * dimension + 1:
        if values[0] == dimension:
            values = values[1:] 
    
    if len(values) < dimension * dimension:
        raise ValueError(f"Insufficient data. Expected {dimension*dimension} values, found {len(values)}")
    
    # Reshape flattened data into a matrix
    # matrix[i][j] represents the value from Node i to Node j (or constraint)
    matrix = [values[i : i + dimension] for i in range(0, len(values), dimension)]
    
    return dimension, matrix

def run_benchmark(filepath, runs=2):
    print(f"Benchmarking: {filepath.split('/')[-1]}")
    instance = SOPInstance(filepath)
    best_global = None
    
    for i in range(runs):
        print(f"\nStarting Run {i+1}/{runs}...")
        solver = HASSOP(instance, n_ants=8)
        
        # Solving
        sol = solver.solve(max_iterations=50)
        
        # --- NEW: Print the sequence for this specific run ---
        print_run_sequence(instance, sol, i + 1)
        # ----------------------------------------------------

        # Update global best
        if sol and (best_global is None or sol.length < best_global.length):
            best_global = sol
            
    if best_global:
        print(f"\nBest Global Solution Found: {best_global.length}")
        # Optional: Print the global best sequence again at the very end
        # print("Global Best Sequence:")
        # print(" -> ".join([str(instance.rev_map[x]) for x in best_global.sequence]))
        
        # Uncomment this if you want to see the plot
        # visualize_solution(instance, best_global, title="Best Global Solution")
    else:
        print("Failed to find a valid solution.")
# if __name__ == "__main__":
#     # Point this to your actual file
#     base_path = os.path.dirname(os.path.abspath(__file__))
#     filename = 'ft53.1.sop' 
#     full_path = os.path.join(base_path, 'archives', 'problems', 'sop', filename)
    
#     if os.path.exists(full_path):
#         # 1. Load
#         instance = SOPInstance(full_path)
        
#         # 3. Run Benchmark
#         run_benchmark(full_path, runs=2)
#     else:
#         print(f"File not found: {full_path}")


# Import các class từ file SOP.py của bạn
# Giả sử code cũ nằm trong file SOP.py, nếu bạn paste code này vào cùng file thì không cần import
# from SOP import SOPInstance, HASSOP 

# --- Hàm hỗ trợ để tắt print trong quá trình chạy (để output bảng cho sạch) ---
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def run_benchmark_table_v(problem_folder, n_runs=5, max_iterations=50):
    """
    Hàm tạo bảng Benchmark giống Table V trong bài báo.
    
    Args:
        problem_folder: Đường dẫn tới thư mục chứa file .sop
        n_runs: Số lần chạy cho mỗi bài toán (Bài báo dùng 5)
        max_iterations: Số vòng lặp tối đa cho mỗi lần chạy
    """
    
    # Danh sách kết quả tổng hợp
    benchmark_data = []
    
    # Lấy danh sách file và sắp xếp tên
    files = sorted([f for f in os.listdir(problem_folder) if f.endswith('.sop')])
    
    print(f"{'='*80}")
    print(f"BẮT ĐẦU BENCHMARK (Table V Simulation)")
    print(f"Số lần chạy mỗi bài: {n_runs}")
    print(f"{'='*80}\n")

    for filename in files:
        filepath = os.path.join(problem_folder, filename)
        prob_name = filename.replace('.sop', '') # Tên bài toán (bỏ đuôi .sop)
        
        print(f"Đang xử lý: {prob_name} ... ", end='', flush=True)
        
        try:
            # Load bài toán
            # Dùng suppress_stdout để tắt các dòng print "Loading..." từ class gốc
            with suppress_stdout():
                instance = SOPInstance(filepath)
            
            run_scores = [] # Lưu kết quả độ dài đường đi (Result)
            run_times = []  # Lưu thời gian chạy (Time)
            
            for i in range(n_runs):
                start_time = time.time()
                
                # Khởi tạo và chạy thuật toán
                # Lưu ý: Paper dùng time limit (120s/600s), ở đây ta dùng iteration để test nhanh
                # Nếu muốn chính xác như paper, cần sửa class HASSOP để dừng theo thời gian.
                with suppress_stdout():
                    solver = HASSOP(instance, n_ants=10) # 10 kiến theo paper
                    sol = solver.solve(max_iterations=max_iterations, early_stop_rounds=20)
                
                end_time = time.time()
                
                if sol:
                    run_scores.append(sol.length)
                    run_times.append(end_time - start_time)
            
            # Tính toán các chỉ số thống kê cho HAS-SOP
            if run_scores:
                best_result = np.min(run_scores)
                avg_result = np.mean(run_scores)
                # Std.Dev trong bài báo thường là Sample Std Dev (ddof=1)
                std_dev = np.std(run_scores, ddof=1) if len(run_scores) > 1 else 0.0
                avg_time = np.mean(run_times)
                
                # Thêm vào bảng dữ liệu
                benchmark_data.append({
                    "PROB": prob_name,
                    "Best Result": int(best_result),
                    "Avg. Result": round(avg_result, 1),
                    "Std. Dev.": round(std_dev, 1),
                    "Avg. Time (sec)": round(avg_time, 1)
                })
                print(f"Xong. (Best: {int(best_result)})")
            else:
                print("Lỗi: Không tìm thấy lời giải.")

        except Exception as e:
            print(f"Lỗi ngoại lệ: {e}")

    # --- TẠO DATAFRAME VÀ HIỂN THỊ ---
    if benchmark_data:
        df = pd.DataFrame(benchmark_data)
        
        # Sắp xếp lại cột cho giống bài báo
        cols = ["PROB", "Best Result", "Avg. Result", "Std. Dev.", "Avg. Time (sec)"]
        df = df[cols]
        
        print("\n" + "="*60)
        print("KẾT QUẢ BENCHMARK (HAS-SOP)")
        print("="*60)
        # In bảng dạng string đẹp
        print(df.to_string(index=False))
        
        # Lưu ra CSV nếu cần
        df.to_csv("table_v_results.csv", index=False)
    else:
        print("\nKhông có dữ liệu để hiển thị.")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Thay đường dẫn này bằng đường dẫn thực tế của bạn
    # Lưu ý: Dùng r"..." để tránh lỗi ký tự đặc biệt trong đường dẫn Windows
    my_folder_path = r"D:\Master_BK\AdvanceALGO\archives\problems\sop"
    
    # Kiểm tra folder có tồn tại không trước khi chạy
    if os.path.exists(my_folder_path):
        # Chạy thử 5 lần (như bài báo) với 100 vòng lặp mỗi lần
        run_benchmark_table_v(my_folder_path, n_runs=5, max_iterations=100)
    else:
        print(f"Không tìm thấy thư mục: {my_folder_path}")