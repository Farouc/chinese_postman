# new_main.py
from input import parse_file
from graph import Graph

import argparse
import json
import logging
import random
import time
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


import networkx as nx
from tqdm import tqdm

# ============================================================
# Default params
# ============================================================
DEFAULT_MAX_DP_K = 24 # max k for exact algo DP (open)
DEFAULT_KNN_M = 16 # nb of neighbors for heuristic algorithm (closed)
LARGE_K_FOR_GREEDY = 1000 # using greedy matching if no networkx and k >= this

# ============================================================
# CLI & Logging
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Euler/CPP solver")
    p.add_argument("-i", "--input", required=True, help="Path to the instance file")

    # Modes & algorithm parameters
    p.add_argument("--mode", choices=["auto", "open", "closed"], default="auto")
    # auto = open if k is small, closed otherwise
    # open = Eulerian trail (no requirement to return to the starting point)
    # closed = Eulerian circuit (must return to the starting point)
    p.add_argument("--max-dp-k", type=int, default=DEFAULT_MAX_DP_K)
    # If k <= max-dp-k, use the exact DP algorithm (open)
    # Otherwise, use the heuristic algorithm (closed)
    p.add_argument("--knn-m", type=int, default=DEFAULT_KNN_M)
    # Number of neighbors to consider for the heuristic algorithm (closed)

    # Export & logs
    p.add_argument("--export", help="Output path for the route (JSON/CSV)")
    p.add_argument("--export-format", choices=["json", "csv"], default="json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="count", default=1)
    # verbosity: 0=warning, 1=info, 2=debug
    # (the higher v is, the more verbose the output)

    # Print route options
    p.add_argument("--print-route", action="store_true", help="Print a summary of the computed route")
    p.add_argument("--print-full-route", action="store_true", help="Print the full route (can be huge!)")
    p.add_argument("--print-limit", type=int, default=100)
    p.add_argument("--print-edges", action="store_true")

    # Presets
    p.add_argument("--fast", action="store_true", help="Use fast solver (closed sparse)")
    p.add_argument("--exact", action="store_true", help="Force exact DP solver (if k is small)")
    p.add_argument("--viz", action="store_true", help="Solve + plot automatically")

    # Plot flags
    p.add_argument("--plot_graph", action="store_true")
    
    # Open tour options
    p.add_argument("--open-at-node")
    p.add_argument("--open-near-node")

    return p



def configure_logging(verbosity: int):
    """Configure logging level based on verbosity.
    Parameters
    ----------
    verbosity : int
        0 = warning, 1 = info, 2 = debug
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ============================================================
# Graph utils
# ============================================================
def edges_uv(edges: List[Tuple]) -> List[Tuple]:
    '''
    Extract (u,v) from edges with weights and coordinates.
    '''
    return [(e[0], e[1]) for e in edges]

def edges_uvw(edges: List[Tuple]) -> List[Tuple]:
    '''
    Extract (u,v,w) from edges with weights and coordinates.
    If no weight, assume weight=1.0'''

    out = []
    for e in edges:
        if len(e) >= 3:
            out.append((e[0], e[1], float(e[2])))
        else:
            out.append((e[0], e[1], 1.0))
    return out

def normalize_vertices(vertices: List, uv_edges: List[Tuple]) -> List:
    '''
    Ensure all vertices in edges are in vertices list.
    '''
    V = set(vertices)
    for u, v in uv_edges:
        V.add(u); V.add(v)
    return list(V)

def build_adj_list_all_vertices(vertices: List, uv_edges: List[Tuple]) -> Dict:
    '''
    Build adjacency list for all vertices (including isolated).
    '''
    adj = {v: [] for v in vertices}
    for u, v in uv_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj

def build_weighted_adj(vertices: List, uvw_edges: List[Tuple]) -> Dict:
    '''
    Build weighted adjacency list for all vertices (including isolated).
    If multiple edges between same (u,v), keep the one with smallest weight.
    '''
    adj = {v: [] for v in vertices}
    best = {}
    for u, v, w in uvw_edges:
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) not in best or w < best[(a, b)]:
            best[(a, b)] = w
    for (a, b), w in best.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj

def non_isolated_vertices(adj: Dict) -> set:
    '''
    Return set of non-isolated vertices from adjacency list.
    '''
    return {u for u, nbrs in adj.items() if len(nbrs) > 0}

def is_connected_on_non_isolated(adj: Dict) -> bool:
    '''
    Check if graph is connected ignoring isolated vertices.
    '''
    active = list(non_isolated_vertices(adj))
    if not active:
        return True
    start, visited, stack = active[0], set(), [active[0]]
    # Depth-first search
    while stack:
        u = stack.pop()
        if u in visited: continue
        visited.add(u)
        for v in adj[u]:
            if v not in visited: stack.append(v)
    return visited == set(active)

def odd_degree_vertices(adj: Dict) -> List:
    '''
    Return list of vertices with odd degree.
    '''
    return [u for u, nbrs in adj.items() if len(nbrs) % 2 == 1]

# ============================================================
# Dijkstra + reconstruction
# ============================================================
def dijkstra(adj_w: Dict, source) -> Tuple[Dict, Dict]:
    '''
    Dijkstra from source on weighted adjacency list.
    Returns parent map and distance map.
    '''
    dist, parent = {source: 0.0}, {source: None}
    pq, visited = [(0.0, source)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        for v, w in adj_w[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                #update
                dist[v], parent[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return parent, dist

def shortest_path_vertices(parent: Dict, s, t) -> Optional[List]:
    '''
    Reconstruct shortest path from s to t using parent map.
    If no path, return None.
    '''
    if t not in parent: return None
    path, cur = [], t
    while cur is not None:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path if path and path[0] == s else None

# ============================================================
# MultiGraph & Hierholzer
# ============================================================
class MultiGraph:
    ''' Undirected multigraph with edge counts.
    A MultiGraph is used to represent the graph with duplicated edges
    for finding Eulerian trails.
    Attributes
    ----------  
    vertices : set
        Set of vertices in the graph.
    adj_count : Dict
        Adjacency list with edge counts.
    Methods
    ------- 
    add_edge(u, v, count=1)
        Adds an undirected edge between u and v with given count.
    degree(u) -> int
        Returns the degree of vertex u (sum of edge counts).
    '''
    def __init__(self, vertices: List):
        self.vertices = set(vertices)
        self.adj_count: Dict = {v: defaultdict(int) for v in vertices}
    def add_edge(self, u, v, count: int = 1):
        '''
        Adds an undirected edge between u and v with given count.
        If u or v not in vertices, they are added.
        '''
        self.adj_count.setdefault(u, defaultdict(int))
        self.adj_count.setdefault(v, defaultdict(int))
        self.adj_count[u][v] += count
        self.adj_count[v][u] += count
        self.vertices.add(u); self.vertices.add(v)
    def degree(self, u) -> int:
        '''
        Returns the degree of vertex u (sum of edge counts).
        If u not in vertices, returns 0.
        '''
        return sum(self.adj_count[u].values())

def hierholzer_eulerian_trail(mg: MultiGraph, start=None) -> List:
    '''
    Find Eulerian trail in undirected multigraph using Hierholzer's algorithm.
    An Eulerian trail (or Eulerian path) is a walk through a graph that uses every edge exactly once.

        If the graph has:
        0 odd-degree vertices → the trail is a cycle (starts and ends at the same vertex).
        2 odd-degree vertices → the trail is an open path (starts at one odd-degree vertex and ends at the other).
        > 2 odd-degree vertices → no Eulerian trail exists.

    If start is None, start at an odd-degree vertex if exists, else any vertex.
    Returns list of vertices in the Eulerian trail.
    '''
    odds = [u for u in mg.vertices if mg.degree(u) % 2 == 1] # odd degree vertices
    if start is None:
        # if no start given, start at odd vertex if exists, else any vertex
        start = odds[0] if odds else (next(iter(mg.vertices)) if mg.vertices else None) 
    if start is None: 
        # empty graph
        return []
    stack, path = [start], []
    local_adj = {u: dict(mg.adj_count[u]) for u in mg.adj_count} # copy of adjacency with counts
    # Hierholzer's algorithm
    while stack:
        # current vertex
        u = stack[-1]
        # if u has unused edges
        if local_adj.get(u) and sum(local_adj[u].values()) > 0:
            # get any neighbor v
            v = next(iter(local_adj[u]))
            # use edge u-v
            local_adj[u][v] -= 1
            if local_adj[u][v] == 0: 
                # remove edge if count is zero
                del local_adj[u][v]
            if u in local_adj[v]:
                # remove edge v-u
                local_adj[v][u] -= 1
                if local_adj[v][u] == 0: del local_adj[v][u]
            stack.append(v)
        else: #if no unused edges
            # backtrack, i.e., add to path
            path.append(stack.pop())
    return list(reversed(path))



# ============================================================
# Exact matching (DP bitmask) for small k
# ============================================================

def min_perfect_matching_cost_dp(odd_ids: List, dist_matrix: Dict) -> Tuple[float, List[Tuple]]:
    '''
    Exact minimum weight perfect matching (between odd vertices) on complete graph of odd vertices using DP with bitmask 
    (to encode the indices of odd vertices).

    odd_ids : list of odd vertex ids
    dist_matrix : dict of dict with distances between odd vertices
    Returns total cost and list of matched pairs.
    '''
    m = len(odd_ids)
    full_mask = (1 << m) - 1 # bitmask with all odd vertices included

    from functools import lru_cache # memoization for DP

    @lru_cache(None) # cache results of dp calls
    def dp(mask: int): # mask indicates which odd vertices are still unmatched
        '''
        Returns (min_cost, list_of_pairs) for matching vertices in mask.'''
        if mask == 0:
            # all matched, base case, all paired , so best_cost=0 and no pairs
            return 0.0, []
        # find first unmatched vertex i
        i = (mask & -mask).bit_length() - 1
        # i is the smallest index with bit 1 in mask
        # try pairing i with all other unmatched j > i
        best_cost, best_pairs = float('inf'), None
        rem = mask ^ (1 << i)
        #1<<i means decaling the mask to the left by i bits
        #rem is the mask without i
        #try pairing i with j
        j = i + 1
        while j < m:
            # we look for j > i and j is also unmatched 
            if (rem >> j) & 1:
                # j is unmatched, try pairing i-j
                gi, gj = odd_ids[i], odd_ids[j]
                c = dist_matrix[gi].get(gj, float('inf')) # cost of edge i-j
                sub_cost, sub_pairs = dp(rem ^ (1 << j)) #^ is the xor operator
                #rem ^ (1 << j) is the mask without i and j
                tot = c + sub_cost
                if tot < best_cost:
                    best_cost = tot
                    best_pairs = sub_pairs + [(gi, gj)]
            j += 1
        return best_cost, best_pairs

    return dp(full_mask)

# ============================================================
# Duplication & multigraph construction
# ============================================================

def duplicate_path_on_multigraph(mg: MultiGraph, path: List):
    '''
    Duplicate edges along the given path in the multigraph.
    '''
    for i in range(len(path) - 1):
        mg.add_edge(path[i], path[i + 1], 1)

def build_multigraph_from_edges(vertices: List, uv_edges: List[Tuple]) -> MultiGraph:
    '''
    Build a MultiGraph from vertices and undirected edges (u,v).
    '''
    mg = MultiGraph(vertices)
    for u, v in uv_edges:
        mg.add_edge(u, v, 1)
    return mg

# ============================================================
# Odd-pair distances — for small k
# ============================================================

def all_pairs_shortest_for_odds(vertices: List, uvw_edges: List[Tuple], odd_list: List) -> Tuple[Dict, Dict, Dict]:
    '''
    Compute all-pairs shortest paths between odd vertices using Dijkstra from each odd vertex.
    Returns parents map, dists map, and distance matrix between odd vertices.
    Is used for exact matching (DP) when k is small.
    '''
    adj_w = build_weighted_adj(vertices, uvw_edges)
    parents, dists = {}, {}
    for u in odd_list:
        pu, du = dijkstra(adj_w, u)
        parents[u] = pu
        dists[u] = du
    dist_matrix = defaultdict(dict)
    for u in odd_list:
        for v in odd_list:
            dist_matrix[u][v] = 0.0 if u == v else dists[u].get(v, float('inf'))
    return parents, dists, dist_matrix

# ============================================================
# k-NN troncated Dijkstra — for large k 
# ============================================================

def dijkstra_k_nearest_odds(adj_w: Dict, source, odd_set: set, m: int):
    '''
    Dijkstra from source, stop when m odd vertices found.

    Returns list of found (odd_vertex, distance) and parent map.
    '''
    pq = [(0.0, source)]
    dist = {source: 0.0}
    parent = {source: None}
    visited = set()
    found = []
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u in odd_set and u != source:
            found.append((u, d))
            if len(found) >= m:
                break
        for v, w in adj_w[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return found, parent

def build_sparse_knn_graph_for_odds(vertices: List, uvw_edges: List[Tuple], odd_list: List, m: int):
    '''
    Build sparse graph between odd vertices using k-NN Dijkstra truncated.

    Returns parents map and list of sparse edges (u,v,w).
    '''
    adj_w = build_weighted_adj(vertices, uvw_edges) # full weighted adjacency
    odd_set = set(odd_list) # set of odd vertices
    parents_map = {}
    candidate_edges = []

    iterable = tqdm(odd_list, desc=f"k-NN (m={m})")
    for u in iterable:
        found, parent = dijkstra_k_nearest_odds(adj_w, u, odd_set, m)
        parents_map[u] = parent
        for v, d in found: # found odd vertex v with distance d
            a, b = (u, v) if u < v else (v, u) 
            candidate_edges.append((a, b, d))
    best = {}
    for a, b, d in candidate_edges: # keep best edge if multiple found
        if (a, b) not in best or d < best[(a, b)]:
            best[(a, b)] = d
    sparse_edges = [(a, b, w) for (a, b), w in best.items()]
    return parents_map, sparse_edges

def matching_on_sparse_graph(odd_list: List, sparse_edges: List[Tuple], use_networkx: bool = True) -> List[Tuple]:
    '''
    Find matching on sparse graph of odd vertices using networkx if available, else greedy.
        if use_networkx is True, uses Blossom algorithm for min weight perfect matching (Optimal).
        else, uses greedy matching (approximation).
    Returns list of matched pairs.
    '''
    if use_networkx:
        try:
            # Optimal with NetworkX using Blossom algorithm
            G = nx.Graph()
            for u in odd_list:
                G.add_node(u)
            for u, v, w in sparse_edges:
                if w != float('inf'):
                    G.add_edge(u, v, weight=w)
            # use Blossom algorithm for min weight perfect matching
            M = nx.algorithms.matching.min_weight_matching(G, maxcardinality=True, weight='weight')
            return [(a, b) for a, b in M]
        except Exception:
            pass
    # Glouton
    sparse_edges_sorted = sorted(sparse_edges, key=lambda x: x[2])
    #sort edges by incrieasing weight
    used = set() # matched vertices
    pairs = [] # matched pairs
    for u, v, w in sparse_edges_sorted:
        if u not in used and v not in used and w != float('inf'):
            used.add(u); used.add(v)
            pairs.append((u, v))
    return pairs

# ============================================================
# High level solvers
# ============================================================

def solve_small_k_open_minimal(vertices: List, edges: List[Tuple], odds: List, max_dp_k: int, timings: Dict) -> Tuple[List, Dict]:
    '''
    Exact solver for small k (odd vertices) using DP matching (open Eulerian trail with minimal added length).
    '''
    uv = edges_uv(edges)
    uvw = edges_uvw(edges)

    t0 = time.time()
    parents, dists, dist_matrix = all_pairs_shortest_for_odds(vertices, uvw, odds)
    timings["apsp_odds_sec"] = round(time.time() - t0, 3)

    best_cost, best_s, best_t, best_pairs = float('inf'), None, None, []
    k = len(odds)
    t1 = time.time()
    for i in range(k):
        for j in range(i + 1, k):
            s, t = odds[i], odds[j]
            remaining = [x for x in odds if x not in (s, t)]
            if not remaining:
                cost_pairs, pairs = 0.0, []
            else:
                cost_pairs, pairs = min_perfect_matching_cost_dp(remaining, dist_matrix)
            if cost_pairs < best_cost:
                best_cost, best_s, best_t, best_pairs = cost_pairs, s, t, pairs
    timings["dp_matching_sec"] = round(time.time() - t1, 3)

    mg = build_multigraph_from_edges(vertices, uv)
    added_length = 0.0
    for (u, v) in best_pairs:
        parent_u = parents[u]
        path = shortest_path_vertices(parent_u, u, v)
        if path is None:
            logging.warning(f"Failed to rebuild path {u}->{v}")
            continue
        duplicate_path_on_multigraph(mg, path)
        added_length += dist_matrix[u][v]

    t2 = time.time()
    trail = hierholzer_eulerian_trail(mg, start=best_s)
    timings["hierholzer_sec"] = round(time.time() - t2, 3)

    meta = {
        "mode": "open_exact",
        "k": k,
        "pairs": len(best_pairs),
        "added_length": round(added_length, 3),
    }
    return trail, meta

def solve_large_k_closed_sparse(vertices: List, edges: List[Tuple], odds: List, knn_m: int, use_nx: bool, timings: Dict) -> Tuple[List, Dict]:
    '''
    Heuristic solver for large k (odd vertices) using sparse k-NN graph and matching (closed Eulerian trail).
    '''
    uv = edges_uv(edges)
    uvw = edges_uvw(edges)
    mg = build_multigraph_from_edges(vertices, uv)

    t0 = time.time()
    parents_map, sparse_edges = build_sparse_knn_graph_for_odds(vertices, uvw, odds, m=knn_m)
    timings["knn_build_sec"] = round(time.time() - t0, 3)
    logging.info(f"Sparse odd-graph edges: ~{len(sparse_edges)}")

    t1 = time.time()
    pairs = matching_on_sparse_graph(odds, sparse_edges, use_networkx=use_nx)
    timings["sparse_matching_sec"] = round(time.time() - t1, 3)
    logging.info(f"Matched {len(pairs)} pairs on sparse odd-graph")

    t2 = time.time()
    added_length = 0.0
    weight_lookup = {(min(a,b), max(a,b)): w for (a, b, w) in sparse_edges}
    dup_edges_count = 0
    for (u, v) in pairs:
        parent_u = parents_map.get(u)
        path = None
        if parent_u is not None and v in parent_u:
            path = shortest_path_vertices(parent_u, u, v)
        else:
            parent_v = parents_map.get(v)
            if parent_v is not None and u in parent_v:
                path = shortest_path_vertices(parent_v, v, u)
        if path is None:
            logging.warning(f"Skipping unmatched path between {u} and {v}")
            continue
        duplicate_path_on_multigraph(mg, path)
        dup_edges_count += len(path) - 1
        w = weight_lookup.get((min(u, v), max(u, v)), 0.0)
        added_length += w
    timings["duplication_sec"] = round(time.time() - t2, 3)

    t3 = time.time()
    cand = [x for x in mg.vertices if mg.degree(x) > 0]
    start = cand[0] if cand else None
    tour = hierholzer_eulerian_trail(mg, start=start)
    timings["hierholzer_sec"] = round(time.time() - t3, 3)

    meta = {
        "mode": "closed_sparse",
        "k": len(odds),
        "pairs": len(pairs),
        "dup_edges": dup_edges_count,
        "added_length": round(added_length, 3),
    }
    return tour, meta

# ============================================================
# Orchestration
# ============================================================

def solve_eulerian_with_args(vertices: List, edges: List[Tuple], args) -> Tuple[List, Dict]:
    '''
    Solve Eulerian trail/circuit based on args.
    '''
    timings = {}
    uv = edges_uv(edges)

    t0 = time.time()
    vertices = normalize_vertices(vertices, uv)
    adj = build_adj_list_all_vertices(vertices, uv)
    timings["build_adj_sec"] = round(time.time() - t0, 3)

    # Connexity
    t1 = time.time()
    if not is_connected_on_non_isolated(adj):
        logging.error("Graph is not connected on its non-isolated component set; no eulerian trail exists.")
        return [], {"error": "not_connected"}
    timings["connectivity_sec"] = round(time.time() - t1, 3)

    odds = odd_degree_vertices(adj)
    k = len(odds)
    logging.info(f"Odd-degree vertices: k={k}")

    # k == 0 : 
    if k == 0:
        mg = build_multigraph_from_edges(vertices, uv)
        start = 0 if (0 in mg.vertices and mg.degree(0) > 0) else (next(iter(non_isolated_vertices(adj))) if non_isolated_vertices(adj) else None)
        t2 = time.time()
        tour = hierholzer_eulerian_trail(mg, start=start)
        timings["hierholzer_sec"] = round(time.time() - t2, 3)
        meta = {"mode": "closed_direct", "k": 0}
        # ouverture optionnelle
        if args.mode in ("open",) or args.open_at_node or args.open_near_node:
            open_node = args.open_at_node if args.open_at_node is not None else None
            near_node = args.open_near_node if args.open_near_node is not None else None
            trail = open_tour_at_node(tour, node=open_node, near_node=near_node)
            meta["opened"] = True
            return trail, {**meta, "timings_sec": timings}
        return tour, {**meta, "timings_sec": timings}

    # k == 2 : 
    if k == 2:
        mg = build_multigraph_from_edges(vertices, uv)
        s = odds[0]
        t2 = time.time()
        trail = hierholzer_eulerian_trail(mg, start=s)
        timings["hierholzer_sec"] = round(time.time() - t2, 3)
        meta = {"mode": "open_direct", "k": 2}
        return trail, {**meta, "timings_sec": timings}

    # 3 <= k
    chosen_mode = args.mode
    if chosen_mode == "auto":
        chosen_mode = "open" if k <= args.max_dp_k else "closed"

    if chosen_mode == "open":
        if k <= args.max_dp_k:
            logging.info(f"Using OPEN exact (DP) since k <= {args.max_dp_k}.")
            trail, meta = solve_small_k_open_minimal(vertices, edges, odds, args.max_dp_k, timings)
            return trail, {**meta, "timings_sec": timings}
        else:
            logging.info(f"k > {args.max_dp_k}: using CLOSED sparse tour then opening it.")
            use_nx = True
            if not use_nx and k >= LARGE_K_FOR_GREEDY:
                logging.info("networkx not available; using greedy heuristic matching.")
            tour, meta = solve_large_k_closed_sparse(vertices, edges, odds, args.knn_m, use_nx, timings)
            open_node = args.open_at_node if args.open_at_node is not None else None
            near_node = args.open_near_node if args.open_near_node is not None else None
            trail = open_tour_at_node(tour, node=open_node, near_node=near_node)
            meta["opened"] = True
            return trail, {**meta, "timings_sec": timings}

    # chosen_mode == "closed"
    logging.info("Using CLOSED (Chinese Postman) scalable solver.")
    use_nx = True
    if not use_nx and k >= LARGE_K_FOR_GREEDY:
        logging.info("networkx not available; using greedy heuristic matching.")
    tour, meta = solve_large_k_closed_sparse(vertices, edges, odds, args.knn_m, use_nx, timings)
    if args.open_at_node or args.open_near_node:
        open_node = args.open_at_node if args.open_at_node is not None else None
        near_node = args.open_near_node if args.open_near_node is not None else None
        trail = open_tour_at_node(tour, node=open_node, near_node=near_node)
        meta["opened"] = True
        return trail, {**meta, "timings_sec": timings}
    return tour, {**meta, "timings_sec": timings}



# ============================================================
# Route & printing utils
# ============================================================
def rotate_to_start(seq: List, start_node) -> List:
    '''
    Rotate sequence to start at start_node.
    E.g. [A,B,C,D], start_node=B -> [B,C,D,A]
    '''
    try:
        idx = seq.index(start_node)
    except ValueError:
        return seq
    return seq[idx:] + seq[:idx]
def open_tour_at_node(tour: List, node=None, near_node=None) -> List:
    if not tour:
        return tour
    seq = list(tour)
    if node is not None and node in seq:
        seq = rotate_to_start(seq, node)
    elif near_node is not None and near_node in seq:
        seq = rotate_to_start(seq, near_node)
    if len(seq) >= 2 and seq[0] == seq[-1]:
        seq = seq[:-1]
    return seq
def edges_from_trail(trail: List) -> List[Tuple]:
    '''
    Convert trail of vertices to list of edges (u,v).
    '''
    return [(trail[i], trail[i+1]) for i in range(len(trail)-1)]

def print_route_to_console(trail: List, print_edges: bool = False,
                           limit: int = 100, full: bool = False):
    '''
    Print route to console, either as list of edges or list of vertices.
    If full is False and route is longer than limit, print head and tail with ellipsis.
    '''
    if not trail:
        print("Route: <vide>"); return
    if print_edges:
        eds = edges_from_trail(trail)
        print(f"Route (arêtes) count = {len(eds)}")
        if full or len(eds) <= limit: print(eds)
        else:
            head, tail = eds[: limit//2], eds[-(limit - limit//2):]
            print(head + [("...", "...")] + tail)
    else:
        print(f"Route (nœuds) length = {len(trail)}")
        if full or len(trail) <= limit: print(trail)
        else:
            head, tail = trail[: limit//2], trail[-(limit - limit//2):]
            print(head + ["..."] + tail)

def print_summary(meta: Dict, vertices: List, edges: List, trail: List):
    '''
    Print solution summary to the console.
    '''
    print("\n=== Solution Summary ===")
    print(f"Resolution mode        : {meta.get('mode')}")
    print(f"Total vertices         : {len(vertices)}")
    print(f"Total edges            : {len(edges)}")
    print(f"Odd-degree vertices (k): {meta.get('k')}")
    print(f"Matched pairs          : {meta.get('pairs', '-')}")
    print(f"Duplicated edges       : {meta.get('dup_edges', '-')}")
    print(f"Added length           : {meta.get('added_length', '-')}")
    print(f"Final trail length     : {len(trail)}")
    print(f"Total time             : {meta.get('total_time_sec')} s")
    print("========================\n")

# ============================================================
# Export
# ============================================================
def export_trail(path: Optional[str], trail: List, meta: Dict, fmt: str = "json"):
    '''
    Export trail and meta to file in JSON or CSV format.
    '''
    if not path: return
    extra_stats = {"total_nodes": len(trail), "total_edges": len(edges_from_trail(trail))}
    meta = {**meta, **extra_stats}
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"trail": trail, "meta": meta}, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("node\n")
            for v in trail: f.write(f"{v}\n")

# ============================================================
# Main
# ============================================================
if __name__=="__main__":
    args = build_argparser().parse_args()
    configure_logging(args.verbose)
    random.seed(args.seed)

    if args.fast: args.mode = "closed"
    if args.exact: args.mode = "open"
    if args.viz: args.no_plot, args.print_route = False, True

    vertices, edges = parse_file(args.input)
    edges0 = list(edges)

    t0 = time.time()
    trail, meta = solve_eulerian_with_args(vertices, edges, args)
    meta["total_time_sec"] = round(time.time() - t0, 3)

    print_summary(meta, vertices, edges, trail)

    if args.print_route:
        print_route_to_console(trail, print_edges=args.print_edges,
                               limit=args.print_limit, full=args.print_full_route)

    if args.export:
        export_trail(args.export, trail, meta, fmt=args.export_format)

    graph = Graph(vertices, edges0)
    if args.plot_graph:
        graph.plot()
