# Introduction
A generic directed graph structure for Zig.
It is intended for in-memory dynamic usage and supports O(1) node insertions, edge insertions, and edge removal; node removal is O(|E|) where
|E| is the number of edges from and to the node being removed. In addition graph reversal is O(|V|), where |V| is the number of vertices of the graph.

The main inteface is provided as:
```zig
AutoDirectedGraph(N: type, V: type, W: type).init(allocator)
```
where `N`, `V`, and `W` are the node, node value, and edge value types, respectively. `N` has to be automatically hashable, for graphs of strings we have:
```zig
StringDirectedGraph(N: type, V: type, W: type).init(allocator)
```
If more control is required, then just as Zig's standard library, one can provide a `HashContext`
```zig
DirectedGraph(N: type, V: type, W: type, HashContext: type, comptime store_hash: bool).init(allocator)
```

This structure uses the terminology *targets* and *sources* for the respective two types of neighbors of a node. A slice of all targets can be obtained
using `digraph.targets(node)` and similarly `digraph.sources(node)`

# Traversal through iterators
Graph traversal is provided through iterators. 
## DFS and BFS
```zig
var dfs = try digraph.dfsIterator(.{.start = start);
var bfs = try digraph.bfsIterator(.{.start = start});
```
One can also specify the traversal direction as `.targetwards` or `.sourcewards` defaulting to `.targetwards`. As well as an iterator used for the
underlying stack/queue and set used by the iterators, otherwise the graph's allocator is used. 
The following obtains a BFS iterator that traverses from `start` in reverse allocating seen nodes and queueing future ones using a custom allocator.
```zig
var bfs = try digraph.bfsIterator(.{.start = start, .direction = .sourcewards, .allocator = allocator});
```
## Dijkstra
Dijkstra's algorithm is encoded as an iterator that returns an entry of the next nearest node and its distance to the starting node.
A distance type has to be provided and must be able to contain the edge weight type `W` if it either an integer or floating point number type.
If `W` is not an arithmetic type, then each edge is assumed to have weight `1`. 
```zig
digraph.dijkstraIterator(Distance: type, options)
```
The following code prints all nodes of distance equal to 1337
to the starting node.
```zig
var dijkstra = digraph.dijkstraIterator(u64, .{.start = start});
while (try dijkstra.next()) |entry| {
   if (entry.distance == 1337) std.debug.print("{any}\n", .{entry.node});
   if (entry.distance > 1337) break;
}
```
## A*
An experimental iterator using a heuristic function. See the source for details.
