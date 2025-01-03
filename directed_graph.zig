const std = @import("std");
const meta = std.meta;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const ArrayHashMapUnmanaged = std.ArrayHashMapUnmanaged;
const AutoArrayHashMapUnmanaged = std.AutoArrayHashMapUnmanaged;
const LinearFifo = std.fifo.LinearFifo;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const hash_map = std.hash_map;
const array_hash_map = std.array_hash_map;

/// A directed graph with nodes being hashed using AutoContext.
pub fn AutoDirectedGraph(N: type, V: type, W: type) type {
    return DirectedGraph(N, V, W, array_hash_map.AutoContext(N), !array_hash_map.autoEqlIsCheap(N));
}

/// A directed graph indexed by strings.
pub fn StringDirectedGraph(V: type, W: type) type {
    return DirectedGraph([]const u8, W, V, hash_map.StringContext, true);
}

/// Directed graph that attempts to strike a balance between being ergonomic, dynamic, and fast.
/// It supports fast node and edge insertions and removals, as well as querying. By utlizing
/// array hashmaps one can in O(1) obtain a slice of all sources (predecessors, in neighbors, or tails),
/// and all targets (succssors, out neighbors, or heads).
/// It supports values associated to nodes and edges.
/// Notably, it uses iterators for more generic graph traversal.
/// The downside is having to store the node labels and weights for each set of sources and targets.
pub fn DirectedGraph(N: type, V: type, W: type, HashContext: type, comptime store_hash: bool) type {
    return struct {
        const Self = @This();
        const AdjacentMap = ArrayHashMapUnmanaged(N, W, HashContext, store_hash);
        const AdjacencyMaps = struct {
            source_map: AdjacentMap = .{},
            target_map: AdjacentMap = .{},
        };
        const NodeMap = ArrayHashMapUnmanaged(N, AdjacencyMaps, HashContext, store_hash);

        allocator: Allocator,
        node_values: ArrayListUnmanaged(V) = .{},
        node_map: NodeMap = .{},

        /// Check whether an edge belongs to the directed graph.
        pub fn containsEdge(self: Self, a: N, b: N) bool {
            const adjacent = self.node_map.get(a) orelse return false;
            return adjacent.targets.get(b);
        }

        /// Check whether a node belongs to the directed graph.
        pub fn containsNode(self: Self, node: N) bool {
            return self.node_map.contains(node);
        }

        /// Frees all resources associated to the directed graph.
        /// The caller is responsible for freeing any data stored by the
        /// node and edge values.
        pub fn deinit(self: *Self) void {
            const allocator = self.allocator;
            for (self.node_map.values()) |*adjacent| {
                adjacent.target_map.deinit(allocator);
                adjacent.source_map.deinit(allocator);
            }
            self.node_map.deinit(allocator);
            self.node_values.deinit(allocator);
        }

        /// Returns the weight of the directed edge from 'a' to 'b' or null if not found.
        pub fn getEdge(self: Self, a: N, b: N) ?W {
            const adjacent = self.node_map.get(a) orelse return null;
            return adjacent.targets.get(b);
        }

        /// Returns the value of 'node' or null if not found.
        pub fn getNode(self: Self, node: N) ?V {
            const index = self.getNodeIndex(node) orelse return null;
            return self.node_values.items[index];
        }

        /// Returns the index of 'node' or null if not found.
        pub fn getNodeIndex(self: Self, node: N) ?usize {
            return self.node_map.getIndex(node);
        }

        /// Returns an empty directed graph.
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        // Returns a slice of all node values.
        pub fn nodeValues(self: Self) []V {
            return self.node_values.items;
        }

        /// Returns a slice of all nodes.
        pub fn nodes(self: Self) []N {
            return self.node_map.keys();
        }

        /// Returns the number of nodes contained in the graph.
        pub fn order(self: Self) usize {
            return self.node_map.count();
        }

        /// Clobbers any existing data.
        pub fn putNode(self: *Self, node: N, value: V) Allocator.Error!void {
            const result = try self.node_map.getOrPut(self.allocator, node);
            if (result.found_existing) {
                self.node_values.items[result.index] = value;
            } else {
                result.value_ptr.* = .{};
                try self.node_values.append(self.allocator, value);
            }
        }

        /// Clobbers any existing data.
        /// Asserts that the graph contains the two given nodes.
        pub fn putEdge(self: *Self, a: N, b: N, weight: W) Allocator.Error!void {
            assert(self.containsNode(a));
            assert(self.containsNode(b));
            try self.node_map.getPtr(a).?.target_map.put(self.allocator, b, weight);
            try self.node_map.getPtr(b).?.source_map.put(self.allocator, a, weight);
        }

        /// Reverses the graph, flipping the direction of the edges.
        pub fn reverse(self: *Self) void {
            for (self.node_map.values()) |*adjacent| {
                std.mem.swap(
                    @TypeOf(adjacent.source_map, adjacent.target_map),
                    &adjacent.source_map,
                    &adjacent.target_map,
                );
            }
        }

        /// Returns the a slice of all sources of 'node'.
        /// Asserts that 'node' is contained in the graph.
        pub fn sources(self: Self, node: N) []N {
            assert(self.containsNode(node));
            return self.node_map.get(node).?.source_map.keys();
        }

        /// If there is a directed edge from 'a' to 'b', then remove it by
        /// swapping it with the last elements of the respective targets and
        /// sources maps. Returns true if an edge was removed, false otherwise.
        pub fn swapRemoveEdge(self: *Self, a: N, b: N) bool {
            const adjacent = self.node_map.getPtr(a) orelse return false;
            if (!adjacent.target_map.swapRemove(b)) return false;
            assert(self.node_map.getPtr(b).?.source_map.swapRemove(a));
            return true;
        }

        /// If 'node' is contained in the directed graph, then remove it by
        /// swapping it with the last node. Returns true if 'node' existed
        /// in the graph, otherwise return false.
        /// In addition all source and target nodes have their adjacency maps
        /// updated by swap removing the node.
        /// This operation also swap removes the corresponding value from the
        /// node_values array list.
        pub fn swapRemoveNode(self: *Self, node: N) bool {
            const index = self.getNodeIndex(node) orelse return false;
            var adjacent = self.node_map.values()[index];
            for (adjacent.source_map.keys()) |source| {
                assert(self.node_map.getPtr(source).?.target_map.swapRemove(node));
            }
            for (adjacent.target_map.keys()) |target| {
                assert(self.node_map.getPtr(target).?.source_map.swapRemove(node));
            }
            adjacent.source_map.deinit(self.allocator);
            adjacent.target_map.deinit(self.allocator);
            self.node_values.swapRemove(index);
            self.node_map.swapRemoveAt(index);
            return true;
        }

        /// Returns a slice of all targets of 'node'.
        /// Asserts that 'node' is contained in the graph.
        pub fn targets(self: Self, node: N) []N {
            assert(self.containsNode(node));
            return self.node_map.get(node).?.target_map.keys();
        }
        
        // Iterators.
        pub const NodeSet = ArrayHashMapUnmanaged(N, void, HashContext, store_hash);
        /// Direction of traversal, either in the direction of the edges 'targetwards'
        /// or in the reverse direction 'sourcewards'.
        pub const TraversalDirection = enum { sourcewards, targetwards };
        /// Iterates through the nodes of a graph in a layer-by-layer manner.
        pub const BFSIterator = struct {
            pub const InitOptions = struct {
                allocator: ?Allocator = null,
                direction: TraversalDirection = .targetwards,
                start: N,
            };
            pub const ResetOptions = struct {
                direction: ?TraversalDirection = null,
                start: N,
            };
            
            pub const Queue = LinearFifo(N, .Dynamic);
            
            allocator: Allocator, 
            direction: TraversalDirection,
            graph: *Self,
            queue: Queue,
            seen: NodeSet,

            /// Frees the backing allocations.
            pub fn deinit(iter: *BFSIterator) void {
                iter.queue.deinit();
                iter.seen.deinit(iter.allocator);
            }

            /// Returns a BFSIterator of a given direction and start node.
            /// Note that the start node has not been visited yet, but will be the
            /// first one to be visited.
            pub fn init(graph: *Self, options: InitOptions) Allocator.Error!BFSIterator {
                var iter = BFSIterator{
                    .allocator = options.allocator orelse graph.allocator,
                    .direction = options.direction,
                    .graph = graph,
                    .queue = undefined,
                    .seen = .{},
                };
                iter.queue = Queue.init(iter.allocator);
                try iter.queue.writeItem(options.start);
                return iter;
            }

            /// Returns the next node of the iterator or null if every node in the connected
            /// component of the starting node has been visited.
            pub fn next(iter: *BFSIterator) Allocator.Error!?N {
                return while (iter.queue.readItem()) |node| {
                    const result = try iter.seen.getOrPut(iter.allocator, node);
                    if (result.found_existing) continue;
                    const neighbors = switch (iter.direction) {
                        .sourcewards => iter.graph.sources(node),
                        .targetwards => iter.graph.targets(node),
                    };
                    for (neighbors) |neighbor| {
                        if (iter.seen.contains(neighbor)) continue;
                        try iter.queue.writeItem(neighbor);
                    }
                    break node;
                } else null;
            }

            /// Returns the what the next value of the iterator will be, or null if none.
            pub fn peek(iter: *BFSIterator) ?N {
                return if (iter.queue.count > 0) iter.queue.peekItem(0) else null;
            }

            /// Resets the iterator to a given starting node.
            pub fn reset(iter: *BFSIterator, options: ResetOptions) void {
                if (options.direction) |dir| iter.direction = dir;
                iter.queue.discard(iter.queue.count);
                iter.queue.writeItemAssumeCapacity(options.start); 
                iter.seen.clearRetainingCapacity();
            }
        };
        /// A breadth first iterator through all the nodes in the connected component of 'start'.
        pub fn bfsIterator(self: *Self, options: BFSIterator.InitOptions) Allocator.Error!BFSIterator {
            return BFSIterator.init(self, options);
        }
        
        /// Iterates through the nodes of a graph in a greedy manner.
        pub const DFSIterator = struct {
            pub const InitOptions = struct {
                allocator: ?Allocator = null,
                direction: TraversalDirection = .targetwards,
                start: N,
            };
            pub const ResetOptions = struct {
                direction: ?TraversalDirection = null,
                start: N,
            };
            
            pub const Stack = ArrayListUnmanaged(N);
            
            allocator: Allocator,
            direction: TraversalDirection,
            graph: *Self,
            seen: NodeSet,
            stack: Stack,

            /// Frees the backing allocations.
            pub fn deinit(iter: *DFSIterator) void {
                iter.seen.deinit(iter.allocator);
                iter.stack.deinit(iter.allocator);
            }

            /// Returns a DFSIterator of a given direction and start node.
            /// Note that the start node has not been visited yet, but will be the
            /// first one to be visited.
            pub fn init(graph: *Self, options: InitOptions) Allocator.Error!DFSIterator {
                var iter = DFSIterator{
                    .allocator = options.allocator orelse graph.allocator,
                    .direction = options.direction,
                    .graph = graph,
                    .seen = .{},
                    .stack = .{},
                };
                try iter.stack.append(iter.graph.allocator, options.start);
                return iter;
            }

            /// Returns the next node of the iterator of null if every node in the connected
            /// component of the starting node has been visited.
            pub fn next(iter: *DFSIterator) Allocator.Error!?N {
                return while (iter.stack.popOrNull()) |node| {
                    const result = try iter.seen.getOrPut(iter.allocator, node);
                    if (result.found_existing) continue;
                    const neighbors = switch (iter.direction) {
                        .sourcewards => iter.graph.sources(node),
                        .targetwards => iter.graph.targets(node),
                    };
                    for (neighbors) |neighbor| {
                        if (iter.seen.contains(neighbor)) continue;
                        try iter.stack.append(iter.allocator, neighbor);
                    }
                    break node;
                } else null;
            }

            /// Returns the what the next value of the iterator will be, or null if none.
            pub fn peek(iter: *DFSIterator) ?N {
                return iter.stack.getLastOrNull();
            }

            /// Resets the iterator to a given starting node.
            pub fn reset(iter: *DFSIterator, options: ResetOptions) void {
                if (options.direction) |dir| iter.direction = dir;
                iter.stack.clearRetainingCapacity();
                iter.stack.appendAssumeCapacity(options.start);
                iter.seen.clearRetainingCapacity();
            }
        };
        /// A depth first iterator traversing to every node in the connected component of 'start'.
        pub fn dfsIterator(self: *Self, options: DFSIterator.InitOptions) Allocator.Error!DFSIterator {
            return DFSIterator.init(self, options);
        }

        /// A depth limited (first) search iterator. Essentially a DFS iterator but it will only search
        /// up to and including a given depth. The returned entries of each iterations reports the found
        /// node and the depth it was reached at. There is no guarantee that the reported depth is the
        /// minimal one, see DijkstraIterator for this.
        pub const DLSIterator = struct {
            // The entry type returned by the iterator.
            pub const Entry = struct {
                node: N,
                depth: usize,
            };
            pub const InitOptions = struct {
                allocator: ?Allocator = null,
                depth: usize,
                direction: TraversalDirection = .targetwards,
                start: N,
            };
            pub const ResetOptions = struct {
                depth: ?usize = null,
                direction: ?TraversalDirection = null, 
                start: N,
            };

            const Stack = ArrayListUnmanaged(Entry);
            
            allocator: Allocator,
            depth: usize,
            direction: TraversalDirection,
            graph: *Self,
            stack: Stack,
            
            pub fn deinit(iter: *DLSIterator) void {
                iter.stack.deinit(iter.allocator);
            }
            
            pub fn init(graph: *Self, options: InitOptions) Allocator.Error!DLSIterator {
                var iter = DLSIterator{
                    .allocator = options.allocator orelse graph.allocator,
                    .graph = graph,
                    .stack = .{},
                    .depth = options.depth,
                    .direction = options.direction,
                };
                try iter.stack.append(iter.allocator, .{.node = options.start, .depth = 0});
                return iter;
            }
            
            pub fn next(iter: *DLSIterator) !?Entry {
                return while (iter.stack.popOrNull()) |entry| {
                    const neighbors = switch (iter.direction) {
                        .sourcewards => iter.graph.sources(entry.node),
                        .targetwards => iter.graph.targets(entry.node),
                    };
                    if (entry.depth == iter.depth) break entry;
                    for (neighbors) |neighbor| {
                        try iter.stack.append(iter.allocator, .{.node = neighbor, .depth = entry.depth + 1});
                    }
                    break entry;
                } else null;
            }
            
            pub fn peek(iter: *DLSIterator) ?Entry {
                return iter.stack.getLastOrNull(); 
            }
            
            pub fn reset(iter: *DLSIterator, options: ResetOptions) void {
                if (options.depth) |depth| iter.depth = depth;
                if (options.direction) |direction| iter.direction = direction;
                iter.stack.clearRetainingCapacity();
                iter.stack.appendAssumeCapacity(.{.node = options.start, .depth = 0});
            }
        };
        pub fn dlsIterator(self: *Self, options: DLSIterator.InitOptions) Allocator.Error!DLSIterator {
            return DLSIterator.init(self, options);
        }
        /// Iterative Deepening (depth first) Search.
        pub const IDSIterator = struct {
            const Entry = DLSIterator.Entry;
            const InitOptions = struct {
                allocator: ?Allocator = null,
                // Initial depth.
                depth: usize = 0,
                direction: TraversalDirection = .targetwards,
                start: N,
            };
            const ResetOptions = struct {
                depth: usize = 0, 
                direction: ?TraversalDirection = null,
                start: ?N,
            };
            
            const Stack = ArrayListUnmanaged(Entry);
            
            dls: DLSIterator,
            // Needs to be stored for repeated resets of the DLS iterator.
            depth_reached: bool,
            start: N,

            pub fn deinit(iter: *IDSIterator) void {
                iter.dls.deinit();
            }
            
            pub fn init(graph: *Self, options: InitOptions) Allocator.Error!IDSIterator {
                return IDSIterator{
                    .depth_reached = false,
                    .dls = try graph.dlsIterator(.{
                        .allocator = options.allocator,
                        .depth = options.depth,
                        .direction = options.direction,
                        .start = options.start,
                    }),
                    .start = options.start,
                };
            }
            
            pub fn next(iter: *IDSIterator) Allocator.Error!?Entry {
                if (try iter.dls.next()) |entry| {
                    if (entry.depth == iter.dls.depth) iter.depth_reached = true;
                    return entry;
                } else {
                    if (!iter.depth_reached) return null;
                    iter.depth_reached = false;
                    iter.dls.reset(.{.start = iter.start, .depth = iter.dls.depth + 1});
                    return try iter.next();
                }
            }
            
            pub fn reset(iter: *IDSIterator, options: ResetOptions) void {
                iter.depth_reached = false;
                if (options.start) |node| iter.start = node;
                iter.dls.reset(.{
                    .depth = options.depth,
                    .direction = options.direction,
                    .start = options.start,
                });
            }
        };
        pub fn idsIterator(self: *Self, options: IDSIterator.InitOptions) Allocator.Error!IDSIterator {
       	    return IDSIterator.init(self, options); 
        }

        /// The weight type used by the DijkstraIterator. Currently only arithmetic
        /// weights are supported if given some other type of edge value then assume
        /// the weight of every edge is exactly 1.
        const ArithmeticWeight = switch (@typeInfo(W)) {
            .Int, .Float => W,
            else => u1,
        };
        /// Iterated through the nodes of a graph in order of distance to the starting node.
        pub fn DijkstraIterator(Distance: type) type {
            // Verify that 'Distance' can safely hold 'ArithmeticWeight'.
            // Todo: Handle different signs of Distance and ArithmeticWeight.
            switch (@typeInfo(Distance)) {
                .Int => |info_distance| switch (@typeInfo(ArithmeticWeight)) {
                    .Int => |info_weight| {
                        if (info_distance.bits < info_weight.bits) @compileError("Distance type can not contain all of weight type");
                    },
                    else => @compileError("Distance type: " ++ @typeName(Distance) ++
                        " must be an integer type to match weight type: " ++ @typeName(ArithmeticWeight)),
                },
                .Float => |info_distance| switch (@typeInfo(ArithmeticWeight)) {
                    .Float => |info_weight| {
                        if (info_distance.bits < info_weight.bits) @compileError("Distance type can not contain all of weight type");
                    },
                    else => @compileError("Distance must be float to match weight"),
                },
                else => @compileError("Distance type must be an int or float."),
            }
            return struct {
                const ThisIterator = @This();
                /// The node-distance pair that the Dijkstra iterator returns.
                const DijkstraEntry = struct {
                    node: N,
                    distance: Distance,
                };
                const InitOptions = struct {
                    allocator: ?Allocator = null,
                    direction: TraversalDirection = .targetwards,
                    start: N,
                };
                const ResetOptions = struct {
                    direction: ?TraversalDirection = null,
                    start: N,
                };
                
                // Compare function used by the priority queue.
                fn lessThan(_: void, a: DijkstraEntry, b: DijkstraEntry) std.math.Order {
                    return std.math.order(a.distance, b.distance);
                }
                const PriorityQueue = std.PriorityQueue(DijkstraEntry, void, lessThan);
                
                allocator: Allocator,
                direction: TraversalDirection,
                graph: *Self,
                queue: PriorityQueue,
                seen: NodeSet,

                /// Frees the backing allocations for the iterator.
                pub fn deinit(iter: *ThisIterator) void {
                    iter.queue.deinit();
                    iter.seen.deinit(iter.graph.allocator);
                }

                /// Returns an empty Dijkstra iterator beginning at 'start' with given direction.
                /// Note that the start node has not been visited yet, but will be the
                /// first one to be visited.
                pub fn init(graph: *Self, options: InitOptions) Allocator.Error!ThisIterator {
                    var iter = ThisIterator{
                        .allocator = options.allocator orelse graph.allocator,
                        .direction = options.direction,
                        .graph = graph,
                        .queue = undefined,
                        .seen = .{},
                    };
                    iter.queue = PriorityQueue.init(iter.allocator, {});
                    try iter.queue.add(.{ .node = options.start, .distance = 0 });
                    return iter;
                }

                /// Returns the node-distance pair of the nearest unvisited node of the iterator.
                pub fn next(iter: *ThisIterator) Allocator.Error!?DijkstraEntry {
                    return while (iter.queue.removeOrNull()) |nearest| {
                        const result = try iter.seen.getOrPut(iter.allocator, nearest.node);
                        if (result.found_existing) continue;
                        const neighbor_map = switch (iter.direction) {
                            .sourcewards => iter.graph.node_map.get(nearest.node).?.source_map,
                            .targetwards => iter.graph.node_map.get(nearest.node).?.target_map,
                        };
                        for (neighbor_map.keys(), neighbor_map.values()) |neighbor, weight| {
                            if (iter.seen.contains(neighbor)) continue;
                            const arithmetic_weight = if (W == ArithmeticWeight) weight else 1;
                            const distance = nearest.distance + arithmetic_weight;
                            try iter.queue.add(.{ .node = neighbor, .distance = distance });
                        }
                        break nearest;
                    } else null;
                }

                /// Returns the next value of the iterator, without traversing to it.
                pub fn peek(iter: *ThisIterator) ?DijkstraEntry {
                    return iter.queue.peek();
                }
                
                /// Resets the iterator to a new starting point and optionally change the direction.
                /// The backing data structures are cleared but their capacity is retained for efficiency.
                pub fn reset(iter: *ThisIterator, options: ResetOptions) void {
                    if (options.direction) |dir| iter.direction = dir;
                    iter.queue.len = 0;
                    iter.queue.add(options.start) catch unreachable;
                    iter.seen.clearRetainingCapacity();
                }
            };
        }
        /// A Dijkstra iterator traverses the graph in order of distance to the beginning node.
        /// Eventually every node in the connected component including start is reached.
        pub fn dijkstraIterator(
            self: *Self,
            Distance: type,
            options: DijkstraIterator(Distance).InitOptions,
        ) Allocator.Error!DijkstraIterator(Distance) {
            return DijkstraIterator(Distance).init(self, options);
        }
        /// A^* iterator, Dijkstra guided by a heuristic function.
        /// Does not assume that the heuristic is admissible or consistent.
        /// Note: This is an experimental feature and is subject to change or removal.
        /// Todo: add flag to store or compute heuristic 'store_heuristic'?
        /// Todo: add flag to assert that the heuristic is consistent,
        ///       this would essentially turn the iterator into a DijkstraIterator.
        pub fn AStarIterator(Distance: type, HeuristicContext: type) type {
            return struct {
                const ThisIterator = @This();
                // The type returned by the A^* iterator.
                const Entry = struct {
                    node: N,
                    tentative: Distance,
                };
                const HeuristicFn = *const fn (HeuristicContext, N) Distance;
                const InitOptions = struct {
                    allocator: ?Allocator = null,
                    context: HeuristicContext,
                    direction: TraversalDirection = .targetwards,
                    heuristic: HeuristicFn,
                    start: N,
                };
                const ResetOptions = struct {
                    context: ?HeuristicContext = null,
                    direction: ?TraversalDirection = null,
                    heuristic: HeuristicFn,
                    start: N,
                };
                
                const QueueEntry = struct {
                    node: N,
                    // An estimate of the distance to the goal, obtained as tentative + heuristic(node).
                    estimate: Distance,
                };
                fn lessThan(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
                    return std.math.order(a.estimate, b.estimate);
                }
                const PriorityQueue = std.PriorityQueue(QueueEntry, void, lessThan);
                const TreeEntry = struct {
                    previous: ?N,
                    // The tentative distance from the starting node.
                    tentative: Distance,
                };
                const Tree = ArrayHashMapUnmanaged(N, TreeEntry, HashContext, store_hash);

		allocator: Allocator,
                context: HeuristicContext,
                direction: TraversalDirection,
                graph: *Self,
                heuristic: HeuristicFn,
                // The following two data structures should always be in sync.
                open: NodeSet,
                queue: PriorityQueue,
                tree: Tree,

                /// Initializes the iterator to traverse in a given direction at a certain starting point.
                pub fn init(graph: *Self, options: InitOptions) Allocator.Error!ThisIterator {
                    var iter = ThisIterator {
                        .allocator = options.allocator orelse graph.allocator,
                        .context = options.context,
                        .direction = options.direction,
                        .graph = graph,
                        .heuristic = options.heuristic,
                        .open = .{},
                        .queue = undefined,
                        .tree = .{},
                    };
                    const start = options.start;
                    try iter.open.put(iter.allocator, start, {});
                    iter.queue = PriorityQueue.init(iter.allocator, {});
                    try iter.queue.add(.{.node = start, .estimate = iter.heuristic(iter.context, start)});
                    try iter.tree.put(iter.allocator, start, .{.previous = null, .tentative = 0});
                    return iter;
                }

                /// Frees the backing data structures.
                pub fn deinit(iter: *ThisIterator) void {
                    iter.open.deinit(iter.allocator);
                    iter.queue.deinit();
                    iter.tree.deinit(iter.allocator);
                }

                pub fn next(iter: *ThisIterator) !?Entry {
                    return while (iter.queue.removeOrNull()) |queue_entry| {
                        if (!iter.open.swapRemove(queue_entry.node)) continue;
                        const neighbor_map = switch (iter.direction) {
                            .sourcewards => iter.graph.node_map.get(queue_entry.node).?.source_map,
                            .targetwards => iter.graph.node_map.get(queue_entry.node).?.target_map,
                        };
                        const tree_entry = iter.tree.get(queue_entry.node).?;
                        for (neighbor_map.keys(), neighbor_map.values()) |neighbor, weight| {
                            const arithmetic_weight = if (W == ArithmeticWeight) weight else 1;
                            const tentative = tree_entry.tentative + arithmetic_weight;
                            const result = try iter.tree.getOrPut(iter.allocator, neighbor);
                            const neighbor_entry = result.value_ptr;
                            if (!result.found_existing or tentative < neighbor_entry.tentative) {
                                neighbor_entry.tentative = tentative;
                                neighbor_entry.previous = queue_entry.node;
                                try iter.queue.add(.{ .node = neighbor, .estimate = tentative + iter.heuristic(iter.context, neighbor) });
                                try iter.open.put(iter.allocator, neighbor, {});
                            }
                        }
                        break .{
                            .node = queue_entry.node,
                            .tentative = tree_entry.tentative,
                        };
                    } else null;
                }

                pub fn peek(iter: *ThisIterator) ?Entry {
                    return while (iter.queue.peek()) |entry| : (_ = iter.queue.removeOrNull()) {
                        if (!iter.open.contains(entry.node)) continue;
                        break .{
                            .node = entry.node,
                            .tentative = entry.estimate - iter.heuristic(iter.context, entry.node),
                        };
                    } else null;
                }
            };
        }
        /// Returns an A^* iterator over the nodes using a given heuristic function.
        /// NOTE: This is an experimental feature and is subject to change or removal.
        pub fn aStarIterator(
            self: *Self,
            Distance: type,
            HeuristicContext: type,
            options: AStarIterator(Distance, HeuristicContext).InitOptions
        ) Allocator.Error!AStarIterator(Distance, HeuristicContext) {
            return AStarIterator(Distance, HeuristicContext).init(self, options);
        }
    };
}

const expect = std.testing.expect;
test "putNode" {
    const ally = std.testing.allocator;
    var graph = AutoDirectedGraph(u64, void, void).init(ally);
    defer graph.deinit();
    try expect(graph.order() == 0);
    try graph.putNode(0, {});
    try graph.putNode(1, {});
    try graph.putNode(0, {});
    try expect(graph.nodes()[0] == 0);
    try expect(graph.nodes()[1] == 1);
}

test "swapRemoveNode" {
    const ally = std.testing.allocator;
    var graph = AutoDirectedGraph(u64, void, void).init(ally);
    defer graph.deinit();
    try expect(graph.order() == 0);
    try graph.putNode(0, {});
    try expect(graph.order() == 1);
    try expect(graph.swapRemoveNode(0));
    try expect(!graph.swapRemoveNode(0));
    try expect(graph.order() == 0);
    try graph.putNode(0, {});
    try graph.putNode(1, {});
    try graph.putEdge(0, 1, {});
    try expect(graph.targets(0).len == 1);
    try expect(graph.sources(1).len == 1);
    try expect(graph.swapRemoveNode(1));
    // swapRemoveNode should clean up all targets and sources.
    try expect(graph.targets(0).len == 0);
}

test "iterators" {
    const ally = std.testing.allocator;
    var tree = AutoDirectedGraph(u64, void, void).init(ally);
    defer tree.deinit();
    try tree.putNode(1, {});
    try tree.putNode(2, {});
    try tree.putNode(3, {});
    try tree.putNode(4, {});
    try tree.putNode(5, {});
    try tree.putEdge(1, 2, {});
    try tree.putEdge(1, 3, {});
    try tree.putEdge(3, 4, {});
    _ = tree.swapRemoveEdge(3, 4);
    try tree.putEdge(3, 4, {});
    try tree.putEdge(3, 5, {});

    // BFS
    const root = 1;
    var bfs = try tree.bfsIterator(.{.start = root});
    defer bfs.deinit();
    while (try bfs.next()) |_| {}
    try expect(std.mem.eql(u64, bfs.seen.keys(), &[_]u64{ 1, 2, 3, 4, 5 }));
    bfs.reset(.{.start = 3});
    try expect(bfs.seen.count() == 0);
    try expect(bfs.peek() == 3);
    while (try bfs.next()) |_| {}
    try expect(std.mem.eql(u64, bfs.seen.keys(), &[_]u64{ 3, 4, 5 }));
    bfs.reset(.{.start = 5});
    bfs.direction = .sourcewards;
    while (try bfs.next()) |_| {}
    try expect(std.mem.eql(u64, bfs.seen.keys(), &[_]u64{ 5, 3, 1 }));

    // DFS
    var dfs = try tree.dfsIterator(.{.start = root});
    defer dfs.deinit();
    while (try dfs.next()) |_| {}
    try expect(std.mem.eql(u64, dfs.seen.keys(), &[_]u64{ 1, 3, 5, 4, 2 }));
    dfs.reset(.{.start = 3});
    try expect(dfs.seen.count() == 0);
    try expect(dfs.peek() == 3);
    while (try dfs.next()) |_| {}
    try expect(std.mem.eql(u64, dfs.seen.keys(), &[_]u64{ 3, 5, 4 }));
    dfs.reset(.{.start = 5, .direction = .sourcewards});
    while (try dfs.next()) |_| {}
    try expect(std.mem.eql(u64, dfs.seen.keys(), &[_]u64{ 5, 3, 1 }));
    
    // IDS
    var ids = try tree.idsIterator(.{.start = root});
    defer ids.deinit();
    var ids_traversal = ArrayList(u64).init(ally);
    defer ids_traversal.deinit();
    while (try ids.next()) |entry| try ids_traversal.append(entry.node);
    try expect(std.mem.eql(u64, ids_traversal.items, &[_]u64{ // Depth 0..3
    	1,
    	1, 3, 2,
    	1, 3, 5, 4, 2,
   	    1, 3, 5, 4, 2,
    }));

    // Dijkstra
    var dijkstra = try tree.dijkstraIterator(u64, .{.start = root});
    defer dijkstra.deinit();
    var distances = ArrayList(u64).init(ally);
    defer distances.deinit();
    while (try dijkstra.next()) |nearest| try distances.append(nearest.distance);
    try expect(std.mem.eql(u64, dijkstra.seen.keys(), &[_]u64{ 1, 2, 3, 4, 5 }));
    try expect(std.mem.eql(u64, distances.items, &[_]u64{ 0, 1, 1, 2, 2 }));
}

// For background, see: https://en.wikipedia.org/wiki/Collatz_conjecture
// We use the reverse construction.
// Note that a BFS or Dijkstra iterator would probably make more sense here.
test "dynamic dls for the collatz graph" {
    const ally = std.testing.allocator;
    
    var collatz = AutoDirectedGraph(u64, u64, void).init(ally);
    defer collatz.deinit();
    const root: u64 = 1;
    try collatz.putNode(root, 0);
    
    // Expected outputs.
    const orbit_max = 7;
    const numbers_within_max = [_]u64{ 1, 2, 4, 8, 16, 32, 5, 64, 10, 128, 21, 20, 3 };
    const orbits_within_max = [_]u64{ 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7 };
 
    // Generate the collatz graph bottom up to a maximum depth of 7. 
    var iter = try collatz.dlsIterator(.{.depth = orbit_max - 1, .direction = .sourcewards, .start = root});
    defer iter.deinit();
    while (iter.peek()) |entry| {
    	const n = entry.node;
        const orbit = entry.depth + 1;
        const a = 2 * n;
        // Todo: implement getOrPutNode.
        if (!collatz.containsNode(a) or orbit < collatz.getNode(a).?) try collatz.putNode(a, orbit);
        try collatz.putEdge(a, n, {});
        if (n % 6 == 4) {
            const b = (n - 1) / 3;
            if (!collatz.containsNode(b) or orbit < collatz.getNode(b).?) try collatz.putNode(b, orbit);
            try collatz.putEdge(b, n, {});
        }
        _ = try iter.next();
    }
    try expect(collatz.order() == numbers_within_max.len);
    for (numbers_within_max) |n| try expect(collatz.containsNode(n));
    for (numbers_within_max, orbits_within_max) |n, o| try expect(collatz.getNode(n).? == o);
}
