"""
best_bot.py - Carnegie Cookoff 2026 Strategic Bot

Architecture based on:
1. Blackboard Pattern - Centralized state management
2. Conflict-Based Search (CBS) - Multi-agent pathfinding
3. Market-Based Task Allocation - Dynamic auction system
4. Behavior Trees - Hierarchical decision making
5. Game-Theoretic Sabotage - Adversarial strategies

Key Design Principles:
- Space-Time A* with reservation tables for deadlock-free pathfinding
- Dynamic task auction replaces static role assignment
- Proactive sabotage when on enemy map
- Strategic switch timing based on game theory
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Set, Any, Callable
from enum import Enum, auto
import heapq

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food
from tiles import Box

# =============================================================================
# CONSTANTS AND LOOKUPS
# =============================================================================

FOOD_BY_NAME = {ft.food_name: ft for ft in FoodType}
FOOD_BY_ID = {ft.food_id: ft for ft in FoodType}

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Planning horizon for CBS (number of turns to plan ahead)
CBS_HORIZON = 15
# Maximum time to spend on CBS computation (iterations)
CBS_MAX_ITERATIONS = 50
# Weights for auction bid calculation
AUCTION_ALPHA = 1.0   # Distance weight
AUCTION_BETA = 50.0   # Switch cost weight
AUCTION_GAMMA = 2.0   # Urgency weight


# =============================================================================
# TASK TYPES
# =============================================================================

class TaskType(Enum):
    """Types of atomic tasks in the task decomposition"""
    BUY_INGREDIENT = auto()
    CHOP = auto()
    COOK = auto()
    TAKE_FROM_PAN = auto()
    PLATE_FOOD = auto()
    SERVE = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    GET_PLATE = auto()
    WASH_DISHES = auto()
    IDLE = auto()
    # Sabotage tasks
    STEAL_FROM_PAN = auto()
    TRASH_ITEM = auto()
    BLOCK_CHOKEPOINT = auto()
    SPACE_DENIAL = auto()


class TaskStatus(Enum):
    """Status of a task in the auction system"""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Task:
    """Represents an atomic task in the task decomposition"""
    task_id: int
    task_type: TaskType
    target_pos: Optional[Tuple[int, int]] = None
    food_type: Optional[FoodType] = None
    order_id: Optional[int] = None
    priority: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[int] = None
    dependencies: List[int] = field(default_factory=list)
    
    def __hash__(self):
        return self.task_id
    
    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False


@dataclass
class SpaceTimeNode:
    """A node in space-time for pathfinding (x, y, t)"""
    x: int
    y: int
    t: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.t))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t
    
    def __lt__(self, other):
        return (self.x, self.y, self.t) < (other.x, other.y, other.t)


@dataclass
class Constraint:
    """A constraint for CBS: agent cannot be at position at time"""
    agent_id: int
    pos: Tuple[int, int]
    time: int


@dataclass
class CBSNode:
    """A node in the CBS constraint tree"""
    constraints: List[Constraint]
    solution: Dict[int, List[Tuple[int, int]]]  # agent_id -> path
    cost: int
    
    def __lt__(self, other):
        return self.cost < other.cost


# =============================================================================
# BLACKBOARD - CENTRALIZED STATE MANAGEMENT
# =============================================================================

class Blackboard:
    """
    Central state repository implementing the Blackboard Pattern.
    All components read from and write to this shared state.
    """
    
    def __init__(self):
        # Map information
        self.tile_cache: Dict[Team, Dict[str, List[Tuple[int, int]]]] = {}
        self.workstations: Dict[Team, Dict[str, Any]] = {}
        self.map_width: int = 0
        self.map_height: int = 0
        
        # Bot state
        self.bot_positions: Dict[int, Tuple[int, int]] = {}
        self.bot_holdings: Dict[int, Any] = {}
        self.bot_paths: Dict[int, List[Tuple[int, int]]] = {}
        self.bot_tasks: Dict[int, Optional[Task]] = {}
        
        # Enemy state
        self.enemy_positions: Dict[int, Tuple[int, int]] = {}
        self.enemy_predicted_positions: Dict[int, List[Tuple[int, int]]] = {}
        
        # Task management
        self.active_tasks: List[Task] = []
        self.task_counter: int = 0
        self.completed_task_ids: Set[int] = set()
        
        # Reservation table for space-time pathfinding
        self.reservation_table: Dict[Tuple[int, int, int], int] = {}  # (x, y, t) -> agent_id
        
        # Turn state
        self.current_turn: int = 0
        self.moved_bots: Set[int] = set()
        self.acted_bots: Set[int] = set()
        
        # Strategic state
        self.is_on_enemy_map: bool = False
        self.switch_used: bool = False
        self.own_team: Optional[Team] = None
        self.map_team: Optional[Team] = None
        
        # Food tracking
        self.food_on_counters: Dict[Tuple[int, int], Dict] = {}
        self.pans_on_cookers: Dict[Tuple[int, int], Dict] = {}
        self.plates_available: int = 0
        
        # Graph analysis for chokepoints
        self.chokepoints: List[Tuple[int, int]] = []
        self.corridor_tiles: Set[Tuple[int, int]] = set()
    
    def reset_turn_state(self, turn: int) -> None:
        """Reset per-turn tracking state"""
        if self.current_turn != turn:
            self.current_turn = turn
            self.moved_bots.clear()
            self.acted_bots.clear()
            # Clear old reservations
            self._cleanup_old_reservations(turn)
    
    def _cleanup_old_reservations(self, current_turn: int) -> None:
        """Remove reservations from past turns"""
        to_remove = [key for key in self.reservation_table if key[2] < current_turn]
        for key in to_remove:
            del self.reservation_table[key]
    
    def new_task_id(self) -> int:
        """Generate a new unique task ID"""
        self.task_counter += 1
        return self.task_counter


# =============================================================================
# SPACE-TIME A* PATHFINDER
# =============================================================================

class SpaceTimeAStar:
    """
    A* pathfinding in space-time (x, y, t) with reservation table support.
    Handles dynamic obstacles and allows wait actions.
    """
    
    def __init__(self, blackboard: Blackboard):
        self.bb = blackboard
    
    @staticmethod
    def chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Chebyshev (chess king) distance"""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    def is_reserved(self, x: int, y: int, t: int, agent_id: int) -> bool:
        """Check if position is reserved by another agent at time t"""
        key = (x, y, t)
        reserved_by = self.bb.reservation_table.get(key)
        return reserved_by is not None and reserved_by != agent_id
    
    def find_path(
        self,
        controller: RobotController,
        map_team: Team,
        agent_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        start_time: int = 0,
        constraints: Optional[List[Constraint]] = None,
        max_time: int = CBS_HORIZON
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to goal using Space-Time A*.
        Returns list of (x, y) positions or None if no path found.
        """
        if constraints is None:
            constraints = []
        
        # Build constraint lookup
        constraint_set = set()
        for c in constraints:
            if c.agent_id == agent_id:
                constraint_set.add((c.pos[0], c.pos[1], c.time))
        
        # A* in space-time
        open_set = []
        start_node = SpaceTimeNode(start[0], start[1], start_time)
        goal_h = self.chebyshev(start, goal)
        heapq.heappush(open_set, (goal_h, 0, start_node))
        
        came_from: Dict[SpaceTimeNode, Optional[SpaceTimeNode]] = {start_node: None}
        g_score: Dict[SpaceTimeNode, int] = {start_node: 0}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # Goal check (must be adjacent to goal since we can't stand on unwalkable tiles)
            if self.chebyshev((current.x, current.y), goal) <= 1:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append((node.x, node.y))
                    node = came_from[node]
                return list(reversed(path))
            
            # Time limit
            if current.t >= start_time + max_time:
                continue
            
            # Explore neighbors (including WAIT action)
            neighbors = [(0, 0)]  # Wait in place
            for dx, dy in DIRECTIONS:
                neighbors.append((dx, dy))
            
            for dx, dy in neighbors:
                nx, ny = current.x + dx, current.y + dy
                nt = current.t + 1
                
                # Check bounds
                if not (0 <= nx < self.bb.map_width and 0 <= ny < self.bb.map_height):
                    continue
                
                # Check walkability (skip for wait action at current position)
                if dx != 0 or dy != 0:
                    tile = controller.get_tile(map_team, nx, ny)
                    if tile is None or not tile.is_walkable:
                        continue
                
                # Check constraints
                if (nx, ny, nt) in constraint_set:
                    continue
                
                # Check reservation table
                if self.is_reserved(nx, ny, nt, agent_id):
                    continue
                
                # Check edge conflict (swapping positions)
                if self.is_reserved(current.x, current.y, nt, agent_id):
                    continue
                
                neighbor = SpaceTimeNode(nx, ny, nt)
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.chebyshev((nx, ny), goal)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        
        return None
    
    def reserve_path(self, agent_id: int, path: List[Tuple[int, int]], start_time: int) -> None:
        """Reserve the path in the reservation table"""
        for i, (x, y) in enumerate(path):
            t = start_time + i
            self.bb.reservation_table[(x, y, t)] = agent_id


# =============================================================================
# CONFLICT-BASED SEARCH (CBS)
# =============================================================================

class ConflictBasedSearch:
    """
    Conflict-Based Search for optimal multi-agent pathfinding.
    Finds collision-free paths for all agents.
    """
    
    def __init__(self, blackboard: Blackboard, pathfinder: SpaceTimeAStar):
        self.bb = blackboard
        self.pathfinder = pathfinder
    
    def find_conflict(
        self,
        solution: Dict[int, List[Tuple[int, int]]]
    ) -> Optional[Tuple[int, int, Tuple[int, int], int]]:
        """
        Find the first conflict in the solution.
        Returns (agent1, agent2, position, time) or None if no conflict.
        """
        agents = list(solution.keys())
        max_len = max(len(path) for path in solution.values()) if solution else 0
        
        for t in range(max_len):
            positions_at_t: Dict[Tuple[int, int], int] = {}
            
            for agent_id in agents:
                path = solution[agent_id]
                pos = path[min(t, len(path) - 1)]
                
                if pos in positions_at_t:
                    return (positions_at_t[pos], agent_id, pos, t)
                positions_at_t[pos] = agent_id
            
            # Check edge conflicts (agents swapping positions)
            if t > 0:
                for i, agent1 in enumerate(agents):
                    for agent2 in agents[i+1:]:
                        path1 = solution[agent1]
                        path2 = solution[agent2]
                        
                        pos1_t = path1[min(t, len(path1) - 1)]
                        pos1_tm1 = path1[min(t-1, len(path1) - 1)]
                        pos2_t = path2[min(t, len(path2) - 1)]
                        pos2_tm1 = path2[min(t-1, len(path2) - 1)]
                        
                        if pos1_t == pos2_tm1 and pos2_t == pos1_tm1:
                            return (agent1, agent2, pos1_t, t)
        
        return None
    
    def solve(
        self,
        controller: RobotController,
        map_team: Team,
        agent_goals: Dict[int, Tuple[int, int]],
        start_time: int = 0
    ) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        """
        Find collision-free paths for all agents using CBS.
        Returns dictionary mapping agent_id to path, or None if no solution.
        """
        # Initial solution: plan independently
        initial_solution = {}
        for agent_id, goal in agent_goals.items():
            start = self.bb.bot_positions.get(agent_id)
            if start is None:
                continue
            path = self.pathfinder.find_path(
                controller, map_team, agent_id, start, goal, start_time
            )
            if path is None:
                # Fallback: stay in place
                path = [start]
            initial_solution[agent_id] = path
        
        if not initial_solution:
            return {}
        
        # CBS search
        root = CBSNode(
            constraints=[],
            solution=initial_solution,
            cost=sum(len(p) for p in initial_solution.values())
        )
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < CBS_MAX_ITERATIONS:
            iterations += 1
            node = heapq.heappop(open_list)
            
            conflict = self.find_conflict(node.solution)
            if conflict is None:
                return node.solution
            
            agent1, agent2, pos, time = conflict
            
            # Branch on both agents
            for agent_id in [agent1, agent2]:
                new_constraint = Constraint(agent_id, pos, time)
                new_constraints = node.constraints + [new_constraint]
                
                # Replan for constrained agent
                new_solution = dict(node.solution)
                goal = agent_goals.get(agent_id)
                start = self.bb.bot_positions.get(agent_id)
                
                if goal and start:
                    new_path = self.pathfinder.find_path(
                        controller, map_team, agent_id, start, goal,
                        start_time, new_constraints
                    )
                    if new_path:
                        new_solution[agent_id] = new_path
                        new_cost = sum(len(p) for p in new_solution.values())
                        new_node = CBSNode(new_constraints, new_solution, new_cost)
                        heapq.heappush(open_list, new_node)
        
        # Return best found solution if timeout
        return initial_solution


# =============================================================================
# TASK AUCTIONEER - MARKET-BASED ALLOCATION
# =============================================================================

class TaskAuctioneer:
    """
    Market-based task allocation using auction mechanism.
    Dynamically assigns tasks to bots based on bid costs.
    """
    
    def __init__(self, blackboard: Blackboard, pathfinder: SpaceTimeAStar):
        self.bb = blackboard
        self.pathfinder = pathfinder
    
    def decompose_order(
        self,
        controller: RobotController,
        order: Dict,
        map_team: Team
    ) -> List[Task]:
        """Decompose an order into atomic tasks based on recipe DAG"""
        tasks = []
        required = self._required_food_types(order)
        
        # Check what we already have
        ready_counts = self._count_ready_food(controller, map_team)
        missing: List[FoodType] = []
        
        for ft in required:
            if ready_counts.get(ft.food_id, 0) > 0:
                ready_counts[ft.food_id] -= 1
            else:
                missing.append(ft)
        
        # Create tasks for missing ingredients
        for ft in missing:
            # Check if ingredient is on counter needing processing
            raw_pos = self._find_raw_ingredient(controller, map_team, ft)
            
            if raw_pos:
                if ft.can_chop and not self._is_chopped_at(controller, map_team, raw_pos):
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.CHOP,
                        target_pos=raw_pos,
                        food_type=ft,
                        order_id=order["order_id"],
                        priority=order["reward"] * 1.2
                    ))
                elif ft.can_cook:
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.COOK,
                        target_pos=raw_pos,
                        food_type=ft,
                        order_id=order["order_id"],
                        priority=order["reward"] * 1.5
                    ))
            else:
                # Need to buy ingredient
                tasks.append(Task(
                    task_id=self.bb.new_task_id(),
                    task_type=TaskType.BUY_INGREDIENT,
                    food_type=ft,
                    order_id=order["order_id"],
                    priority=order["reward"]
                ))
        
        # Check if we need a plate
        if not self._has_plate(controller, map_team):
            tasks.append(Task(
                task_id=self.bb.new_task_id(),
                task_type=TaskType.GET_PLATE,
                order_id=order["order_id"],
                priority=order["reward"] * 0.8
            ))
        
        # Add serve task
        if not missing:
            tasks.append(Task(
                task_id=self.bb.new_task_id(),
                task_type=TaskType.SERVE,
                order_id=order["order_id"],
                priority=order["reward"] * 2.0
            ))
        
        return tasks
    
    def calculate_bid(
        self,
        controller: RobotController,
        map_team: Team,
        bot_id: int,
        task: Task
    ) -> float:
        """Calculate bid cost for bot to perform task"""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return float('inf')
        
        bot_pos = (bot_state["x"], bot_state["y"])
        holding = bot_state.get("holding")
        
        # Determine task location
        task_location = self._get_task_location(controller, map_team, task)
        if task_location is None:
            return float('inf')
        
        # Distance cost
        dist = self.pathfinder.chebyshev(bot_pos, task_location)
        
        # Switch cost (penalty for needing to drop current item)
        switch_cost = 0
        if holding is not None:
            if not self._holding_compatible_with_task(holding, task):
                switch_cost = AUCTION_BETA
        
        # Urgency cost (inverse of time remaining)
        urgency = 0
        if task.order_id is not None:
            orders = controller.get_orders(self.bb.own_team)
            for o in orders:
                if o["order_id"] == task.order_id:
                    time_left = max(1, o["expires_turn"] - controller.get_turn())
                    urgency = AUCTION_GAMMA * (100.0 / time_left)
                    break
        
        return AUCTION_ALPHA * dist + switch_cost - urgency
    
    def assign_tasks(
        self,
        controller: RobotController,
        map_team: Team,
        bot_ids: List[int],
        tasks: List[Task]
    ) -> Dict[int, Task]:
        """Assign tasks to bots using greedy auction"""
        if not tasks or not bot_ids:
            return {}
        
        # Calculate all bids
        bids: List[Tuple[float, int, Task]] = []
        for task in tasks:
            if task.status != TaskStatus.PENDING:
                continue
            for bot_id in bot_ids:
                cost = self.calculate_bid(controller, map_team, bot_id, task)
                if cost < float('inf'):
                    bids.append((cost, bot_id, task))
        
        # Sort by lowest cost
        bids.sort(key=lambda x: x[0])
        
        # Greedy assignment
        assignments: Dict[int, Task] = {}
        assigned_bots: Set[int] = set()
        assigned_tasks: Set[int] = set()
        
        for cost, bot_id, task in bids:
            if bot_id not in assigned_bots and task.task_id not in assigned_tasks:
                assignments[bot_id] = task
                assigned_bots.add(bot_id)
                assigned_tasks.add(task.task_id)
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = bot_id
        
        return assignments
    
    def _get_task_location(
        self,
        controller: RobotController,
        map_team: Team,
        task: Task
    ) -> Optional[Tuple[int, int]]:
        """Get the target location for a task"""
        ws = self.bb.workstations.get(map_team, {})
        
        if task.target_pos:
            return task.target_pos
        
        if task.task_type == TaskType.BUY_INGREDIENT:
            return ws.get("shop")
        elif task.task_type == TaskType.BUY_PAN:
            return ws.get("shop")
        elif task.task_type == TaskType.BUY_PLATE:
            return ws.get("shop")
        elif task.task_type == TaskType.GET_PLATE:
            sinktable = ws.get("sinktable")
            if sinktable:
                tile = controller.get_tile(map_team, sinktable[0], sinktable[1])
                if tile and getattr(tile, "num_clean_plates", 0) > 0:
                    return sinktable
            return ws.get("shop")
        elif task.task_type == TaskType.COOK:
            return ws.get("cooker")
        elif task.task_type == TaskType.SERVE:
            return ws.get("submit")
        elif task.task_type == TaskType.CHOP:
            return task.target_pos or ws.get("prep_counter")
        elif task.task_type == TaskType.WASH_DISHES:
            return ws.get("sink")
        
        return None
    
    def _holding_compatible_with_task(self, holding: Dict, task: Task) -> bool:
        """Check if current holding is compatible with task"""
        holding_type = holding.get("type")
        
        if task.task_type == TaskType.BUY_INGREDIENT:
            return False  # Must be empty
        elif task.task_type == TaskType.PLATE_FOOD:
            return holding_type == "Plate"
        elif task.task_type == TaskType.SERVE:
            return holding_type == "Plate"
        elif task.task_type == TaskType.COOK:
            return holding_type == "Food" and holding.get("cooked_stage", 0) == 0
        
        return True
    
    def _required_food_types(self, order: Dict) -> List[FoodType]:
        """Get required food types for an order"""
        req = []
        for name in order.get("required", []):
            ft = FOOD_BY_NAME.get(name)
            if ft is not None:
                req.append(ft)
        return req
    
    def _count_ready_food(
        self,
        controller: RobotController,
        map_team: Team
    ) -> Dict[int, int]:
        """Count ready food items"""
        counts: Dict[int, int] = defaultdict(int)
        
        # Check counters
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                ft = FOOD_BY_ID.get(item.food_id)
                if ft and self._food_is_ready(item, ft):
                    counts[item.food_id] += 1
        
        return counts
    
    def _food_is_ready(self, food: Food, ft: FoodType) -> bool:
        """Check if food is ready (properly chopped/cooked)"""
        if ft.can_chop and not food.chopped:
            return False
        if ft.can_cook and food.cooked_stage != 1:
            return False
        return True
    
    def _find_raw_ingredient(
        self,
        controller: RobotController,
        map_team: Team,
        ft: FoodType
    ) -> Optional[Tuple[int, int]]:
        """Find raw ingredient on counter"""
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food) and item.food_id == ft.food_id:
                if not self._food_is_ready(item, ft):
                    return pos
        return None
    
    def _is_chopped_at(
        self,
        controller: RobotController,
        map_team: Team,
        pos: Tuple[int, int]
    ) -> bool:
        """Check if food at position is chopped"""
        tile = controller.get_tile(map_team, pos[0], pos[1])
        item = getattr(tile, "item", None) if tile else None
        return isinstance(item, Food) and item.chopped
    
    def _has_plate(
        self,
        controller: RobotController,
        map_team: Team
    ) -> bool:
        """Check if team has a plate available"""
        # Check counters
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Plate) and not item.dirty:
                return True
        
        # Check sinktable
        ws = self.bb.workstations.get(map_team, {})
        sinktable = ws.get("sinktable")
        if sinktable:
            tile = controller.get_tile(map_team, sinktable[0], sinktable[1])
            if tile and getattr(tile, "num_clean_plates", 0) > 0:
                return True
        
        # Check bot holdings
        for bid in controller.get_team_bot_ids(self.bb.own_team):
            state = controller.get_bot_state(bid)
            holding = state.get("holding") if state else None
            if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
                return True
        
        return False


# =============================================================================
# BEHAVIOR TREE NODES
# =============================================================================

class BTStatus(Enum):
    """Status returned by behavior tree nodes"""
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BTNode:
    """Base class for behavior tree nodes"""
    def execute(self, controller: RobotController, bot_id: int, bb: Blackboard) -> BTStatus:
        raise NotImplementedError


class Selector(BTNode):
    """Tries children in order until one succeeds"""
    def __init__(self, children: List[BTNode]):
        self.children = children
    
    def execute(self, controller: RobotController, bot_id: int, bb: Blackboard) -> BTStatus:
        for child in self.children:
            status = child.execute(controller, bot_id, bb)
            if status == BTStatus.SUCCESS:
                return BTStatus.SUCCESS
            if status == BTStatus.RUNNING:
                return BTStatus.RUNNING
        return BTStatus.FAILURE


class Sequence(BTNode):
    """Runs children in order until one fails"""
    def __init__(self, children: List[BTNode]):
        self.children = children
    
    def execute(self, controller: RobotController, bot_id: int, bb: Blackboard) -> BTStatus:
        for child in self.children:
            status = child.execute(controller, bot_id, bb)
            if status == BTStatus.FAILURE:
                return BTStatus.FAILURE
            if status == BTStatus.RUNNING:
                return BTStatus.RUNNING
        return BTStatus.SUCCESS


class Condition(BTNode):
    """Checks a condition"""
    def __init__(self, check: Callable[[RobotController, int, Blackboard], bool]):
        self.check = check
    
    def execute(self, controller: RobotController, bot_id: int, bb: Blackboard) -> BTStatus:
        return BTStatus.SUCCESS if self.check(controller, bot_id, bb) else BTStatus.FAILURE


class Action(BTNode):
    """Executes an action"""
    def __init__(self, action: Callable[[RobotController, int, Blackboard], BTStatus]):
        self.action = action
    
    def execute(self, controller: RobotController, bot_id: int, bb: Blackboard) -> BTStatus:
        return self.action(controller, bot_id, bb)


# =============================================================================
# SABOTAGE MODULE
# =============================================================================

class SabotageModule:
    """
    Game-theoretic sabotage strategies when on enemy map.
    Implements resource denial, pan snatching, and traffic blocking.
    """
    
    def __init__(self, blackboard: Blackboard, pathfinder: SpaceTimeAStar):
        self.bb = blackboard
        self.pathfinder = pathfinder
    
    def get_sabotage_tasks(
        self,
        controller: RobotController,
        map_team: Team
    ) -> List[Task]:
        """Generate sabotage tasks for enemy map"""
        tasks = []
        
        # Pan snatching: steal food from enemy cookers
        cookers = self.bb.tile_cache.get(map_team, {}).get("COOKER", [])
        for pos in cookers:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(pan.food, Food):
                # Higher priority for cooked food (more damage)
                priority = 200 if pan.food.cooked_stage == 1 else 100
                tasks.append(Task(
                    task_id=self.bb.new_task_id(),
                    task_type=TaskType.STEAL_FROM_PAN,
                    target_pos=pos,
                    priority=priority
                ))
        
        # Counter stealing: grab plates and food
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, (Plate, Food)):
                tasks.append(Task(
                    task_id=self.bb.new_task_id(),
                    task_type=TaskType.TRASH_ITEM,
                    target_pos=pos,
                    priority=50
                ))
        
        # Chokepoint blocking
        for pos in self.bb.chokepoints[:2]:  # Only top 2 chokepoints
            tasks.append(Task(
                task_id=self.bb.new_task_id(),
                task_type=TaskType.BLOCK_CHOKEPOINT,
                target_pos=pos,
                priority=30
            ))
        
        return tasks
    
    def execute_sabotage(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task
    ) -> bool:
        """Execute a sabotage task"""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]
        ws = self.bb.workstations.get(map_team, {})
        trash_pos = ws.get("trash")
        
        # If holding something, trash it first
        if holding:
            if task.task_type != TaskType.SPACE_DENIAL:
                if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                return True
        
        if task.task_type == TaskType.STEAL_FROM_PAN:
            if task.target_pos and self._move_towards(controller, bot_id, map_team, task.target_pos):
                controller.take_from_pan(bot_id, task.target_pos[0], task.target_pos[1])
            return True
        
        elif task.task_type == TaskType.TRASH_ITEM:
            if task.target_pos and self._move_towards(controller, bot_id, map_team, task.target_pos):
                controller.pickup(bot_id, task.target_pos[0], task.target_pos[1])
            return True
        
        elif task.task_type == TaskType.BLOCK_CHOKEPOINT:
            if task.target_pos:
                self._move_towards(controller, bot_id, map_team, task.target_pos)
            return True
        
        return False
    
    def _move_towards(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        target: Tuple[int, int]
    ) -> bool:
        """Move towards target, return True if adjacent"""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        
        bx, by = bot_state["x"], bot_state["y"]
        dist = self.pathfinder.chebyshev((bx, by), target)
        
        if dist <= 1:
            return True
        
        if bot_id in self.bb.moved_bots:
            return False
        
        # Simple greedy movement
        best_move = None
        best_dist = dist
        
        for dx, dy in DIRECTIONS:
            if controller.can_move(bot_id, dx, dy):
                new_dist = self.pathfinder.chebyshev((bx + dx, by + dy), target)
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_move = (dx, dy)
        
        if best_move:
            controller.move(bot_id, best_move[0], best_move[1])
            self.bb.moved_bots.add(bot_id)
            return best_dist <= 1
        
        return False


# =============================================================================
# SWITCH STRATEGY MODULE
# =============================================================================

class SwitchStrategy:
    """
    Game-theoretic switch timing optimization.
    Determines optimal moment to switch to enemy map.
    
    IMPORTANT: Switching too early hurts production. Only switch when:
    1. Late game AND significantly behind
    2. Production is no longer viable
    """
    
    def __init__(self, blackboard: Blackboard):
        self.bb = blackboard
    
    def should_switch(self, controller: RobotController) -> bool:
        """Determine if we should switch maps now"""
        if not controller.can_switch_maps():
            return False
        
        turn = controller.get_turn()
        own_team = controller.get_team()
        enemy_team = controller.get_enemy_team()
        
        my_money = controller.get_team_money(own_team)
        enemy_money = controller.get_team_money(enemy_team)
        
        # Don't switch before turn 400 - focus on production
        if turn < 400:
            return False
        
        # Only switch in very late game if significantly behind
        # Scenario 1: End-game (turn > 450) and significantly behind
        if turn > 450 and my_money + 100 < enemy_money:
            return True
        
        # Scenario 2: Very late game (turn > 480) and any deficit
        if turn > 480 and my_money < enemy_money:
            return True
        
        # Don't switch just because enemy has high-value orders
        # That's their business, focus on our own production
        
        return False
    
    def calculate_switch_value(
        self,
        controller: RobotController
    ) -> float:
        """Calculate expected value of switching now"""
        turn = controller.get_turn()
        own_team = controller.get_team()
        enemy_team = controller.get_enemy_team()
        
        my_money = controller.get_team_money(own_team)
        enemy_money = controller.get_team_money(enemy_team)
        
        # Base value from money difference
        money_diff = enemy_money - my_money
        
        # Discount for lost production time
        remaining_turns = GameConstants.TOTAL_TURNS - turn
        production_loss = min(50, remaining_turns * 0.5)
        
        # Bonus for late game (less production to lose)
        late_game_bonus = max(0, (turn - 350) * 0.5)
        
        return money_diff - production_loss + late_game_bonus


# =============================================================================
# MAIN BOT PLAYER CLASS
# =============================================================================

class BotPlayer:
    """
    Main bot controller implementing the complete strategic architecture.
    """
    
    def __init__(self, map_copy):
        # Initialize blackboard
        self.bb = Blackboard()
        self.bb.map_width = map_copy.width
        self.bb.map_height = map_copy.height
        
        # Initialize modules
        self.pathfinder = SpaceTimeAStar(self.bb)
        self.cbs = ConflictBasedSearch(self.bb, self.pathfinder)
        self.auctioneer = TaskAuctioneer(self.bb, self.pathfinder)
        self.sabotage = SabotageModule(self.bb, self.pathfinder)
        self.switch_strategy = SwitchStrategy(self.bb)
        
        # Build behavior tree
        self.behavior_tree = self._build_behavior_tree()
        
        # Configuration
        self.enable_switch = True
        self.disable_sabotage = False
    
    def _build_behavior_tree(self) -> BTNode:
        """Build the main behavior tree for decision making"""
        return Selector([
            # Priority 1: Sabotage mode when on enemy map
            Sequence([
                Condition(self._is_on_enemy_map),
                Action(self._execute_sabotage_action)
            ]),
            # Priority 2: Emergency - order expiring soon
            Sequence([
                Condition(self._has_urgent_order),
                Action(self._execute_urgent_delivery)
            ]),
            # Priority 3: Execute assigned task
            Action(self._execute_assigned_task)
        ])
    
    # =========================================================================
    # BEHAVIOR TREE CONDITIONS
    # =========================================================================
    
    def _is_on_enemy_map(
        self,
        controller: RobotController,
        bot_id: int,
        bb: Blackboard
    ) -> bool:
        """Check if bot is on enemy map"""
        return bb.is_on_enemy_map and not self.disable_sabotage
    
    def _has_urgent_order(
        self,
        controller: RobotController,
        bot_id: int,
        bb: Blackboard
    ) -> bool:
        """Check if there's an urgently expiring order"""
        turn = controller.get_turn()
        orders = controller.get_orders(bb.own_team)
        
        for order in orders:
            if order.get("is_active"):
                time_left = order["expires_turn"] - turn
                if time_left < 20:
                    return True
        return False
    
    # =========================================================================
    # BEHAVIOR TREE ACTIONS
    # =========================================================================
    
    def _execute_sabotage_action(
        self,
        controller: RobotController,
        bot_id: int,
        bb: Blackboard
    ) -> BTStatus:
        """Execute sabotage actions on enemy map"""
        tasks = self.sabotage.get_sabotage_tasks(controller, bb.map_team)
        if not tasks:
            return BTStatus.FAILURE
        
        # Pick highest priority sabotage task
        task = max(tasks, key=lambda t: t.priority)
        self.sabotage.execute_sabotage(controller, bot_id, bb.map_team, task)
        return BTStatus.SUCCESS
    
    def _execute_urgent_delivery(
        self,
        controller: RobotController,
        bot_id: int,
        bb: Blackboard
    ) -> BTStatus:
        """Rush delivery of expiring order"""
        # This will be handled by task execution with elevated priority
        return BTStatus.FAILURE
    
    def _execute_assigned_task(
        self,
        controller: RobotController,
        bot_id: int,
        bb: Blackboard
    ) -> BTStatus:
        """Execute the task assigned via auction"""
        task = bb.bot_tasks.get(bot_id)
        if task is None:
            return BTStatus.FAILURE
        
        self._execute_task(controller, bot_id, bb.map_team, task)
        return BTStatus.SUCCESS
    
    # =========================================================================
    # MAP INITIALIZATION
    # =========================================================================
    
    def _ensure_tile_cache(self, controller: RobotController, map_team: Team) -> None:
        """Build tile cache for map"""
        if map_team in self.bb.tile_cache:
            return
        
        positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        m = controller.get_map(map_team)
        self.bb.map_width = m.width
        self.bb.map_height = m.height
        
        for x in range(m.width):
            for y in range(m.height):
                tile = controller.get_tile(map_team, x, y)
                if tile is None:
                    continue
                positions[tile.tile_name].append((x, y))
        
        self.bb.tile_cache[map_team] = positions
        
        # Build workstation references
        submit_pos = positions.get("SUBMIT", [])
        cooker_pos = positions.get("COOKER", [])
        counter_pos = positions.get("COUNTER", [])
        shop_pos = positions.get("SHOP", [])
        sink_pos = positions.get("SINK", [])
        sinktable_pos = positions.get("SINKTABLE", [])
        trash_pos = positions.get("TRASH", [])
        
        def nearest_to(tile_name: str, targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
            if not targets:
                return None
            best = None
            best_dist = 10**9
            for pos in positions.get(tile_name, []):
                for t in targets:
                    dist = self.pathfinder.chebyshev(pos, t)
                    if dist < best_dist:
                        best_dist = dist
                        best = pos
            return best
        
        prep_counter = nearest_to("COUNTER", cooker_pos) if cooker_pos else (counter_pos[0] if counter_pos else None)
        plate_counter = nearest_to("COUNTER", submit_pos) if submit_pos else (counter_pos[0] if counter_pos else None)
        
        self.bb.workstations[map_team] = {
            "prep_counter": prep_counter,
            "plate_counter": plate_counter,
            "cooker": cooker_pos[0] if cooker_pos else None,
            "cookers": cooker_pos,
            "shop": shop_pos[0] if shop_pos else None,
            "submit": submit_pos[0] if submit_pos else None,
            "sink": sink_pos[0] if sink_pos else None,
            "sinktable": sinktable_pos[0] if sinktable_pos else None,
            "trash": trash_pos[0] if trash_pos else None,
        }
        
        # Identify chokepoints (tiles with degree 2)
        self._identify_chokepoints(controller, map_team)
    
    def _identify_chokepoints(self, controller: RobotController, map_team: Team) -> None:
        """Identify chokepoint tiles for strategic blocking"""
        chokepoints = []
        floor_tiles = self.bb.tile_cache.get(map_team, {}).get("FLOOR", [])
        
        for pos in floor_tiles:
            neighbors = 0
            for dx, dy in DIRECTIONS:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self.bb.map_width and 0 <= ny < self.bb.map_height:
                    tile = controller.get_tile(map_team, nx, ny)
                    if tile and tile.is_walkable:
                        neighbors += 1
            
            # Corridors have exactly 2 walkable neighbors
            if neighbors == 2:
                self.bb.corridor_tiles.add(pos)
                chokepoints.append(pos)
        
        # Sort by centrality (distance from center)
        center = (self.bb.map_width // 2, self.bb.map_height // 2)
        chokepoints.sort(key=lambda p: self.pathfinder.chebyshev(p, center))
        self.bb.chokepoints = chokepoints
    
    # =========================================================================
    # TASK EXECUTION
    # =========================================================================
    
    def _execute_task(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task
    ) -> None:
        """Execute a specific task"""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]
        ws = self.bb.workstations.get(map_team, {})
        
        if task.task_type == TaskType.BUY_INGREDIENT:
            self._execute_buy_ingredient(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.CHOP:
            self._execute_chop(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.COOK:
            self._execute_cook(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.GET_PLATE:
            self._execute_get_plate(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.PLATE_FOOD:
            self._execute_plate_food(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.SERVE:
            self._execute_serve(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.BUY_PAN:
            self._execute_buy_pan(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.TAKE_FROM_PAN:
            self._execute_take_from_pan(controller, bot_id, map_team, task, holding)
        elif task.task_type == TaskType.WASH_DISHES:
            self._execute_wash_dishes(controller, bot_id, map_team, task, holding)
    
    def _execute_buy_ingredient(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute buy ingredient task"""
        ws = self.bb.workstations.get(map_team, {})
        shop_pos = ws.get("shop")
        
        # If holding the ingredient we need (or any food), check what to do with it
        if holding and holding.get("type") == "Food":
            ft = task.food_type
            if ft:
                # If this food needs chopping, take it to a counter
                if ft.can_chop and not holding.get("chopped"):
                    counter = self._find_empty_counter(controller, map_team, ws.get("prep_counter"))
                    if counter and self._move_towards(controller, bot_id, map_team, counter):
                        controller.place(bot_id, counter[0], counter[1])
                    return
                # If this food needs cooking, take it to a cooker with empty pan
                if ft.can_cook and holding.get("cooked_stage", 0) == 0:
                    # Find empty pan
                    cookers = ws.get("cookers", [])
                    for ck_pos in cookers:
                        tile = controller.get_tile(map_team, ck_pos[0], ck_pos[1])
                        pan = getattr(tile, "item", None) if tile else None
                        if isinstance(pan, Pan) and pan.food is None:
                            if self._move_towards(controller, bot_id, map_team, ck_pos):
                                controller.place(bot_id, ck_pos[0], ck_pos[1])
                            return
                    # No empty pan - place on counter
                    counter = self._find_empty_counter(controller, map_team, ws.get("prep_counter"))
                    if counter and self._move_towards(controller, bot_id, map_team, counter):
                        controller.place(bot_id, counter[0], counter[1])
                    return
            # Food is ready or doesn't need processing - try to plate it
            plate_pos = self._find_plate_on_counter(controller, map_team)
            if plate_pos:
                if self._move_towards(controller, bot_id, map_team, plate_pos):
                    controller.add_food_to_plate(bot_id, plate_pos[0], plate_pos[1])
                return
            # No plate - place food on counter
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding something else (pan, plate), put it down first
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, shop_pos)
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Buy the ingredient
        if shop_pos and task.food_type:
            if self._move_towards(controller, bot_id, map_team, shop_pos):
                if controller.get_team_money(self.bb.own_team) >= task.food_type.buy_cost:
                    controller.buy(bot_id, task.food_type, shop_pos[0], shop_pos[1])
    
    def _execute_chop(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute chop task"""
        ws = self.bb.workstations.get(map_team, {})
        
        # If holding food that needs chopping, place it on counter first
        if holding and holding.get("type") == "Food":
            if not holding.get("chopped"):
                # Food needs chopping - place on counter
                counter = self._find_empty_counter(controller, map_team, ws.get("prep_counter"))
                if counter and self._move_towards(controller, bot_id, map_team, counter):
                    controller.place(bot_id, counter[0], counter[1])
                return
            # Food is already chopped - handle appropriately
            ft = FOOD_BY_ID.get(holding.get("food_id"))
            if ft and ft.can_cook and holding.get("cooked_stage", 0) == 0:
                # Needs cooking next - find empty pan
                cookers = ws.get("cookers", [])
                for ck_pos in cookers:
                    tile = controller.get_tile(map_team, ck_pos[0], ck_pos[1])
                    pan = getattr(tile, "item", None) if tile else None
                    if isinstance(pan, Pan) and pan.food is None:
                        if self._move_towards(controller, bot_id, map_team, ck_pos):
                            controller.place(bot_id, ck_pos[0], ck_pos[1])
                        return
            # Place on counter for now
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding something else, put it down
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, ws.get("prep_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Chop food at target position
        if task.target_pos:
            if self._move_towards(controller, bot_id, map_team, task.target_pos):
                controller.chop(bot_id, task.target_pos[0], task.target_pos[1])
    
    def _execute_cook(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute cook task"""
        ws = self.bb.workstations.get(map_team, {})
        cookers = ws.get("cookers", [])
        
        # Find empty pan
        empty_cooker = None
        for ck_pos in cookers:
            tile = controller.get_tile(map_team, ck_pos[0], ck_pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and pan.food is None:
                empty_cooker = ck_pos
                break
        
        # If holding food that needs cooking, put it in a pan
        if holding and holding.get("type") == "Food":
            if holding.get("cooked_stage", 0) == 0:  # Uncooked
                if empty_cooker and self._move_towards(controller, bot_id, map_team, empty_cooker):
                    controller.place(bot_id, empty_cooker[0], empty_cooker[1])
                return
            else:
                # Food is already cooked - place on counter
                counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
                if counter and self._move_towards(controller, bot_id, map_team, counter):
                    controller.place(bot_id, counter[0], counter[1])
                return
        
        # If holding something else, put it down first
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, ws.get("prep_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Pick up food to cook from target position
        if task.target_pos:
            tile = controller.get_tile(map_team, task.target_pos[0], task.target_pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                if self._move_towards(controller, bot_id, map_team, task.target_pos):
                    controller.pickup(bot_id, task.target_pos[0], task.target_pos[1])
            return
        
        # Try to find food that needs cooking on counters
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food) and item.cooked_stage == 0:
                ft = FOOD_BY_ID.get(item.food_id)
                if ft and ft.can_cook:
                    # Check if chopping is needed first
                    if ft.can_chop and not item.chopped:
                        continue  # Needs chopping first
                    if self._move_towards(controller, bot_id, map_team, pos):
                        controller.pickup(bot_id, pos[0], pos[1])
                    return
    
    def _execute_get_plate(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute get plate task"""
        ws = self.bb.workstations.get(map_team, {})
        sinktable_pos = ws.get("sinktable")
        shop_pos = ws.get("shop")
        
        # If already holding a clean plate, place it near submit station
        if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
            # Place the plate on a counter near submit for easy access
            counter = self._find_empty_counter(controller, map_team, ws.get("submit"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding something else, need to put it down first
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Try sinktable first (free plates)
        if sinktable_pos:
            tile = controller.get_tile(map_team, sinktable_pos[0], sinktable_pos[1])
            if tile and getattr(tile, "num_clean_plates", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, sinktable_pos):
                    controller.take_clean_plate(bot_id, sinktable_pos[0], sinktable_pos[1])
                return
        
        # Buy from shop
        if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
            if controller.get_team_money(self.bb.own_team) >= ShopCosts.PLATE.buy_cost:
                controller.buy(bot_id, ShopCosts.PLATE, shop_pos[0], shop_pos[1])
    
    def _execute_plate_food(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute plate food task"""
        ws = self.bb.workstations.get(map_team, {})
        
        if holding and holding.get("type") == "Plate":
            # We have a plate - find ready food to add
            food_pos = self._find_ready_food(controller, map_team, task.food_type)
            if food_pos and self._move_towards(controller, bot_id, map_team, food_pos):
                controller.add_food_to_plate(bot_id, food_pos[0], food_pos[1])
            return
        
        if holding and holding.get("type") == "Food":
            # We have food - find plate to add to
            plate_pos = self._find_plate_on_counter(controller, map_team)
            if plate_pos:
                if self._move_towards(controller, bot_id, map_team, plate_pos):
                    controller.add_food_to_plate(bot_id, plate_pos[0], plate_pos[1])
                return
            # No plate on counter - place food and get a plate instead
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Not holding anything useful - pick up food from target or find ready food
        if task.target_pos:
            if self._move_towards(controller, bot_id, map_team, task.target_pos):
                controller.pickup(bot_id, task.target_pos[0], task.target_pos[1])
        else:
            food_pos = self._find_ready_food(controller, map_team, task.food_type)
            if food_pos and self._move_towards(controller, bot_id, map_team, food_pos):
                controller.pickup(bot_id, food_pos[0], food_pos[1])
    def _execute_serve(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute serve task"""
        ws = self.bb.workstations.get(map_team, {})
        submit_pos = ws.get("submit")
        
        # If holding a clean plate (with or without food), try to submit
        if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
            if submit_pos and self._move_towards(controller, bot_id, map_team, submit_pos):
                controller.submit(bot_id, submit_pos[0], submit_pos[1])
            return
        
        # If holding something else, put it down first
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, submit_pos)
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Pick up plate with food
        plate_pos = self._find_plate_on_counter(controller, map_team)
        if plate_pos and self._move_towards(controller, bot_id, map_team, plate_pos):
            controller.pickup(bot_id, plate_pos[0], plate_pos[1])
    
    def _execute_buy_pan(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute buy pan task"""
        ws = self.bb.workstations.get(map_team, {})
        shop_pos = ws.get("shop")
        cooker_pos = ws.get("cooker")
        
        if holding and holding.get("type") == "Pan":
            # Place on cooker
            if cooker_pos and self._move_towards(controller, bot_id, map_team, cooker_pos):
                controller.place(bot_id, cooker_pos[0], cooker_pos[1])
            return
        
        if holding is not None:
            return
        
        if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
            if controller.get_team_money(self.bb.own_team) >= ShopCosts.PAN.buy_cost:
                controller.buy(bot_id, ShopCosts.PAN, shop_pos[0], shop_pos[1])
    
    def _execute_take_from_pan(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute take from pan task"""
        ws = self.bb.workstations.get(map_team, {})
        
        # If holding food, try to plate it or place it
        if holding and holding.get("type") == "Food":
            # Check if burnt - trash it
            if holding.get("cooked_stage", 0) == 2:
                trash_pos = ws.get("trash")
                if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                return
            # Try to add to plate
            plate_pos = self._find_plate_on_counter(controller, map_team)
            if plate_pos:
                if self._move_towards(controller, bot_id, map_team, plate_pos):
                    controller.add_food_to_plate(bot_id, plate_pos[0], plate_pos[1])
                return
            # Place on counter
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding something else, put it down
        if holding is not None:
            counter = self._find_empty_counter(controller, map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # Take food from pan
        if task.target_pos and self._move_towards(controller, bot_id, map_team, task.target_pos):
            controller.take_from_pan(bot_id, task.target_pos[0], task.target_pos[1])
    
    def _execute_wash_dishes(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        task: Task,
        holding: Optional[Dict]
    ) -> None:
        """Execute wash dishes task"""
        ws = self.bb.workstations.get(map_team, {})
        sink_pos = ws.get("sink")
        
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return
        
        if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
            tile = controller.get_tile(map_team, sink_pos[0], sink_pos[1])
            if tile and getattr(tile, "num_dirty_plates", 0) > 0:
                controller.wash_sink(bot_id, sink_pos[0], sink_pos[1])
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _move_towards(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        target: Tuple[int, int]
    ) -> bool:
        """Move bot towards target. Returns True if adjacent."""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        
        bx, by = bot_state["x"], bot_state["y"]
        dist = self.pathfinder.chebyshev((bx, by), target)
        
        if dist <= 1:
            return True
        
        if bot_id in self.bb.moved_bots:
            return False
        
        # Get blocked positions from other bots
        blocked = set()
        for other_id in controller.get_team_bot_ids(self.bb.own_team):
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other and other.get("map_team") == map_team.name:
                blocked.add((other["x"], other["y"]))
        
        # Use pathfinding to get next step
        path = self.pathfinder.find_path(
            controller, map_team, bot_id, (bx, by), target, self.bb.current_turn
        )
        
        if path and len(path) > 1:
            next_pos = path[1]
            dx, dy = next_pos[0] - bx, next_pos[1] - by
            
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                self.bb.moved_bots.add(bot_id)
                return self.pathfinder.chebyshev(next_pos, target) <= 1
        
        # Fallback: greedy movement
        best_move = None
        best_dist = dist
        
        for dx, dy in DIRECTIONS:
            if controller.can_move(bot_id, dx, dy):
                nx, ny = bx + dx, by + dy
                if (nx, ny) not in blocked:
                    new_dist = self.pathfinder.chebyshev((nx, ny), target)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_move = (dx, dy)
        
        if best_move:
            controller.move(bot_id, best_move[0], best_move[1])
            self.bb.moved_bots.add(bot_id)
            return best_dist <= 1
        
        return False
    
    def _find_empty_counter(
        self,
        controller: RobotController,
        map_team: Team,
        near: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Find an empty counter near the given position"""
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        if not counters:
            return None
        
        if near is None:
            near = counters[0]
        
        best = None
        best_dist = 10**9
        
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or getattr(tile, "item", None) is not None:
                continue
            dist = self.pathfinder.chebyshev(pos, near)
            if dist < best_dist:
                best_dist = dist
                best = pos
        
        return best
    
    def _find_ready_food(
        self,
        controller: RobotController,
        map_team: Team,
        food_type: Optional[FoodType]
    ) -> Optional[Tuple[int, int]]:
        """Find ready food on counters"""
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                if food_type is None:
                    return pos
                if item.food_id == food_type.food_id:
                    ft = FOOD_BY_ID.get(item.food_id)
                    if ft and self._food_is_ready(item, ft):
                        return pos
        
        return None
    
    def _food_is_ready(self, food: Food, ft: FoodType) -> bool:
        """Check if food is ready for plating"""
        if ft.can_chop and not food.chopped:
            return False
        if ft.can_cook and food.cooked_stage != 1:
            return False
        return True
    
    def _find_plate_on_counter(
        self,
        controller: RobotController,
        map_team: Team
    ) -> Optional[Tuple[int, int]]:
        """Find a clean plate on counter"""
        counters = self.bb.tile_cache.get(map_team, {}).get("COUNTER", [])
        
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Plate) and not item.dirty:
                return pos
        
        return None
    
    def _find_box_with_food(
        self,
        controller: RobotController,
        map_team: Team,
        ft: FoodType
    ) -> Optional[Tuple[int, int]]:
        """Find a box containing the specified food type"""
        boxes = self.bb.tile_cache.get(map_team, {}).get("BOX", [])
        
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) <= 0:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and item.food_id == ft.food_id:
                return pos
        
        return None
    
    def _select_best_order(
        self,
        controller: RobotController,
        team: Team
    ) -> Optional[Dict]:
        """Select the best order to work on"""
        orders = controller.get_orders(team)
        if not orders:
            return None
        
        turn = controller.get_turn()
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return None
        
        def score(o: Dict) -> float:
            time_left = max(1, o["expires_turn"] - turn)
            reward = o["reward"]
            penalty = o["penalty"]
            required = self._required_food_types(o)
            num_ingredients = len(required)
            
            # Feasibility check - estimate time needed
            est_time = self._estimate_completion_time(controller, self.bb.map_team, required)
            
            # If not enough time, heavily penalize
            if est_time > time_left - 10:
                return -1000
            
            # Base value: reward potential
            value = reward
            
            # Simplicity bonus: fewer ingredients = easier to complete
            # Prioritize 2-ingredient orders over 5-ingredient ones
            value += (5 - num_ingredients) * 30
            
            # Urgency bonus: closer deadlines get priority (if feasible)
            if time_left < 100:
                value += (100 - time_left) * 0.5
            
            # Penalty avoidance: factor in penalty we'd incur if we fail
            value += penalty * 0.3
            
            return value
        
        scored = [(o, score(o)) for o in active]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best feasible order
        for o, s in scored:
            if s > -1000:
                return o
        
        # If no feasible orders, return the simplest one
        return min(active, key=lambda o: len(self._required_food_types(o)))
    
    def _required_food_types(self, order: Dict) -> List[FoodType]:
        """Get required food types for order"""
        req = []
        for name in order.get("required", []):
            ft = FOOD_BY_NAME.get(name)
            if ft is not None:
                req.append(ft)
        return req
    
    def _estimate_completion_time(
        self,
        controller: RobotController,
        map_team: Team,
        required: List[FoodType]
    ) -> int:
        """Estimate turns to complete order based on GameConstants"""
        time = 0
        
        for ft in required:
            if ft.can_cook:
                time += 25  # Cook time is 20 turns + movement buffer
            if ft.can_chop:
                time += 8   # Chop takes several interactions + movement
            time += 10      # Buy + movement per ingredient
        
        time += 15  # Getting a plate + plating all foods
        time += 10  # Moving to submit + submitting
        
        return time
    
    # =========================================================================
    # MAIN TURN EXECUTION
    # =========================================================================
    
    def play_turn(self, controller: RobotController):
        """Main entry point for turn execution"""
        self.bb.reset_turn_state(controller.get_turn())
        self.bb.own_team = controller.get_team()
        
        bot_ids = controller.get_team_bot_ids(self.bb.own_team)
        if not bot_ids:
            return
        
        # Determine current map
        first_state = controller.get_bot_state(bot_ids[0])
        if first_state is None:
            return
        
        self.bb.map_team = Team[first_state["map_team"]]
        self.bb.is_on_enemy_map = self.bb.map_team != self.bb.own_team
        
        # Initialize map cache
        self._ensure_tile_cache(controller, self.bb.map_team)
        
        # Update bot positions
        for bot_id in bot_ids:
            state = controller.get_bot_state(bot_id)
            if state:
                self.bb.bot_positions[bot_id] = (state["x"], state["y"])
                self.bb.bot_holdings[bot_id] = state.get("holding")
        
        # Check if we should switch maps
        if self.enable_switch and self.switch_strategy.should_switch(controller):
            controller.switch_maps()
            first_state = controller.get_bot_state(bot_ids[0])
            if first_state is None:
                return
            self.bb.map_team = Team[first_state["map_team"]]
            self.bb.is_on_enemy_map = True
            self._ensure_tile_cache(controller, self.bb.map_team)
        
        # Sabotage mode
        if self.bb.is_on_enemy_map and not self.disable_sabotage:
            self._execute_sabotage_mode(controller, bot_ids)
            return
        
        # Production mode
        self._execute_production_mode(controller, bot_ids)
    
    def _execute_sabotage_mode(
        self,
        controller: RobotController,
        bot_ids: List[int]
    ) -> None:
        """Execute sabotage strategy on enemy map"""
        tasks = self.sabotage.get_sabotage_tasks(controller, self.bb.map_team)
        
        for bot_id in bot_ids:
            if tasks:
                # Assign task with highest priority
                task = max(tasks, key=lambda t: t.priority)
                tasks.remove(task)
                self.sabotage.execute_sabotage(controller, bot_id, self.bb.map_team, task)
            else:
                # Default: block near submit station
                ws = self.bb.workstations.get(self.bb.map_team, {})
                submit = ws.get("submit")
                if submit:
                    self._move_towards(controller, bot_id, self.bb.map_team, submit)
    
    def _execute_production_mode(
        self,
        controller: RobotController,
        bot_ids: List[int]
    ) -> None:
        """Execute production strategy on own map"""
        # Select best order
        order = self._select_best_order(controller, self.bb.own_team)
        if order is None:
            # No orders available - all bots fallback
            for bot_id in bot_ids:
                self._execute_fallback_behavior(controller, bot_id, [])
            return
        
        required = self._required_food_types(order)
        
        # Generate tasks for order - duplicate important tasks for parallel work
        tasks = self._generate_production_tasks(controller, order, required)
        
        # If we have fewer tasks than bots, generate additional helper tasks
        # This ensures both bots stay productive
        if len(tasks) < len(bot_ids):
            # Add additional ingredient buying tasks for items that need cooking
            for ft in required:
                if ft.can_cook:
                    # Check if we need more of this ingredient
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.BUY_INGREDIENT,
                        food_type=ft,
                        order_id=order["order_id"],
                        priority=60  # Lower priority helper task
                    ))
        
        # Assign tasks via auction
        assignments = self.auctioneer.assign_tasks(
            controller, self.bb.map_team, bot_ids, tasks
        )
        
        # Update blackboard
        self.bb.bot_tasks = {bid: assignments.get(bid) for bid in bot_ids}
        
        # Execute assigned tasks (or fallback behavior)
        for bot_id in bot_ids:
            task = self.bb.bot_tasks.get(bot_id)
            if task:
                self._execute_task(controller, bot_id, self.bb.map_team, task)
            else:
                # No task assigned - execute fallback
                self._execute_fallback_behavior(controller, bot_id, required)
    
    def _generate_production_tasks(
        self,
        controller: RobotController,
        order: Dict,
        required: List[FoodType]
    ) -> List[Task]:
        """Generate all tasks needed for the current order"""
        tasks = []
        ws = self.bb.workstations.get(self.bb.map_team, {})
        
        # Check what's already in progress or ready
        plate_obj = self._get_team_plate(controller)
        missing = self._get_missing_ingredients(controller, required, plate_obj)
        
        # PRIORITY 1: Ensure we have a plate FIRST (highest priority)
        # This is critical - we can't serve without a plate
        if not plate_obj:
            tasks.append(Task(
                task_id=self.bb.new_task_id(),
                task_type=TaskType.GET_PLATE,
                order_id=order["order_id"],
                priority=250  # High priority - get plate before anything else
            ))
        
        # PRIORITY 2: Check if plate is complete for serving
        if not missing and plate_obj:
            if self._plate_matches_order(plate_obj, required):
                tasks.append(Task(
                    task_id=self.bb.new_task_id(),
                    task_type=TaskType.SERVE,
                    order_id=order["order_id"],
                    priority=300  # Highest priority - deliver completed order
                ))
        
        # PRIORITY 3: Check for cooked food ready to be taken out
        cookers = ws.get("cookers", [])
        for ck_pos in cookers:
            tile = controller.get_tile(self.bb.map_team, ck_pos[0], ck_pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(pan.food, Food):
                if pan.food.cooked_stage == 1:
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.TAKE_FROM_PAN,
                        target_pos=ck_pos,
                        priority=220  # Very high - don't let food burn
                    ))
                elif pan.food.cooked_stage == 2:  # Burnt
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.TAKE_FROM_PAN,
                        target_pos=ck_pos,
                        priority=180  # Remove burnt food to free up pan
                    ))
        
        # PRIORITY 4: Add plating tasks for ready food (only if we have a plate)
        if plate_obj:
            ready_for_plating = self._find_ready_foods_for_plating(controller, required, plate_obj)
            for food_pos, ft in ready_for_plating:
                tasks.append(Task(
                    task_id=self.bb.new_task_id(),
                    task_type=TaskType.PLATE_FOOD,
                    target_pos=food_pos,
                    food_type=ft,
                    order_id=order["order_id"],
                    priority=200  # High priority to get food on plate
                ))
        
        # PRIORITY 5: Ensure we have pans on cookers for cookable items
        needs_cooking = any(ft.can_cook for ft in missing)
        if needs_cooking:
            for ck_pos in cookers:
                tile = controller.get_tile(self.bb.map_team, ck_pos[0], ck_pos[1])
                pan = getattr(tile, "item", None) if tile else None
                if not isinstance(pan, Pan):
                    tasks.append(Task(
                        task_id=self.bb.new_task_id(),
                        task_type=TaskType.BUY_PAN,
                        target_pos=ck_pos,
                        priority=150
                    ))
                    break  # Only need one pan task at a time
        
        # PRIORITY 6: Tasks for missing ingredients (only if we have or are getting a plate)
        for ft in missing:
            # Check if on counter needs processing
            raw_pos = self._find_ingredient_needing_processing(controller, ft)
            
            if raw_pos:
                tile = controller.get_tile(self.bb.map_team, raw_pos[0], raw_pos[1])
                item = getattr(tile, "item", None) if tile else None
                
                if isinstance(item, Food):
                    if ft.can_chop and not item.chopped:
                        tasks.append(Task(
                            task_id=self.bb.new_task_id(),
                            task_type=TaskType.CHOP,
                            target_pos=raw_pos,
                            food_type=ft,
                            order_id=order["order_id"],
                            priority=140
                        ))
                    elif ft.can_cook and item.cooked_stage == 0 and (not ft.can_chop or item.chopped):
                        tasks.append(Task(
                            task_id=self.bb.new_task_id(),
                            task_type=TaskType.COOK,
                            target_pos=raw_pos,
                            food_type=ft,
                            order_id=order["order_id"],
                            priority=130
                        ))
            else:
                # Need to acquire ingredient - only buy if we have plate capacity
                # For items that don't need cooking (noodles, sauce), only buy if plate exists
                if ft.can_cook or plate_obj:
                    box_pos = self._find_box_with_food(controller, self.bb.map_team, ft)
                    if box_pos:
                        tasks.append(Task(
                            task_id=self.bb.new_task_id(),
                            task_type=TaskType.BUY_INGREDIENT,
                            target_pos=box_pos,
                            food_type=ft,
                            order_id=order["order_id"],
                            priority=100 if ft.can_cook else 80  # Cookable items higher priority
                        ))
                    else:
                        tasks.append(Task(
                            task_id=self.bb.new_task_id(),
                            task_type=TaskType.BUY_INGREDIENT,
                            food_type=ft,
                            order_id=order["order_id"],
                            priority=100 if ft.can_cook else 80
                        ))
        
        return tasks
    
    def _execute_fallback_behavior(
        self,
        controller: RobotController,
        bot_id: int,
        required: List[FoodType]
    ) -> None:
        """Execute fallback behavior when no task assigned"""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        
        holding = bot_state.get("holding")
        ws = self.bb.workstations.get(self.bb.map_team, {})
        
        # If holding dirty plate, wash it
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            sink_pos = ws.get("sink")
            if sink_pos and self._move_towards(controller, bot_id, self.bb.map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return
        
        # If holding burnt food, trash it
        if holding and holding.get("type") == "Food" and holding.get("cooked_stage", 0) == 2:
            trash_pos = ws.get("trash")
            if trash_pos and self._move_towards(controller, bot_id, self.bb.map_team, trash_pos):
                controller.trash(bot_id, trash_pos[0], trash_pos[1])
            return
        
        # CRITICAL FIX: If holding food with no plate, place it on a counter
        # This prevents the bot from getting stuck holding food forever
        if holding and holding.get("type") == "Food":
            # Try to add food to plate on counter
            plate_pos = self._find_plate_on_counter(controller, self.bb.map_team)
            if plate_pos:
                if self._move_towards(controller, bot_id, self.bb.map_team, plate_pos):
                    controller.add_food_to_plate(bot_id, plate_pos[0], plate_pos[1])
                return
            
            # No plate - place food on empty counter for later
            counter = self._find_empty_counter(controller, self.bb.map_team, ws.get("plate_counter"))
            if counter and self._move_towards(controller, bot_id, self.bb.map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding a clean plate, find food to add to it
        if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
            food_pos = self._find_ready_food(controller, self.bb.map_team, None)
            if food_pos:
                if self._move_towards(controller, bot_id, self.bb.map_team, food_pos):
                    controller.add_food_to_plate(bot_id, food_pos[0], food_pos[1])
                return
            # No food ready, place plate on counter near submit
            counter = self._find_empty_counter(controller, self.bb.map_team, ws.get("submit"))
            if counter and self._move_towards(controller, bot_id, self.bb.map_team, counter):
                controller.place(bot_id, counter[0], counter[1])
            return
        
        # If holding pan, place it on a cooker
        if holding and holding.get("type") == "Pan":
            cookers = ws.get("cookers", [])
            for ck_pos in cookers:
                tile = controller.get_tile(self.bb.map_team, ck_pos[0], ck_pos[1])
                if tile and getattr(tile, "item", None) is None:
                    if self._move_towards(controller, bot_id, self.bb.map_team, ck_pos):
                        controller.place(bot_id, ck_pos[0], ck_pos[1])
                    return
        
        # Wash dishes if dirty plates in sink
        sink_pos = ws.get("sink")
        if sink_pos:
            tile = controller.get_tile(self.bb.map_team, sink_pos[0], sink_pos[1])
            if tile and getattr(tile, "num_dirty_plates", 0) > 0:
                if self._move_towards(controller, bot_id, self.bb.map_team, sink_pos):
                    controller.wash_sink(bot_id, sink_pos[0], sink_pos[1])
                return
        
        # Idle near shop to be ready for next task
        shop_pos = ws.get("shop")
        if shop_pos:
            self._move_towards(controller, bot_id, self.bb.map_team, shop_pos)
    
    def _get_team_plate(self, controller: RobotController) -> Optional[Any]:
        """Get plate held by team or on counter"""
        # Check counters
        counters = self.bb.tile_cache.get(self.bb.map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(self.bb.map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Plate) and not item.dirty:
                return item
        
        # Check bot holdings
        for bid in controller.get_team_bot_ids(self.bb.own_team):
            state = controller.get_bot_state(bid)
            holding = state.get("holding") if state else None
            if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
                return holding
        
        return None
    
    def _get_missing_ingredients(
        self,
        controller: RobotController,
        required: List[FoodType],
        plate_obj: Optional[Any]
    ) -> List[FoodType]:
        """Get list of ingredients still needed"""
        remaining = list(required)
        
        # Remove foods already on plate
        if plate_obj:
            if isinstance(plate_obj, Plate):
                for f in plate_obj.food:
                    if isinstance(f, Food):
                        for i, ft in enumerate(remaining):
                            if f.food_id == ft.food_id and self._food_is_ready(f, ft):
                                remaining.pop(i)
                                break
            elif isinstance(plate_obj, dict):
                for f in plate_obj.get("food", []):
                    for i, ft in enumerate(remaining):
                        if f.get("food_id") == ft.food_id:
                            # Check if ready
                            if (not ft.can_chop or f.get("chopped")) and \
                               (not ft.can_cook or f.get("cooked_stage") == 1):
                                remaining.pop(i)
                                break
        
        # Remove foods ready on counters
        counters = self.bb.tile_cache.get(self.bb.map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(self.bb.map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                ft = FOOD_BY_ID.get(item.food_id)
                if ft and self._food_is_ready(item, ft):
                    for i, req_ft in enumerate(remaining):
                        if req_ft.food_id == item.food_id:
                            remaining.pop(i)
                            break
        
        # Remove foods cooking in pans
        cookers = self.bb.workstations.get(self.bb.map_team, {}).get("cookers", [])
        for ck_pos in cookers:
            tile = controller.get_tile(self.bb.map_team, ck_pos[0], ck_pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(pan.food, Food):
                for i, req_ft in enumerate(remaining):
                    if req_ft.food_id == pan.food.food_id and pan.food.cooked_stage <= 1:
                        remaining.pop(i)
                        break
        
        return remaining
    
    def _find_ingredient_needing_processing(
        self,
        controller: RobotController,
        ft: FoodType
    ) -> Optional[Tuple[int, int]]:
        """Find ingredient on counter that needs processing"""
        counters = self.bb.tile_cache.get(self.bb.map_team, {}).get("COUNTER", [])
        
        for pos in counters:
            tile = controller.get_tile(self.bb.map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food) and item.food_id == ft.food_id:
                if ft.can_chop and not item.chopped:
                    return pos
                if ft.can_cook and item.cooked_stage == 0:
                    if not ft.can_chop or item.chopped:
                        return pos
        
        return None
    
    def _plate_matches_order(self, plate_obj: Any, required: List[FoodType]) -> bool:
        """Check if plate contents match order requirements"""
        if plate_obj is None:
            return False
        
        plate_foods: List[Tuple[int, bool, int]] = []
        
        if isinstance(plate_obj, Plate):
            for f in plate_obj.food:
                if isinstance(f, Food):
                    plate_foods.append((f.food_id, f.chopped, f.cooked_stage))
        elif isinstance(plate_obj, dict):
            for f in plate_obj.get("food", []):
                plate_foods.append((
                    f.get("food_id"),
                    f.get("chopped", False),
                    f.get("cooked_stage", 0)
                ))
        
        required_foods = []
        for ft in required:
            cooked = 1 if ft.can_cook else 0
            required_foods.append((ft.food_id, ft.can_chop, cooked))
        
        return sorted(plate_foods) == sorted(required_foods)
    
    def _find_ready_foods_for_plating(
        self,
        controller: RobotController,
        required: List[FoodType],
        plate_obj: Optional[Any]
    ) -> List[Tuple[Tuple[int, int], FoodType]]:
        """Find ready foods that should be plated"""
        result = []
        missing = self._get_missing_ingredients(controller, required, plate_obj)
        
        counters = self.bb.tile_cache.get(self.bb.map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(self.bb.map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                for ft in missing:
                    if item.food_id == ft.food_id and self._food_is_ready(item, ft):
                        result.append((pos, ft))
                        break
        
        return result
