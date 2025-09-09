import argparse
import os

from pypibt import (
    MultiActionPIBT,
    TaskManager,
    get_grid,
    get_agents,
    get_tasks,
    is_valid_oriented_mapf_solution,
    save_oriented_configs_for_visualizer,
    oriented_config_to_config,
    SolutionVerifier,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "simple_test.map"
        ),
    )
    parser.add_argument(
        "-a",
        "--agents-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "test_agents.csv"
        ),
    )
    parser.add_argument(
        "-t",
        "--tasks-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "test_tasks.csv"
        ),
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=1000)
    parser.add_argument("--enable-logging", action="store_true", help="Enable JSON logging")
    parser.add_argument("--log-file", type=str, default="pibt_results.json", help="Log file path")
    parser.add_argument("--cycle-tasks", action="store_true", help="Enable task cycling (default: True)", default=True)
    parser.add_argument("--disable-verification", action="store_true", help="Disable solution verification")
    args = parser.parse_args()

    # Load grid, agents, and tasks
    grid = get_grid(args.map_file)
    agents = get_agents(args.agents_file)
    tasks = get_tasks(args.tasks_file, len(grid[0]))
    print(f"Loaded {len(tasks)}")
    
    # Limit number of agents if specified
    if args.num_agents is not None:
        agents = agents[:args.num_agents]
    
    print(f"Loaded grid: {grid.shape}")
    print(f"Number of agents: {len(agents)}")
    print(f"Number of tasks: {len(tasks)}")
    
    # Extract starts
    starts = [(agent[0], agent[1]) for agent in agents]
    
    # Create TaskManager for dynamic task assignment
    task_manager = TaskManager(
        task_pool=tasks,
        num_agents=len(agents),
        cycle_tasks=args.cycle_tasks
    )
    
    # Create MultiActionPIBT instance with TaskManager
    pibt = MultiActionPIBT(
        grid=grid,
        starts=starts,
        task_manager=task_manager,
        seed=args.seed,
        enable_logging=args.enable_logging,
        enable_verification=not args.disable_verification
    )
    
    print(f"Using TaskManager with {len(task_manager.task_pool)} tasks")
    print(f"Task cycling: {'enabled' if args.cycle_tasks else 'disabled'}")
    
    # Show initial task assignments
    print("\nInitial task assignments:")
    for i in range(len(agents)):
        task = task_manager.get_task_for_agent(i)
        if task:
            task_id, y, x = task
            # Use original task ID for display (remove internal cycle offset)
            display_task_id = task_id % 1000
            print(f"  Agent {i}: Task {display_task_id} at position ({y}, {x})")
    
    print(f"Initialized MultiActionPIBT with {pibt.N} agents")
    
    # Run the algorithm
    log_filepath = args.log_file if args.enable_logging else None
    oriented_plan = pibt.run(max_timestep=args.max_timestep, log_filepath=log_filepath)
    
    # Get final positions
    final_positions = [(pos[0], pos[1]) for pos in oriented_plan[-1]]
    
    print(f"Planning completed in {len(oriented_plan)-1} timesteps")
    print(f"Final positions: {final_positions}")
    
    # Show task completion statistics
    stats = pibt.task_manager.get_stats()
    print(f"\nTask completion statistics:")
    print(f"  Total tasks in pool: {stats['total_tasks']}")
    print(f"  Tasks completed: {stats['completed_tasks']}")
    print(f"  Tasks still active: {stats['active_assignments']}")
    print(f"  Total assignment/completion events: {stats['total_events']}")
    print(f"  Cycle count: {stats['cycle_count']}")
    
    # Show current task assignments
    print(f"\nFinal task assignments:")
    for i in range(len(agents)):
        task = pibt.task_manager.get_task_for_agent(i)
        if task:
            task_id, y, x = task
            display_task_id = task_id % 1000
            print(f"  Agent {i}: Task {display_task_id} at position ({y}, {x})")
    
    # Save result for visualization
    save_oriented_configs_for_visualizer(oriented_plan, args.output_file)
    print(f"Results saved to: {args.output_file}")
    
    if args.enable_logging:
        events = pibt.task_manager.get_assignment_events()
        print(f"\nTask assignment events logged: {len(events)}")
        print(f"Sample task assignment events (first 10):")
        for i, event in enumerate(events[:10]):
            if len(event) >= 4:
                task_id, agent_id, action, timestep = event
                print(f"  {i+1}. Timestep {timestep}: Task {task_id} {action} to/by agent {agent_id}")
            else:
                task_id, agent_id, action = event[:3]
                print(f"  {i+1}. Task {task_id} {action} to/by agent {agent_id}")
        
        print(f"\nFull logging results saved to: {args.log_file}")
