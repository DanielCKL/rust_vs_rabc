# Rust implementation of Vector-Selection Reinforcement ABC 

- This is a re-implementation of my Master's Degree final Year Project in Rust. 

- ABC stands for 'Artificial Bee Colony', a search/optimization algorithm first proposed by Karaboga et al. It is a generalized optimization algorithm that allows us to find the best set of parameters for an optimization problem.

- Faster convergence for benchmarks like the Rosenbrock and Rastrigin benchmark functions, on par with the other functions for other more elaborate benchmarks

- The only thing that is necessary is to have a fitness function that outputs datatype f64. This algorithm can search for the best value of your fitness function (it can be an optimization problem or a training model) without any prior knowledge apart from the boundary values of each parameter or dimension.

# How to use
1. Download this github repository into your computer
2. Import the library into your project using `mod rust_vs_rabc`
3. Set up a new instance of the optimizer. Here, we will name this instance 'optimize_rana':

       let mut optimize_rana = Optimizer::new().minimize().set_thread_pool(2);
       //the new() method sets up default parameter values
       //the .minimize() method sets the optimizer to search for a minimum value
       //the .set_thread_pool() method sets the optimizer to use multiple thread pools in parallel.
4. Fill in or set the struct fields to customize the running of its methods. Otherwise, default values will be used.
  
       //For example:
    
       Fields that can affect the run:
         -permanent_scout_bees ()
       Optional fields that do not affect the run, and that can be added for further reference
         -fitness_function_name
         -fitness_function_description
         -known_minimum_value
         -known_minimum_point
6. Call the optimization algorithm as below:

           optimize_rana.vs_rabc(
                    &problem_space_bounds, //The problem space bounds (type Vec![f64,f64] (upper bound and lower bound))
                    35u64,                 //Max number of generations to run this algorithm (u64)
                    benchmark_algos::rana, //Fitness function that returns an f64 output.
                                           //If it doesn't return f64, wrap it in another function and only take/cast f64.
                );
7. The results will be stored in the `Optimizer` object (`optimize_rana`) that we created earlier. To access them, you can use the following fields:
- real_time_taken        = The time taken for the algorithm to be run. This is hardware-dependent.
- searches_made_history  = Recorded number of searches made at each iteration
- total_searches_made    = The total number of searches made. This is the best overall measure of performance.
- min_max_value_history  = An 
- min_max_point_history  =

# Upcoming Work
- Add logic to allow early stopping of search based on how long the 
- Add logic to track average time the algorithm did not find any new minimum (palateuated/stagnated around a local minimum)
