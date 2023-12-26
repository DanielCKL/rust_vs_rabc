// TODO: Fine-tune the upper/lower bounds setting system.
//TODO: Include a non-parallel mode in case the API/interface being used cannot handle parallelism.
//submodule for benchmark algorithms

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]

mod benchmark_algos;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use rayon::prelude::*;
//use std::thread;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

#[derive(Default, Debug)]
pub struct Optimizer {
    //COMPULSORY Optimizer parameters
    pub max_generations: u64,                //CANNOT be running FOREVER
    pub problem_space_bounds: Vec<[f64; 2]>, //Cannot possibly be unbounded! Assumed stored as inclusive always [ub,lb]

    //optional parameters
    pub employed_bees: usize,
    pub onlooker_bees: usize,
    pub permanent_scout_bees: usize, //If None (default), will be set by algorithm itself
    maximize: bool,                  //if true, maximize, if false, minimize
    pub local_limit: usize, //Limit for how many times a food source can be exploited before being abandoned.
    pub problem_space_bounds_inclusivity: String,
    pub ignore_nan: bool, //ignore nan values from input function. Will almost certainly lead to unpredictable behavior.
    pub thread_pool_size: usize, //size of the thread pool

    //Problem Space metadata
    //Calculated from length of problem_space_bounds.
    pub number_of_dimensions: usize,
    //Optional Problem Space Metadata
    pub known_minimum_value: Option<f64>, //The minimum value of the function, if known (already-solved real-world problem/known test function). Defaults to Option::None.
    pub known_minimum_point: Option<Vec<f64>>, //The minimum point coordinates of the function, if known (already-solved real-world problem/known test function). Defaults to Option::None.
    pub fitness_function_name: String,         //can be name of test function/real-world problem
    pub fitness_function_description: String,

    //Optimization Algorithm metadata
    pub algorithm_name: String, //Name of the algorithm being used
    pub algorithm_description: String,

    //Performance data to be written AFTER the algorithm has finished running.
    pub searches_made: u64, //how many iterations have been run. Default of 0
    pub min_max_value: f64, //the minimum/maximum reward value found within problem space (single value). Default of 0.0
    pub min_max_point: Vec<f64>, //vector solution that will return min_max_value.
    pub real_time_taken: Duration, //Real time taken. Default of 0.0
}

impl Optimizer {
    //2 different sets of params: 1. Compulsory , 2. optional. All optional to be inside an enum

    //input of all such methods:
    //An n-dimensional vector containing our guess
    //output:
    //A single f64 for greatest precision

    //Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems

    //Number of dimensions is unknown, so easier to pass in as vec, but a lower & upper bound is necessary/easily set & known, so pass in as array.

    //sources: https://www.researchgate.net/publication/225392029_A_powerful_and_efficient_algorithm_for_numerical_function_optimization_Artificial_bee_colony_ABC_algorithm ,
    //https://www.mdpi.com/2227-7390/7/3/289
    //https://ijci.journals.ekb.eg/article_33956_00e14724271769d23b7067336027d6de.pdf
    //default constructor ->MUST be used to create an instance of Optimizer
    pub fn new() -> Self {
        Self {
            employed_bees: 62usize, //Default values for employed, onlooker, and scout bees as recommended in the source paper.
            onlooker_bees: 62usize,
            permanent_scout_bees: 1usize,
            local_limit: 220usize, //550usize seems to be the most optimal
            maximize: true,       //default to finding maximum value in input problem space
            min_max_value: f64::NEG_INFINITY,
            problem_space_bounds_inclusivity: "[]".to_string(), //default to inclusive upper and lower
            thread_pool_size: available_parallelism().unwrap().get(),
            ..Default::default()
        }
    }

    // resets the other metadata and allows you to carry on with the same settings
    //for a fresh run, just create a new() instance in a new scope, or drop the old instance
    pub fn clear(mut self: Self)-> Self
    {
    self.problem_space_bounds=vec![[0.0f64; 2]]; //Cannot possibly be unbounded! Assumed stored as inclusive always [ub;lb]

    //Optional Problem Space Metadata
    self.known_minimum_value= None; //The minimum value of the function; if known (already-solved real-world problem/known test function). Defaults to Option::None.
    self.known_minimum_point= None; //The minimum point coordinates of the function; if known (already-solved real-world problem/known test function). Defaults to Option::None.
    self.fitness_function_name= String::from("");         //can be name of test function/real-world problem
    self.fitness_function_description= String::from("");

    //Optimization Algorithm metadata
    self.algorithm_name= String::from(""); //Name of the algorithm being used
    self.algorithm_description= String::from("");

    //Performance data to be written AFTER the algorithm has finished running.
    self.searches_made= 0; //how many iterations have been run. Default of 0
    self.min_max_value= f64::NEG_INFINITY; //the minimum/maximum reward value found within problem space (single value). Default of 0.0
    self.min_max_point= vec![]; //vector solution that will return min_max_value.
    self
    }

    //Builder-pattern method to switch from maximization to minimization mode.
    pub fn minimize(mut self: Self) -> Self {
        self.maximize = false;
        self.min_max_value = f64::INFINITY;
        self
    }

    pub fn set_thread_pool(mut self: Self, new_thread_pool_size: usize) -> Self {
        self.thread_pool_size = new_thread_pool_size;
        self
    }

    //Builder-pattern method to set there to be no scout bees in case searches are overly expensive. NOT recommended.
    pub fn noscout(mut self: Self) -> Self {
        self.permanent_scout_bees = 0usize;
        self
    }

    //Taken from https://doc.rust-lang.org/src/core/num/f64.rs.html#740
    pub fn next_up(val: f64) -> f64 {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const TINY_BITS: u64 = 0x1; // Smallest positive f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = val.to_bits();
        if val.is_nan() || bits == f64::INFINITY.to_bits() {
            return val; //return as it is if infinity or nan
        }

        let abs = bits & CLEAR_SIGN_MASK; //Get absolute value (clear the sign bit)
        let next_bits = if abs == 0 {
            TINY_BITS
        } else if bits == abs {
            bits + 1
        } else {
            bits - 1 //if negative, the next number will be LESS negative (-1 comes above -2)
        };
        f64::from_bits(next_bits)
    }

    // vec1 + vec2
    fn add_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a + b).collect();
        println!("{:?}", result);
        result
    }

    // vec1 - vec2
    fn deduct_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a - b).collect();
        println!("{:?}", result);
        result
    }

    // vec1 * vec2
    fn multiply_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).collect();
        println!("{:?}", result);
        result
    }

    //set bounds for input to allow them to comply with the default .. data inside.
    pub fn set_bounds(input: &Vec<[f64; 2]>, mode: &str) -> Vec<[f64; 2]> {
        if mode == "(]" {
            //(lb,ub]  =>Lower bound exclusive, upper bound inclusive =>Setting = "(]"
            input
                .to_vec()
                .iter()
                .map(|x| [Self::next_up(x[0]), Self::next_up(x[1])])
                .collect()
        } else if mode == "()" {
            //(lb,ub) => Lower bound exclusive, upper bound exclusive =>Setting = "()"
            input
                .to_vec()
                .iter()
                .map(|x| [Self::next_up(x[0]), x[1]])
                .collect()
        } else if mode == "[)" {
            //[lb,ub) =>Lower bound inclusive, upper bound exclusive =>Setting = "[)"
            input.to_vec()
        } else {
            //[lb, ub] => Default: Lower bound inclusive, upper bound inclusive =>Setting = "[]"
            input
                .to_vec()
                .iter()
                .map(|x| [(x[0]), Self::next_up(x[1])])
                .collect()
        }
    }

    //Taken from https://doc.rust-lang.org/src/core/num/f64.rs.html#740
    fn next_down(val: f64) -> f64 {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const NEG_TINY_BITS: u64 = 0x8000_0000_0000_0001; // Smallest (in magnitude) negative f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = val.to_bits();
        if val.is_nan() || bits == f64::NEG_INFINITY.to_bits() {
            return val; //return as it is if infinity or nan
        }

        let abs = bits & CLEAR_SIGN_MASK; //Get absolute value (clear the sign bit)
        let next_bits = if abs == 0 {
            NEG_TINY_BITS
        } else if bits == abs {
            bits - 1
        } else {
            bits + 1 //if negative, the next number will be MORE negative. (-2 comes below -1)
        };
        f64::from_bits(next_bits)
    }

    pub fn classic_abc(
        self: &mut Self,
        problem_space_bounds: &Vec<[f64; 2]>,
        max_generations: u64,
        fitness_function: fn(&Vec<f64>) -> f64,
    ) {
        //create an optimizer object first with all the default fields and metadata initialized.
        // let mut instance =
        //     Optimizer::default_initializer(problem_space_bounds, max_generations);//, other_args);
        //Start timer here
        let function_real_time = Instant::now();

        //Set the thread pool size based on available threads.
        if self.thread_pool_size == 1 {
            println!("Running with {} threads. To change, use the builder method set_thread_pool(desired_pool_size). For example: NewOptimizer::new().set_thread_pool(7)",self.thread_pool_size);
        } else {
            println!(
                "Running in parallel with {} threads.",
                self.thread_pool_size
            )
        }
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_pool_size)
            .build()
            .unwrap();

        //Set metadata for this function.
        self.problem_space_bounds = problem_space_bounds.to_vec();
        let adjusted_bounds = Optimizer::set_bounds(
            &self.problem_space_bounds,
            &self.problem_space_bounds_inclusivity,
        );

        self.number_of_dimensions = self.problem_space_bounds.len() as usize;
        self.max_generations = max_generations;
        self.algorithm_name = String::from("classic_abc");
        self.algorithm_description=String::from("Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems");

        //Ensure that the metadata set does not have an unacceptable value.
        if self.employed_bees < 2 {
            panic!("Number of employed bees should be greater than or equal to 2");
        }
        if self.ignore_nan {
            println!("WARNING: flag ignore_nan() is on. The program will not panic upon encountering NaN from the fitness function, but erratic behavior/errors may happen further downstream.");
        }

        //default of this algorithim is maximization, set to 'false' to minimize instead.
        let minmax_factor = if self.maximize { 1.0f64 } else { -1.0f64 };

        //Add actions to ALWAYS be performed with each search to avoid further indirection from closures/function nesting.
        macro_rules! perform_search {
            ($x:expr) => {{
            //perform all other actions here

                let results = fitness_function($x);     //ONLY fitness function runs will be performed in parallel. All else is assumed to be cheap & controllable.
                //self.searches_made += 1;
                let adjusted_results=minmax_factor*results;

                // if (adjusted_results) > (minmax_factor * self.min_max_value) {
                //     self.min_max_value = results;
                //     self.min_max_point = $x.clone();
                // }

                //Check if results are nan
                if adjusted_results.is_nan() {
                    if self.ignore_nan {
                        println!("WARNING: Result from fitness function is NaN at  {:?}! f64::NEG_INFINITY/f64::infinity may be acceptable, but not NAN (0/0). This may lead to unpredictable behavior.",$x);
                        }
                        else{
                        panic!("Value in fitness function was Nan. Input parameters were {:?} Aborting program.", $x);
                        }

                    //may consider
                }
                adjusted_results //MUST return this at the very end
            }};
        }
        macro_rules! update_metadata {
            ($vec_fitness_value:expr,$input_vector:expr,$searches_performed:expr) => {{
                //perform all other actions here
                //Get maximum value and coordinates of fitness value
                let (max_index, max_value) = $vec_fitness_value
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();

                //Note: max_value was previously already multiplied by the minmax_factor

                //Update number of searches made
                self.searches_made += $searches_performed as u64;

                //update metadata for max value and the vector that returns that value

                if (*max_value) > (minmax_factor * self.min_max_value) {
                    self.min_max_value = minmax_factor*max_value;
                    self.min_max_point = $input_vector[max_index].clone();
                }
            }};
        }

        //Set up RNG
        let mut random_generator = rand::thread_rng();

        //Generate value for RabC that is not 0
        //let random_float:f64=random_generator.gen_range(f64::MIN_POSITIVE..1.0);

        //BEGIN ALGORITHM HERE:

        //INITIALIZE e employed bee positions in n dimensions: vec![vec![0.0f64 ... n], ... e]
        let mut employed_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees];

        //Generate intial solutions
        //TODO: Possibly make this parallel as well.
        for each_search_point in employed_bees_searches.iter_mut() {
            for (idx, dimension) in each_search_point.iter_mut().enumerate() {
                //for every single dimension
                *dimension = random_generator
                    .gen_range::<f64, _>(adjusted_bounds[idx][0]..adjusted_bounds[idx][1])
                //generate random values within problem space bounds.
            }
        }

        //Create an intermediate copy of the searches already made.
        let mut trial_search_points = employed_bees_searches.clone();
        //vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees]; //Create an intermediate copy of the searches already made.

        //Perform initial search with employed bees on every single point
        let mut food_source_values: Vec<f64> = thread_pool.install(|| {
            employed_bees_searches
                .par_iter()
                .map(|x| -> f64 { perform_search!(x) })
                .collect()
        });
        let mut normalized_food_source_values = vec![0.0f64; self.employed_bees];

        update_metadata!(
            food_source_values,
            employed_bees_searches,
            employed_bees_searches.len()
        );

        //Create an intermediate vector to hold the searches made for onlooker bees
        let mut onlooker_trial_search_points =
            vec![vec![0.0f64; self.number_of_dimensions]; self.onlooker_bees];
        let mut onlooker_mapping_to_employed = vec![0usize; self.onlooker_bees];

        //create a vector to keep track of number of attempts made per food source
        let mut attempts_per_food_source = vec![1usize; self.employed_bees];

        //INITIALIZE scout bee array

        //Scout bee should search randomly at all times, and every time a
        //better solution is found, update to that solution.

        let mut scout_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.permanent_scout_bees];
        let mut trial_scout_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.permanent_scout_bees];
        let mut scout_food_sources_values: Vec<f64> =
            vec![f64::NEG_INFINITY; self.permanent_scout_bees];

        let mut exceeded_max: Vec<usize> = vec![]; //Blank index that will contain (Index of points > max tries, )
        let mut temporary_scout_searches: Vec<Vec<f64>> = vec![];

        let mut chosen_dimension_vec = vec![0usize; self.employed_bees];
        let mut onlooker_chosen_dimension_vec = vec![0usize; self.onlooker_bees];
        //Loop through the algorithm here
        for iteration in 0..self.max_generations {
            //next: vij = xij + φij(xij − xkj), Modify initial positions and search again:
            //employed_bees_searches=
            //        attempts_per_food_source.iter().map(|x|{x+1});  //add 1 to the number of times we explore a food source
            //xkj is an existing solution

            //println!("\nBefore:\n{:?}", employed_bees_searches);

            for i in 0..self.employed_bees {
                //Karaboga: at each cycle at most one scout ... number of employed and onlooker bees were equal.
                //Number of employed and onlooker bees will remain the same throughout the algorithm.

                //First increment the attempt made for the food source
                attempts_per_food_source[i] += 1;

                //For every single employed bee
                //Select a random chosen_dimension
                let chosen_dimension: usize =
                    random_generator.gen_range(0..self.number_of_dimensions);
                chosen_dimension_vec[i] = chosen_dimension;

                //Select the index for an existing random food source
                let mut random_solution_index: usize =
                    random_generator.gen_range(0..self.employed_bees);
                while random_solution_index == i {
                    random_solution_index = random_generator.gen_range(0..self.employed_bees)
                }

                // Modify initial positions by xij + phi_ij(xij − xkj)
                let existing_sol = employed_bees_searches[i][chosen_dimension];

                //Modify
                let tentative_new = existing_sol
                    + (random_generator.gen_range::<f64, _>(-1.0..1.0000000000000002) //1.0000000000000002 is the smallest next value after 1f64
                        * (existing_sol - employed_bees_searches[random_solution_index][chosen_dimension]));

                //Check if out of bounds, if they are out of bounds set them to those bounds (Karaboga et al)
                let (lower_bound, upper_bound) = (
                    self.problem_space_bounds[chosen_dimension][0],
                    self.problem_space_bounds[chosen_dimension][1],
                );

                if tentative_new < lower_bound {
                    trial_search_points[i][chosen_dimension] = lower_bound;
                } else if tentative_new > upper_bound {
                    trial_search_points[i][chosen_dimension] = upper_bound;
                } else {
                    trial_search_points[i][chosen_dimension] = tentative_new;
                }
            }
            //Run the searches in parallel
            let new_search_vec: Vec<f64> = thread_pool.install(|| {
                trial_search_points
                    .par_iter()
                    .map(|x| perform_search!(x))
                    .collect()
            });
            // println!("attempts_per_food_source before={:?}",attempts_per_food_source);
            // println!("trial_search_points before= {:?}",trial_search_points);

            //Update the food source and trial search point values
            for (idx, new_search) in new_search_vec.iter().enumerate() {
                if *new_search > food_source_values[idx] {
                    // Update to new points if the new source has as higher fitness value
                    employed_bees_searches[idx][chosen_dimension_vec[idx]] =
                        trial_search_points[idx][chosen_dimension_vec[idx]];
                    //Update to new fitness value too
                    food_source_values[idx] = *new_search;
                    //If a better value of the food source was found, set the counter to 0 again
                    attempts_per_food_source[idx] = 0;
                } else {
                    //Revert trial_search_points[i][chosen_dimension_vec[idx]] back to employed_bees_searches[i][chosen_dimension_vec[idx]]
                    //Important because we will have to update from this same value later.
                    trial_search_points[idx][chosen_dimension_vec[idx]] =
                        employed_bees_searches[idx][chosen_dimension_vec[idx]];
                };
            }
            // println!("trial_search_points after= {:?}",trial_search_points);
            // println!("attempts_per_food_source={:?}",attempts_per_food_source);
            // println!("employed_bees_searches={:?}",employed_bees_searches);
            // println!("New Search Vec= {:?}", new_search_vec);
            update_metadata!(
                food_source_values,
                employed_bees_searches,
                self.employed_bees
            );

            //calculate probability for onlooker bees
            //Normalize values (if there are negative values, add the (modulus of the smallest negative value) +1)
            let abs_minimum_value = food_source_values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .abs();

            normalized_food_source_values = food_source_values
                .iter()
                .map(|x| *x + abs_minimum_value + 1f64)
                .collect();

            let weighted_selection = WeightedIndex::new(&normalized_food_source_values).unwrap();

            //TODO: Make sure this is ALL  correct for ONLOOKER bees!!!!
            //Set onlooker bees based on probability
            for j in 0..self.onlooker_bees {
                //For every single onlooker bee
                //Select a random dimension
                let dimension: usize = random_generator.gen_range(0..self.number_of_dimensions);
                onlooker_chosen_dimension_vec[j] = dimension;

                //Existing position in employed_bees_searches selected using fit_i/Epsilon_SN__j=1 fit_j
                let selected_existing_position_idx =
                    weighted_selection.sample(&mut random_generator);

                onlooker_trial_search_points[j] =
                    employed_bees_searches[selected_existing_position_idx].clone();
                onlooker_mapping_to_employed[j] = selected_existing_position_idx; //make sure we know which belongs to which -- makes parallelism possible

                //Select the index for an existing random food source
                let mut random_solution_index: usize =
                    random_generator.gen_range(0..self.employed_bees);
                while random_solution_index == selected_existing_position_idx {
                    random_solution_index = random_generator.gen_range(0..self.employed_bees)
                }

                // Modify initial positions by xij + phi_ij(xij − xkj)
                let existing_sol =
                    employed_bees_searches[selected_existing_position_idx][dimension];

                //Modify
                let tentative_new = existing_sol
                    + (random_generator.gen_range::<f64, _>(-1.0..1.0000000000000002)
                        * (existing_sol
                            - employed_bees_searches[random_solution_index][dimension]));

                //Check if out of bounds, if they are out of bounds set them to those bounds (Karaboga et al)
                let (lower_bound, upper_bound) = (
                    self.problem_space_bounds[dimension][0],
                    self.problem_space_bounds[dimension][1],
                );

                if tentative_new < lower_bound {
                    onlooker_trial_search_points[j][dimension] = lower_bound;
                } else if tentative_new > upper_bound {
                    onlooker_trial_search_points[j][dimension] = upper_bound;
                } else {
                    onlooker_trial_search_points[j][dimension] = tentative_new;
                }
                ////////////////////////
                //compare new solutions to old ones
                // let new_search =
                //     perform_search!(&onlooker_trial_search_points[selected_existing_position_idx]);
                // if new_search > food_source_values[selected_existing_position_idx] {
                //     //Update to new points if the new source has as higher fitness value
                //     employed_bees_searches[selected_existing_position_idx][dimension] =
                //         onlooker_trial_search_points[selected_existing_position_idx][dimension];

                //     //Update to new fitness value too
                //     food_source_values[selected_existing_position_idx] = new_search;

                //     //If a better value of the food source was found, set the counter to 0 again
                //     attempts_per_food_source[selected_existing_position_idx] = 0;
                // } else {
                //     //Revert onlooker_trial_search_points[selecterd_existing_position_idx][dimension] back to employed_bees_searches[selected_existing_position_idx][dimension]
                //     onlooker_trial_search_points[selected_existing_position_idx][dimension] =
                //         employed_bees_searches[selected_existing_position_idx][dimension];
                // };
            }
            //Run searches in parallel
            let new_search_vec: Vec<f64> = thread_pool.install(|| {
                onlooker_trial_search_points
                    .par_iter()
                    .map(|x| perform_search!(x))
                    .collect()
            });

            for (idx, new_search) in new_search_vec.iter().enumerate() {
                if *new_search > food_source_values[onlooker_mapping_to_employed[idx]] {
                    // Update to new points if the new source has as higher fitness value
                    employed_bees_searches[onlooker_mapping_to_employed[idx]]
                        [onlooker_chosen_dimension_vec[idx]] =
                        onlooker_trial_search_points[idx][onlooker_chosen_dimension_vec[idx]];
                    //Update to new fitness value too
                    food_source_values[onlooker_mapping_to_employed[idx]] = *new_search;
                    //If a better value of the food source was found, set the counter to 0 again
                    attempts_per_food_source[onlooker_mapping_to_employed[idx]] = 0;
                } else {
                    //Revert onlooker_trial_search_points[i][onlooker_chosen_dimension_vec[idx]] back to employed_bees_searches[i][onlooker_chosen_dimension_vec[idx]]
                    //onlooker_trial_search_points[idx][onlooker_chosen_dimension_vec[idx]] = employed_bees_searches[onlooker_mapping_to_employed[idx]][onlooker_chosen_dimension_vec[idx]];
                    //no need to do anything here, as onlooker_trial_search_points[j] will be assigned afresh with each iteration
                };
            }

            update_metadata!(
                food_source_values,
                employed_bees_searches,
                self.onlooker_bees //number of searches performed
            );

            //Send Scout Bee out
            if self.permanent_scout_bees > 0 {
                //So long as there is 1 or more scout bee:
                for k in 0..self.permanent_scout_bees {
                    //Generate initial solutions -> randomly reach out with the scout bee
                    for (idx, each_dimension) in trial_scout_bees_searches[k].iter_mut().enumerate()
                    {
                        //for every single dimension
                        *each_dimension = random_generator
                            .gen_range::<f64, _>(adjusted_bounds[idx][0]..adjusted_bounds[idx][1])
                        //generate random values within problem space bounds.
                    }

                    // //Perform search
                    // let new_search = perform_search!(&trial_scout_bees_searches[k]);
                    // //If replace with new value if search result is better. Started with f64::NEG_INFINITY
                    // if new_search > scout_food_sources_values[k] {
                    //     scout_bees_searches[k] = trial_scout_bees_searches[k].clone(); //replace with new position if return is higher
                    //     scout_food_sources_values[k] = new_search; //replace with new value if return is higher
                    // }
                }

                //Perform search
                let new_search_vec: Vec<f64> = thread_pool.install(|| {
                    trial_scout_bees_searches
                        .par_iter()
                        .map(|x| perform_search!(x))
                        .collect()
                });

                update_metadata!(
                    food_source_values,
                    employed_bees_searches,
                    self.permanent_scout_bees
                );

                //If replace with new value if search result is better. Started with f64::NEG_INFINITY
                for (idx, new_search) in new_search_vec.iter().enumerate() {
                    if *new_search > scout_food_sources_values[idx] {
                        scout_bees_searches[idx] = trial_scout_bees_searches[idx].clone(); //replace with new position if return is higher
                        scout_food_sources_values[idx] = *new_search; //replace with new value if return is higher
                    }
                }
            }

            //Check to see if maximum iterations has been reached for any bee
            for (idx, item) in attempts_per_food_source.iter_mut().enumerate() {
                if *item >= self.local_limit {
                    exceeded_max.push(idx); //contains index of the locations where the food sources are exhausted
                };
            }

            //only executed if max_length is exceeded.
            if exceeded_max.len() > 0 {
                //set food sources to existing permanent scout food sources
                let mut permanent_scout_bees_counter = self.permanent_scout_bees;
                // println!(
                //     "The food sources that have been abandoned are: {:?}",
                //     exceeded_max
                // );
                for i in exceeded_max.iter() {
                    if permanent_scout_bees_counter > 0 {
                        //Get max food source index and value. Not the most efficient way, but has negligible
                        //time costs as the number of scout bees is small.
                        let (max_index, max_value) = scout_food_sources_values
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap();
                        // println!("all values are: {:?}",scout_food_sources_values);
                        // println!("my value is {}",value);
                        // .map(|(index, _)| index)
                        // .unwrap();
                        food_source_values[*i] = *max_value;
                        employed_bees_searches[*i] = scout_bees_searches[max_index].clone();
                        //println!("Scout Bee {} deployed for abandoned food source {}",permanent_scout_bees_counter,*i);

                        //reset scout bee memory for the scout bee that was taken
                        scout_food_sources_values[max_index] = f64::NEG_INFINITY;
                        permanent_scout_bees_counter -= 1;
                        continue;
                    }

                    //Once the scout bee(s) food source(s) have been taken, turn employed bees to temporary_scouts
                    // println!("Converting employed bee to new scout bee number {}", *i);
                    //Generate initial solutions -> randomly reach out with the employee turned scout bee
                    for (idx, each_dimension) in employed_bees_searches[*i].iter_mut().enumerate() {
                        //for every single dimension, generate random values within problem space bounds.
                        *each_dimension = random_generator
                            .gen_range::<f64, _>(adjusted_bounds[idx][0]..adjusted_bounds[idx][1])
                    }

                    temporary_scout_searches.push(employed_bees_searches[*i].clone());
                    //Perform search
                    //println!("Performing search for position {}",*i); //Tested to be OK, does not run if scout bees' solutions have been written
                    //food_source_values[*i] = perform_search!(&employed_bees_searches[*i]);
                }

                //To be run only if the number of food sources that exceeded their max limit is greater than the number of permanent scout bees on duty
                let temporary_scout_food: Vec<f64> = thread_pool.install(|| {
                    temporary_scout_searches
                        .par_iter()
                        .map(|x| perform_search!(x))
                        .collect()
                });
                for idx in self.permanent_scout_bees..exceeded_max.len() {
                    //exceeded_max contains the INDEX values for food_source_values that need to be replaced.
                    food_source_values[exceeded_max[idx]] =
                        temporary_scout_food[idx - self.permanent_scout_bees];  //Deduct by the offset caused by the permanent_scout_bees
                    // println!(
                    //     "Replacing with temporary_scout_food number {}",
                    //     idx - self.permanent_scout_bees
                    // );
                }

                update_metadata!(
                    food_source_values,
                    employed_bees_searches,
                    exceeded_max.len()
                );
            }
            exceeded_max.clear(); //Reset the counters for which dimensions have exceeded maximum here, as we have already dealt with them
            temporary_scout_searches.clear() //reset the vector that holds the temporary scout searches

            //TODO: Reapply the negative sign if you went for minimization instead of maximization
        }

        self.real_time_taken = function_real_time.elapsed();
    }

    //Reinforcement learning ABC from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200738
    pub fn r_abc() {}

    //VS_RABC from my Final Year Thesis project
    pub fn vs_rabc() {}

    pub fn get_results() {}
}

//verify values of each dimension to make sure they are within bounds
pub fn value_verification<T>(upper: T, lower: T) {
    panic!("Value outside expected range.");
}

pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

#[cfg(test)]
mod search_algos {

    use super::*;
    #[test]
    fn test_classic_abc() {
        //#region
        //Set up problem space bounds

        let mut problem_space_bounds = vec![
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
        ];

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_rana = Optimizer::new().minimize().set_thread_pool(1);

        //set custom metadata. Should be fine even if it's commented out.
        optimize_rana.fitness_function_name = String::from("rana");
        optimize_rana.fitness_function_description=String::from("N-dimensional, multimodal. Range = [-512,512] for all dimensions. minimum point for 2D: `[-488.632577, 512]`  Minimum value for 2D=-511.7328819,  Minimum point for 7D: `[-512, -512, -512, -512, -512, -512, -511.995602]` minimum value for 7D=-3070.2475210");
        optimize_rana.known_minimum_value = Some(-3070.2475210);
        optimize_rana.permanent_scout_bees = 2usize;
        optimize_rana.known_minimum_point = Some(vec![
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -511.995602,
        ]);

        //#endregion
        //Run optimization function here.
        optimize_rana.classic_abc(
            &problem_space_bounds,
            100u64,               //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_rana.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        println!("\n\nObject={:#?}\n\n", optimize_rana);
    }

    #[test]
    fn test_classic_abc_all(){
        //#region
        //Set up problem space bounds

        let mut problem_space_bounds = vec![
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
        ];

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_multiple = Optimizer::new().minimize().set_thread_pool(38);

        //set custom metadata. Should be fine even if it's commented out.
        optimize_multiple.fitness_function_name = String::from("rana");
        optimize_multiple.fitness_function_description=String::from("N-dimensional, multimodal. Range = [-512,512] for all dimensions. minimum point for 2D: `[-488.632577, 512]`  Minimum value for 2D=-511.7328819,  Minimum point for 7D: `[-512, -512, -512, -512, -512, -512, -511.995602]` minimum value for 7D=-3070.2475210");
        optimize_multiple.known_minimum_value = Some(-3070.2475210);
        optimize_multiple.permanent_scout_bees = 2usize;
        optimize_multiple.known_minimum_point = Some(vec![
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -512.0,
            -511.995602,
        ]);

        //#endregion
        //Run optimization function here.
        optimize_multiple.classic_abc(
            &problem_space_bounds,
            3100u64,               //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_multiple.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        println!("\n\nObject={:#?}\n\n", optimize_multiple);
    }


    

    #[test]
    fn run_code() {
        assert_eq!(
            Optimizer::set_bounds(&vec![[-1.0, 1.0]], "[]"),
            vec![[-1.0, 1.0000000000000002]]
        );
        assert_eq!(1.0, 1.0000000000000001); //Shows that there is no difference between 1.0 and 1.0000000000000001
        assert_ne!(1.0, 1.0000000000000002); //shows that there is a difference here. 1.0000000000000002 is truly the smallest number.

        let mut a: f64 = 3.0;
        fn change_value(input: &mut f64) {
            *input = *input + 1f64;
        }

        change_value(&mut a);

        println!("The value of a is {}", a);
    }

    #[test]
    fn test_classic_abc_metadata() {
        //Set up problem space bounds
        let problem_space_bounds = vec![
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
        ];

        //create new instance of struct
        let mut optimize_rana = Optimizer::new();

        //Should work without optional metadata
        optimize_rana.classic_abc(
            &problem_space_bounds,
            10u64,                 //Max iterations
            benchmark_algos::rana, //name of fitness function
        );
        println!("\n\nObject={:#?}\n\n", optimize_rana);
    }

    #[test]
    fn test_classic_abc_minimize_flag() {
        //create new instance of struct
        let optimize_rana = Optimizer::new().minimize();
        assert_eq!(optimize_rana.maximize, false); //make sure minimization flag was really applied!
    }

    #[test]
    fn test_set_bounds() {
        //#region
        //Set up problem space bounds

        let problem_space_bounds = vec![[-512.0, 512.0], [-512.0, 512.0], [-512.0, 512.0]];

        /*
        All ranges used for RNG genneration will be the default Rust '..'
        which are lower-bound inclusive but upper-bound exclusive, aka [lb,ub).
        For example, 'for i in 0..5 {println!("{}",i)}' Yields 0 1 2 3 4.
        Thus, for your range:
        If lower-bound inclusive & upper-bound exclusive, [lb,ub), no action needs to be taken. (Optimzier::set_bounds(input_vec,"[)_or_anything_else"))
        If lower-bound inclusive & upper-bound inclusive, [lb, ub], add smallest unit to upper bound (use Optimzier::set_bounds(input_vec,"[]"))
        If lower-bound exclusive & upper-bound inclusive, (lb, ub], add smallest unit to lower AND upper bounds (use Optimzier::set_bounds(input_vec,"(]"))
        If lower-bound exclusive & upper-bound exclusive, (lb, ub), add smallest unit to lower bound (use Optimzier::set_bounds(input_vec,"()"))
        */

        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, "[)"),
            problem_space_bounds
        ); //Should have no change
        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, ""),
            vec![
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001]
            ]
        ); //Default is "[]", lower and upper-bound inclusive
        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, "RandomGibberish"),
            vec![
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001]
            ]
        ); //Default is "[]", lower and upper-bound inclusive
        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, "[]"),
            vec![
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001],
                [-512.0, 512.0000000000001]
            ]
        ); //Add the smallest amount possible to upper bound
        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, "(]"),
            vec![
                [-511.99999999999994, 512.0000000000001],
                [-511.99999999999994, 512.0000000000001],
                [-511.99999999999994, 512.0000000000001]
            ]
        ); //Add the smallest amount possible to upper bound and lower bound
        assert_eq!(
            Optimizer::set_bounds(&problem_space_bounds, "()"),
            vec![
                [-511.99999999999994, 512.0],
                [-511.99999999999994, 512.0],
                [-511.99999999999994, 512.0]
            ]
        ); //Add the smallest amount possible to lower bound only
    }

    #[test]
    #[should_panic]
    fn test_classic_abc_invalid_options() {
        //Set up problem space bounds
        let problem_space_bounds = vec![
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
        ];

        //create new instance of struct
        let mut optimize_rana = Optimizer::new();

        optimize_rana.employed_bees = 1;
        //Should panic
        optimize_rana.classic_abc(
            &problem_space_bounds,
            100u64,                //Max iterations
            benchmark_algos::rana, //name of fitness function
        )
    }

    #[test]
    #[should_panic]
    fn test_ignore_nan() {
        //Set up problem space bounds
        let problem_space_bounds = vec![
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
            [-512.0, 512.0],
        ];

        //create new instance of struct
        let mut optimize_rana = Optimizer::new();

        //Should panic
        optimize_rana.classic_abc(
            &problem_space_bounds,
            100u64,                             //Max iterations
            benchmark_algos::test_nan_function, //name of fitness function
        )
    }
}

#[cfg(test)]
mod test_functions {
    use super::*;
    use benchmark_algos::*;
    use std::f64::consts::PI;
    // //Note: boundary values are -5.12 and 5.12 //Rastrigin test function.
    // fn rastrigin(input_vector:&Vec<f64>) -> f64
    // {
    // (10.0 * input_vector.len() as f64)  + input_vector.iter().map(|x|{(x.powf(2.0))-10.0*((2.0*PI*x).cos())}).sum::<f64>()
    // }

    //Use to print and return value. Remember: object must have trait bound std::fmt::Debug
    // fn print_and_return<T:std::fmt::Debug>(input:T) -> T
    // {println!("Value = {:?}",input);
    // input
    // }

    //Evaluted in the range [-32.768, 32.768] for ALL dimensions
    //One global minimum of f(x)=0 at x=(0,...0)
    //Generalized to N-dimensions
    #[test]
    fn test_ackley() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let my_vector3 = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let my_vector4 = vec![1.3, 1.4, 1.5];

        assert_eq!(ackley(&my_vector), 0.0);
        assert_eq!(ackley(&my_vector2), 0.0);
        assert_eq!(ackley(&my_vector3), 11.336903284758636); //Exactly the same value as the Python test bed!
        assert_eq!(ackley(&my_vector4), 7.116187998576905);
    }

    #[test]
    fn test_alpine_1() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![2.0];
        let my_vector3 = vec![1.0, 7.0, 7.0, 6.0];

        assert_eq!(alpine_1(&my_vector), 0.0);
        assert_eq!(alpine_1(&my_vector2), 2.0185948536513636);
        assert_eq!(alpine_1(&my_vector3), 12.615776356064499);
    }

    #[test]
    fn test_alpine_2() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![2.0];
        let my_vector3 = vec![1.0, 7.0, 7.0, 6.0];

        assert_eq!(alpine_2(&my_vector), 0.0);
        assert_eq!(alpine_2(&my_vector2), -1.2859407532478362);
        assert_eq!(alpine_2(&my_vector3), 1.7401084806578164);
    }

    #[test]
    fn test_alpine_2_max() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![2.0];
        let my_vector3 = vec![1.0, 7.0, 7.0, 6.0];

        assert_eq!(alpine_2_max(&my_vector), 0.0);
        assert_eq!(alpine_2_max(&my_vector2), 1.2859407532478362);
        assert_eq!(alpine_2_max(&my_vector3), -1.7401084806578164);
    }

    #[test]
    fn test_beale() {
        let my_vector = vec![3.0, 0.5];
        assert_eq!(beale(&my_vector), 0.0);
    }

    #[test]
    fn test_booth() {
        let my_vector = vec![1.0, 3.0];
        assert_eq!(booth(&my_vector), 0.0);
    }

    #[test]
    fn test_bukin_6() {
        let my_vector = vec![-10.0, 1.0];

        assert_eq!(bukin_6(&my_vector), 0.0);
    }

    #[test]
    fn test_cross_in_tray() {
        let my_vector = vec![-1.34941, -1.34941];
        let my_vector2 = vec![-1.34941, 1.34941];
        let my_vector3 = vec![1.34941, -1.34941];
        let my_vector4 = vec![1.34941, 1.34941];

        assert_eq!(
            (cross_in_tray(&my_vector) * 1.0e5).round() / 1.0e5,
            -2.06261
        );
        assert_eq!(
            (cross_in_tray(&my_vector2) * 1.0e5).round() / 1.0e5,
            -2.06261
        );
        assert_eq!(
            (cross_in_tray(&my_vector3) * 1.0e5).round() / 1.0e5,
            -2.06261
        );
        assert_eq!(
            (cross_in_tray(&my_vector4) * 1.0e5).round() / 1.0e5,
            -2.06261
        );
    }
    #[test]
    fn test_cross_leg_table() {
        let myvector = vec![0.0, 9.9];
        let myvector2 = vec![8.9, 0.0];
        let myvector3 = vec![0.0, 0.0];
        let myvector4 = vec![9.0, 9.0];
        let myvector5 = vec![-8.0, -8.0];

        assert_eq!(cross_leg_table(&myvector), -1.0);
        assert_eq!(cross_leg_table(&myvector2), -1.0);
        assert_eq!(cross_leg_table(&myvector3), -1.0);
        assert_eq!(cross_leg_table(&myvector4), -8.12833970460678e-5); //Values taken from my python sample
        assert_eq!(cross_leg_table(&myvector5), -6.522069592867228e-5); //Values taken from my python sample
    }

    #[test]
    fn test_drop_wave() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![2.0, -3.0];
        let my_vector3 = vec![-3.0, 4.0];
        let my_vector4 = vec![1.0, 1.0];
        let my_vector5 = vec![-2.0, -2.0];

        assert_eq!(drop_wave(&my_vector), -1.0);
        assert_eq!(drop_wave(&my_vector2), -0.20642894546663867);
        assert_eq!(drop_wave(&my_vector3), -0.003281863419644392);
        assert_eq!(drop_wave(&my_vector4), -0.23221968746199587);
        assert_eq!(drop_wave(&my_vector5), -0.03067190814418271);
    }

    #[test]
    fn test_easom() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![PI, PI];
        let my_vector3 = vec![-17.0, 76.0];

        assert_eq!(easom(&my_vector), -2.675287991074243e-9); //Directly from the Python test bench.
        assert_eq!(easom(&my_vector2), -1.0);
        assert_eq!(easom(&my_vector3), 0.0);
    }

    #[test]
    fn test_egg_holder() {
        let myvec = vec![
            480.852413, 431.374221, 444.908694, 457.547223, 471.962527, 427.497291, 442.091345,
            455.119420, 469.429312, 424.940608,
        ];
        let myvec2 = vec![
            483.116792, 438.587598, 453.927920, 470.278609, 425.874994, 441.797326, 455.987180,
        ];

        assert_eq!(
            ((egg_holder(&myvec) * (1.0e7)).round()) / (1.0e7),
            -8291.2400675
        ); //verified to 7 decimal places from  https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app1
        assert_eq!(
            ((egg_holder(&myvec2) * (1.0e7)).round()) / (1.0e7),
            -5548.9775483
        );
    }

    #[test]
    fn test_goldstein_price() {
        let myvector = vec![0.0, -1.0];
        assert_eq!(goldstein_price(&myvector), 3.0);
    }

    #[test]
    fn test_griewank() {
        let myvector = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let myvector2 = vec![1.0, 7.0, 7.0, -6.0];

        assert_eq!(griewank(&myvector), 0.0);
        assert_eq!(griewank(&myvector2), 0.9555543043434543);
    }

    #[test]
    fn test_himmelblau() {
        let myvector = vec![3.0, 2.0];
        let myvector2 = vec![-2.805118, 3.131312];
        let myvector3 = vec![-3.779310, -3.283186];
        let myvector4 = vec![3.584428, -1.848126];

        assert_eq!(himmelblau(&myvector), 0.0);
        assert_eq!(((himmelblau(&myvector2) * (1.0e7)).round()) / (1.0e7), 0.0); //down to 7 decimal places
        assert_eq!(((himmelblau(&myvector3) * (1.0e7)).round()) / (1.0e7), 0.0);
        assert_eq!(((himmelblau(&myvector4) * (1.0e7)).round()) / (1.0e7), 0.0);
    }

    #[test]
    fn test_keane() {
        let myvec = vec![3.065318, 1.531047, 0.405617, 0.393987];
        let myvec2 = vec![3.042963, 1.482875, 0.166211];

        assert_eq!((keane(&myvec) * 1.0e7).round() / 1.0e7, -0.6222807); //had to modify this value a bit, the other one seems OK though
        assert_eq!((keane(&myvec2) * 1.0e7).round() / 1.0e7, -0.5157855);
    }

    #[test]
    fn test_levy13() {
        let myvec = vec![1.0, 1.0];

        assert_eq!((keane(&myvec) * 1.0e10).round() / 1.0e10, 0.0); //rounded to 10dp, closest we could get to 0.0
    }

    #[test]
    fn test_matyas() {
        let my_vector = vec![0.0, 0.0];
        assert_eq!(matyas(&my_vector), 0.0);
    }

    #[test]
    fn test_mccormick() {
        let my_vector = vec![-0.54719, -1.54719];
        assert_eq!((mccormick(&my_vector) * 1.0e4).round() / 1.0e4, -1.9132);
    }

    #[test]
    fn test_michalewicz() {
        let my_vector = vec![
            2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087,
            1.655717, 1.570796,
        ];
        assert_eq!(
            (michalewicz(&my_vector) * 1.0e7).round() / 1.0e7,
            -9.6601517
        );
    }

    #[test]
    //Dubious, might want to research further.
    fn test_penalized1() {
        let my_vector = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        assert_eq!((penalized1(&my_vector) * 10e10).round() / 10e10, 0.0);
    }

    #[test]
    //Dubious, might want to leave this one out.
    fn test_penalized2() {
        let my_vector = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!((penalized2(&my_vector) * 10e10).round() / 10e10, 0.0);
    }

    #[test]
    fn test_rana() {
        let my_vector = vec![-512.0, -512.0, -512.0, -512.0, -512.0, -512.0, -511.995602];
        let my_vector2 = vec![-488.632577, 512.0];
        let my_vector3 = vec![-512.0, -512.0, -512.0, -512.0, -512.0, -512.0, -512.0];

        assert_eq!((rana(&my_vector) * 1.0e7).round() / 1.0e7, -3070.2475210);
        assert_eq!((rana(&my_vector2) * 1.0e7).round() / 1.0e7, -511.7328819);
        assert_eq!(rana(&my_vector3), -3070.2463657915314);
    }

    #[test]
    fn test_rastrigin() {
        let my_vector = vec![0.0, 0.0];
        let my_vector2 = vec![
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
            -4.52299366,
        ];
        let my_vector3 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        assert_eq!(rastrigin(&my_vector), 0.0);
        assert_eq!(rastrigin(&my_vector2), 363.1796117445507);
        assert_eq!(rastrigin(&my_vector3), 0.0);
    }

    #[test]
    fn test_rosenbrock() {
        let my_vector = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let my_vector2 = vec![-2.0, 2.0];

        assert_eq!(rosenbrock(&my_vector), 0.0);
        assert_eq!(rosenbrock(&my_vector2), 409.0);
    }

    #[test]
    fn test_schaffer6() {
        let my_vector = vec![0.0, 0.0];
        assert_eq!(schaffer6(&my_vector), 0.0);
    }

    #[test]
    fn test_schwefel() {
        let my_vector = vec![420.9687, 420.9687];
        let my_vector2 = vec![420.9687, 420.9687, 420.9687];
        let my_vector3 = vec![420.9687, 420.9687, 420.9687, 420.9687, 420.9687];
        assert_eq!((schwefel(&my_vector) * 1.0e3).round() / 1.0e3, 0.0);
        assert_eq!((schwefel(&my_vector2) * 1.0e3).round() / 1.0e3, 0.0);
        assert_eq!((schwefel(&my_vector3) * 1.0e3).round() / 1.0e3, 0.0);
    }

    #[test]
    fn test_schwefel12() {
        let my_vector = vec![0.0, 0.0, 0.0, 0.0];
        let my_vector2 = vec![1.0, 7.0, 7.0, 6.0];

        assert_eq!(schwefel12(&my_vector), 0.0);
        assert_eq!(schwefel12(&my_vector2), 731.0);
    }

    #[test]
    fn test_schwefel226() {
        let my_vector = vec![420.968746, 420.968746, 420.968746, 420.968746, 420.968746];
        let my_vector2 = vec![420.9687; 30];

        assert_eq!(schwefel226(&my_vector), -2094.9144363621685);
        assert_eq!(schwefel226(&my_vector2), -12569.48661816488);
    }

    #[test]
    fn test_shekel4_5() {
        //let my_vector=vec![4.00004, 4.00013, 4.00004, 4.00013];
        let my_vector = vec![4.00004, 4.00013, 4.00004, 4.00013];

        assert_eq!((shekel4_5(&my_vector) * 1.0e4).round() / 1.0e4, -10.1532); //round to 4 decimal places per our source in MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
    }

    #[test]
    fn test_shekel4_7() {
        let my_vector = vec![4.00057, 4.00069, 3.99949, 3.99961];

        assert_eq!((shekel4_7(&my_vector) * 1.0e3).round() / 1.0e3, -10.403); //round to 3 decimal places per our source in MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
    }

    #[test]
    fn test_shekel4_10() {
        let my_vector = vec![4.00075, 4.00059, 3.99966, 3.99951];

        assert_eq!((shekel4_10(&my_vector) * 1.0e4).round() / 1.0e4, -10.5364); //round to 4 decimal places per our source in MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
    }

    #[test]
    fn test_shekel10() {
        let my_vector = vec![
            8.024968, 9.151929, 5.113991, 7.620959, 4.564020, 4.711005, 2.996030, 6.125993,
            0.734057, 4.981999,
        ];

        assert_eq!((shekel10(&my_vector) * 1e7).round() / 1e7, -10.2078768);
    }

    #[test]
    fn test_sine_envelope() {
        let my_vector = vec![
            -1.517016, -1.403507, 1.517016, -1.403507, -1.517015, 1.403507,
        ];
        let my_vector2 = vec![-0.086537, 2.064868];

        assert_eq!((sine_envelope(&my_vector) * 1e7).round() / 1e7, -7.4574764);
        assert_eq!((sine_envelope(&my_vector2) * 1e7).round() / 1e7, -1.4914953);
    }

    #[test]
    fn test_sphere() {
        let my_vector = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let my_vector2 = vec![0.0, 0.0];

        assert_eq!(sphere(&my_vector), 0.0);
        assert_eq!(sphere(&my_vector2), 0.0);
    }

    #[test]
    fn test_step2() {
        let my_vector = vec![0.4, 0.4, 0.4, 0.4, 0.4];
        let my_vector2 = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let my_vector3 = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        assert_eq!(step2(&my_vector), 0.0);
        assert_eq!(step2(&my_vector2), 0.0);
        assert_eq!(step2(&my_vector3), 5.0);
    }

    #[test]
    fn test_styblinski_tang() {
        let my_vector = vec![-2.903534, -2.903534, -2.903534, -2.903534, -2.903534];
        assert_eq!(styblinski_tang(&my_vector), -195.830828518857); //value obtained from another python code I trust
    }

    #[test]
    fn test_sum_squares() {
        let my_vector = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(sum_squares(&my_vector), 0.0); //value obtained from another python code I trust
    }

    #[test]
    fn test_three_hump_camel() {
        let my_vector = vec![0.0, 0.0];
        assert_eq!(three_hump_camel(&my_vector), 0.0);
    }

    #[test]
    fn test_zakharov() {
        let my_vector = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let my_vector2 = vec![1.0, 7.0, 7.0, 6.0];

        assert_eq!(zakharov(&my_vector), 0.0);
        assert_eq!(zakharov(&my_vector2), 811035.0);
    }
}
