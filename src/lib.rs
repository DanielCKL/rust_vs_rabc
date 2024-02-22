//!This crate implements a modified version of ABC (VS-RABC) based on my university final year project, which was originally implemented in Python. It demonstrates faster convergence for benchmarks like the Rosenbrock and Rastrigin benchmark functions, and is on par with the other functions for other more elaborate benchmarks. 
//!
//!The original method for Karaboga's ABC (<https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems>) algorithm is also implemented in Rust here for comparison's sake.
//!
//!Also implemented is Reinforcement-Learning ABC, which is based on Fairee et al's paper here: <https://doi.org/10.1371/journal.pone.0200738>
//! 
//!To use this crate, you can start by adding it to your : 
//!``
//!``` 
//!
//!```
//~
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]

mod benchmark_algos;
use plotters::prelude::*;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use rayon::prelude::*;
use std::env::current_dir;
use std::fs;

//use std::thread;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

#[derive(Default, Debug)]
pub struct Optimizer {
    //COMPULSORY Optimizer parameters
    ///Maximum number of generations that this algorithm will be run for
    pub max_generations: u64,                
    ///Must be bounded. Assumed stored as inclusive of upper bound (ub) and lower bound (lb): [ub,lb]       
    pub problem_space_bounds: Vec<[f64; 2]>, 


    //optional parameters
    pub employed_bees: usize,        //The number of employed bees (and thus the number of food sources/possible solutions that will constantly be searched/exploited).
    pub onlooker_bees: usize,        //The number of onlooker bees (for every employed bee, there should be 1 onlooker bee)
    pub permanent_scout_bees: usize, //If None (default), will be set by algorithm itself
    maximize: bool,                  //if true, maximize, if false, minimize
    pub local_limit: usize, //Limit for how many times a food source can be exploited before being abandoned.
    thread_pool_size: usize, //size of the thread pool
    parallel_mode: bool,    //Enable or disable parallel mode. Defaults to True.

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

    //Performance data
    pub searches_made_history: Vec<usize>, //Track record/history of many searches have been performed. Default of 0
    pub total_searches_made: usize,        //Final number of searches made

    pub min_max_value_history: Vec<f64>,
    pub min_max_value: f64, //the minimum/maximum reward value found within problem space (single value). Default of 0.0

    pub min_max_point_history: Vec<Vec<f64>>,
    pub min_max_point: Vec<f64>, //vector solution that will return min_max_value.

    pub iter_to_min_max: usize, //Number of iterations that were taken to reach the minimum value.

    pub real_time_taken: Duration, //Real time taken. Default of 0.0

    //parameters unique to vs_rabc
    pub traversal_searches: usize, //how many traversal searches made per random vector generated
    pub search_distance_factor: f64, //Search distance calculated as a percentage of the largest problem space bound. Default of 10%
    search_distance: f64,            //actual search distance stored.
    search_distance_delta: f64,      //how much the search distance is allowed to change by
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

    //Return a curve where x=0 -> 0, x=1 -> 1, x=0.5 -> 0.5
    fn sigmoid(self: &Self, x: f64) -> f64 {
        //let exponential_term=(consts::E*((5.0*x) - 2.0)).exp();  //seems best for Rana and rosenbrock
        /////////////////////////////////////////////////////////////////////////
        //0.45 best so far. Previously 15.0 for the leftmost term.
        let exponential_term = (8.0 * (0.43 - x)).exp(); // used to be 8 * 0.35...
                                                         //exponential_term/(1.0+exponential_term);
                                                         //1.07 was best so far
        let results = 1.09 / (1.0 + exponential_term);

        if results > x.powf(0.43) {
            results
        } else {
            x.powf(0.43)
        }
        /////////////////////////////////////////////////////////////////////////
        //1.0
        //What if we try with a capped exponential value?
        // let mut results= 1.05*x.powf(0.45);
        // if results > (x) {}
        // else {results=x;}
        // results
    }

    ///creates a new instance of the optimizer that can be used for ABC, Reinforcement-ABC, and VS-Reinforcement-ABC. 
    ///Also sets default parameter values.
    pub fn new() -> Self {
        Self {
            employed_bees: 50usize, //Default values for employed, onlooker, and scout bees as recommended in the source paper.
            onlooker_bees: 50usize,
            parallel_mode: true, //default to parallel mode.
            permanent_scout_bees: 1usize,
            local_limit: 100usize, //550usize seems to be the most optimal
            maximize: true,        //default to finding maximum value in input problem space
            min_max_value: f64::NEG_INFINITY,
            thread_pool_size: available_parallelism().unwrap().get(),
            traversal_searches: 4, //how many traversal searches made per random vector generated
            search_distance_factor: 0.1,
            search_distance_delta: 0.1, //how much the search distance is allowed to change by
            ..Default::default()
        }
    }

    // resets the other metadata and allows you to carry on with the same settings
    //for a fresh run, just create a new() instance in a new scope, or drop the old instance
    pub fn clear(self: &mut Self) {
        //Performance data to be written AFTER the algorithm has finished running.
        //self.searches_made_history.push(0); //how many iterations have been run. Default of 0
        if self.maximize == false {
            self.min_max_value = f64::INFINITY;
        } else {
            self.min_max_value = f64::NEG_INFINITY;
        } //the minimum/maximum reward value found within problem space (single value)}

        self.min_max_point.clear(); //vector solution that will return min_max_value.
        self.searches_made_history.clear();
        self.min_max_value_history.clear();
        self.min_max_point_history.clear();
        self.iter_to_min_max = 0;
        self.total_searches_made = 0;
        self.real_time_taken = Duration::new(0, 0);
    }

    pub fn set_limit(mut self: Self, limit: usize) -> Self {
        self.local_limit = limit;
        self
    }

    //Builder-pattern method to switch from maximization to minimization mode.
    pub fn minimize(mut self: Self) -> Self {
        self.maximize = false;
        self.min_max_value = f64::INFINITY;
        self
    }

    pub fn not_parallel(mut self: Self) -> Self {
        self.parallel_mode = false;
        self
    }

    //should strongly recommend user allow the system to decide for them
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

    pub fn set_traversal_searches(mut self: Self, mut traversal_searches: usize) -> Self {
        if traversal_searches < 1 {
            println!("Warning: Zero value are NOT allowed. Setting traversal_searches to the default of 4");
            traversal_searches = 4;
        }
        self.traversal_searches = traversal_searches;
        self
    }

    pub fn set_search_distance_delta(mut self: Self, mut search_distance_delta: f64) -> Self {
        if search_distance_delta < 0.0 {
            println!("Warning: Negative values are NOT allowed. Setting search_distance_delta to the default of 0.1");
            search_distance_delta = 0.1;
        }
        self.search_distance_delta = search_distance_delta;
        self
    }

    //Set search distance factor (as a fraction of the largest search space range). The default value is 0.1
    pub fn set_search_distance_factor(mut self: Self, mut search_distance_factor: f64) -> Self {
        if search_distance_factor < 0.0 {
            println!("Warning: Negative values are NOT allowed. Setting search_distance_factor to the default of 0.1");
            search_distance_factor = 0.1;
        }
        if search_distance_factor > 1.0 {
            println!("Warning: Setting this above 1.0 may constantly result in attempts to check the very edge of the problem space. The recommended range is 0<search_distance_factor<1.0");
        }
        self.search_distance_factor = search_distance_factor;
        self
    }

    //Rank a reference to f64 vector and return a vector with the indices of the vector sorted in DESCENDING order.
    fn rank_vecf64_desc(input_vec: &Vec<f64>) -> Vec<usize> {
        let mut input_copy = input_vec.clone();

        let mut output_vec = vec![0usize; input_copy.len()];
        let mut offset_vec = vec![0usize; input_copy.len()];

        let mut max_index:usize;

        for item in output_vec.iter_mut() {
        (max_index, _) = input_copy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            *item = max_index + offset_vec[max_index];

            //update offset
            for i in max_index..input_copy.len() {
                offset_vec[i] += 1;
            }

            //remove maximum from input_copy and offset_vec to reduce the search
            input_copy.remove(max_index);
            offset_vec.remove(max_index);
        }

        // println!("input_vec= {:?}", input_vec);
        // println!("output_vec= {:?}", output_vec);

        output_vec
    }

    // vec1 + vec2
    fn add_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a + b).collect();
        // println!("Vec1 and Vec 2: {:?}, {:?}",vec1,vec2);
        // println!("Results of adding elementwise: {:?}", result);
        result
    }

    // // vec1 - vec2
    fn deduct_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a - b).collect();
        //println!("{:?}", result);
        result
    }

    // // vec1 * vec2
    fn multiply_elementwise(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let result: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).collect();
        //println!("{:?}", result);
        result
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

    fn update_metadata(
        self: &mut Self,              //Updates Struct fields
        vec_fitness_value: &Vec<f64>, //vector of fitness values
        input_vector: &Vec<Vec<f64>>, //vector of coordinates matching vec_fitness_value
        minmax_factor: f64,           //minmax factor value
        searches_performed: usize,    //key in searches performed here
    ) -> Option<String> {
        //perform all other actions here
        //Get maximum value and coordinates of fitness value
        let (max_index, max_value) = vec_fitness_value
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        //Note: max_value was previously already multiplied by the minmax_factor

        //Update maximumn searches and vector holding number of searches
        match self.searches_made_history.last() {
            Some(&v) => {
                self.searches_made_history.push(v + searches_performed);
                self.total_searches_made = v + searches_performed;
            }
            None => {
                self.searches_made_history.push(searches_performed);
                self.total_searches_made = searches_performed;
            } //If the vector is blank, the initial number of searches made was 0
        };

        //update metadata for max value and the vector that returns that value
        match (*max_value) > (minmax_factor * self.min_max_value) {
            true => {
                self.min_max_value = minmax_factor * max_value;
                self.min_max_point = input_vector[max_index].clone();
                self.min_max_value_history.push(self.min_max_value);
                self.min_max_point_history.push(self.min_max_point.clone());
                Some(format!(
                    "Max value updated to {} at {:?}",
                    self.min_max_value, self.min_max_point
                ))
            }
            false => {
                //Write to history with te same value
                self.min_max_value_history.push(self.min_max_value);
                self.min_max_point_history.push(self.min_max_point.clone());
                Some(format!("No changes made."))
            }
        }
    }

    //Perform post-processing of metadata after the algorithm has finished running.
    fn post_process_metadata(self: &mut Self) -> Option<String> {
        let first_minmax_idx = self
            .min_max_value_history
            .iter()
            .position(|x| *x == self.min_max_value)
            .unwrap();

        //Update the number of iterations taken to min/max the value of self.min_max_value
        self.iter_to_min_max = self.searches_made_history[first_minmax_idx];

        match self.maximize {
            true => println!(
                "Number of iterations that it took to get maximum value of {} is {}",
                self.min_max_value, self.iter_to_min_max
            ),
            false => println!(
                "Number of iterations that it took to get minimum value of {} is {}",
                self.min_max_value, self.iter_to_min_max
            ),
        };

        Some(String::from("Test"))
    }

    //update employed bees and other tracking variables in place without returning anything else.
    fn update_employed_bees(
        self: &mut Self,
        random_generator: &mut rand::rngs::ThreadRng,
        attempts_per_food_source: &mut Vec<usize>,
        trial_search_points: &mut Vec<Vec<f64>>,
        chosen_dimension_vec: &mut Vec<usize>,
        employed_bees_searches: &mut Vec<Vec<f64>>,
    ) {
        //Karaboga: at each cycle at most one scout ... number of employed and onlooker bees were equal.
        //Number of employed and onlooker bees will remain the same throughout the algorithm.
        for i in 0..self.employed_bees {
            //Karaboga: at each cycle at most one scout ... number of employed and onlooker bees were equal.
            //Number of employed and onlooker bees will remain the same throughout the algorithm.

            //First increment the attempt made for the food source
            attempts_per_food_source[i] += 1;

            //For every single employed bee
            //Select a random chosen_dimension
            let chosen_dimension: usize = random_generator.gen_range(0..self.number_of_dimensions);
            chosen_dimension_vec[i] = chosen_dimension;

            //Select the index for an existing random food source
            let mut random_solution_index: usize =
                random_generator.gen_range(0..self.employed_bees);
            while random_solution_index == i {
                random_solution_index = random_generator.gen_range(0..self.employed_bees)
            }

            // Modify initial positions by xij + phi_ij(xij − xkj)
            let existing_sol = employed_bees_searches[i][chosen_dimension];

            //Modify tentative_new
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
    }

    //update employed bees and other tracking variables in place without returning anything else.
    fn reinforcement_update_employed_bees(
        self: &mut Self,
        random_generator: &mut rand::rngs::ThreadRng,
        attempts_per_food_source: &mut Vec<usize>,
        trial_search_points: &mut Vec<Vec<f64>>,
        chosen_dimension_vec: &mut Vec<usize>,
        employed_bees_searches: &mut Vec<Vec<f64>>,
        common_reinforcement_vector: &Vec<f64>,
    ) {
        //Karaboga: at each cycle at most one scout ... number of employed and onlooker bees were equal.
        //Number of employed and onlooker bees will remain the same throughout the algorithm.
        let d = self.number_of_dimensions as f64;
        for i in 0..self.employed_bees {
            //Karaboga: at each cycle at most one scout ... number of employed and onlooker bees were equal.
            //Number of employed and onlooker bees will remain the same throughout the algorithm.

            //First increment the attempt made for the food source
            attempts_per_food_source[i] += 1;

            //For every single employed bee
            //Select a random chosen_dimension
            let chosen_dimension: usize = random_generator.gen_range(0..self.number_of_dimensions);
            chosen_dimension_vec[i] = chosen_dimension;

            //Select the index for an existing random food source
            let mut random_solution_index: usize =
                random_generator.gen_range(0..self.employed_bees);
            while random_solution_index == i {
                random_solution_index = random_generator.gen_range(0..self.employed_bees)
            }

            // Modify initial positions by xij + phi_ij(xij − xkj)
            let existing_sol = employed_bees_searches[i][chosen_dimension];

            //Modify tentative_new
            let tentative_new = existing_sol
                + (random_generator.gen_range::<f64, _>(-1.0..1.0000000000000002) //1.0000000000000002 is the smallest next value after 1f64
                    //* (common_reinforcement_vector[chosen_dimension] * d)
                    * self.sigmoid(common_reinforcement_vector[chosen_dimension] * d)
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
    }

    fn update_onlooker_bees(
        self: &mut Self,
        food_source_values: &mut Vec<f64>,
        random_generator: &mut rand::rngs::ThreadRng, //random generator object
        onlooker_chosen_dimension_vec: &mut Vec<usize>,
        onlooker_trial_search_points: &mut Vec<Vec<f64>>,
        employed_bees_searches: &mut Vec<Vec<f64>>,
        onlooker_mapping_to_employed: &mut Vec<usize>,
    ) {
        //calculate probability for onlooker bees
        //Normalize values (if there are negative values, add the (modulus of the smallest negative value) +1)
        let abs_minimum_value = food_source_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs();

        // let mut normalized_food_source_values = vec![0.0f64; self.employed_bees];
        let normalized_food_source_values: Vec<f64> = food_source_values
            .iter()
            .map(|x| *x + abs_minimum_value + 1f64)
            .collect();

        let weighted_selection = WeightedIndex::new(&(*normalized_food_source_values)).unwrap();
        //Set onlooker bees based on probability
        for j in 0..self.onlooker_bees {
            //For every single onlooker bee
            //Select a random dimension
            let dimension: usize = random_generator.gen_range(0..self.number_of_dimensions);
            onlooker_chosen_dimension_vec[j] = dimension;

            //Existing position in employed_bees_searches selected using fit_i/Epsilon_SN__j=1 fit_j
            let selected_existing_position_idx = weighted_selection.sample(random_generator);

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
            let existing_sol = employed_bees_searches[selected_existing_position_idx][dimension];

            //Modify
            let tentative_new = existing_sol
                + (random_generator.gen_range::<f64, _>(-1.0..1.0000000000000002)
                    * (existing_sol - employed_bees_searches[random_solution_index][dimension]));

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
        }
    }

    fn reinforcement_update_onlooker_bees(
        self: &mut Self,
        food_source_values: &mut Vec<f64>,
        random_generator: &mut rand::rngs::ThreadRng, //random generator object
        onlooker_chosen_dimension_vec: &mut Vec<usize>,
        onlooker_trial_search_points: &mut Vec<Vec<f64>>,
        employed_bees_searches: &mut Vec<Vec<f64>>,
        onlooker_mapping_to_employed: &mut Vec<usize>,
        common_reinforcement_vector: &Vec<f64>,
    ) {
        //calculate probability for onlooker bees
        //Normalize values (if there are negative values, add the (modulus of the smallest negative value) +1)
        let abs_minimum_value = food_source_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs();

        // let mut normalized_food_source_values = vec![0.0f64; self.employed_bees];
        let normalized_food_source_values: Vec<f64> = food_source_values
            .iter()
            .map(|x| *x + abs_minimum_value + 1f64)
            .collect();

        let weighted_selection = WeightedIndex::new(&(*normalized_food_source_values)).unwrap();

        let d = self.number_of_dimensions as f64;
        //Set onlooker bees based on probability
        for j in 0..self.onlooker_bees {
            //For every single onlooker bee
            //Select a random dimension
            let dimension: usize = random_generator.gen_range(0..self.number_of_dimensions);
            onlooker_chosen_dimension_vec[j] = dimension;

            //Existing position in employed_bees_searches selected using fit_i/Epsilon_SN__j=1 fit_j
            let selected_existing_position_idx = weighted_selection.sample(random_generator);

            onlooker_trial_search_points[j] =
                employed_bees_searches[selected_existing_position_idx].clone();
            onlooker_mapping_to_employed[j] = selected_existing_position_idx; //make sure we know which belongs to which -- makes parallelism possible

            //Select the index for an existing random food source
            let mut random_solution_index: usize =
                random_generator.gen_range(0..self.employed_bees);
            while random_solution_index == selected_existing_position_idx {
                random_solution_index = random_generator.gen_range(0..self.employed_bees)
            }

            // Modify initial positions by xij + phi_ij(xij − xBSFj) - See Fairee et. Al,  Reinforcement learning for solution updating in Artificial Bee Colony
            let existing_sol = employed_bees_searches[selected_existing_position_idx][dimension];

            //Modify
            let tentative_new = existing_sol
                + (random_generator.gen_range::<f64, _>(-1.0..1.0000000000000002)
                    // * (common_reinforcement_vector[dimension] * d)
                    * self.sigmoid(common_reinforcement_vector[dimension] * d)
                    //* (existing_sol - (self.min_max_point[dimension])));
            *(existing_sol - employed_bees_searches[random_solution_index][dimension]));

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
        }
    }

    //Update food sources and revert trial positions for employed bees
    fn update_employed_food_source_and_trials(
        self: &mut Self,
        new_search_vec: &Vec<f64>,                  //New food source values
        food_source_values: &mut Vec<f64>,          //Existing food source values
        employed_bees_searches: &mut Vec<Vec<f64>>, //Employed Bees Searces
        trial_search_points: &mut Vec<Vec<f64>>,    //Trial search points
        attempts_per_food_source: &mut Vec<usize>,  //Vector of attempts per food sources
        chosen_dimension_vec: &Vec<usize>,          //Vector of dimensions that were chosen per bee
    ) {
        //Update the food source and revert trial search point values
        for (idx, new_search) in new_search_vec.iter().enumerate() {
            if *new_search > food_source_values[idx] {
                // Update the employed bees searches to new points from trial_search_points if the new source has as higher fitness value
                employed_bees_searches[idx][chosen_dimension_vec[idx]] =
                    trial_search_points[idx][chosen_dimension_vec[idx]];
                //Update to new fitness value too
                food_source_values[idx] = *new_search;
                //If a better value of the food source was found, set the counter to 0 again
                attempts_per_food_source[idx] = 0;
            } else {
                //Revert trial_search_points[i][chosen_dimension_vec[idx]] back to employed_bees_searches[i][chosen_dimension_vec[idx]]
                //Important because we will have to check against this same value later.
                trial_search_points[idx][chosen_dimension_vec[idx]] =
                    employed_bees_searches[idx][chosen_dimension_vec[idx]];
            };
        }
    }

    //Update food sources and revert trial positions for employed bees, and update the REINFORCEMENT vector.
    fn reinforcement_update_employed_food_source_and_trials(
        self: &mut Self,
        new_search_vec: &Vec<f64>,                  //New food source values
        food_source_values: &mut Vec<f64>,          //Existing food source values
        employed_bees_searches: &mut Vec<Vec<f64>>, //Employed Bees Searces
        trial_search_points: &mut Vec<Vec<f64>>,    //Trial search points
        attempts_per_food_source: &mut Vec<usize>,  //Vector of attempts per food sources
        chosen_dimension_vec: &Vec<usize>,          //Vector of dimensions that were chosen per bee
        common_reinforcement_vector: &mut Vec<f64>, //Vector of reinforcement values
    ) {
        //Normalize values (if there are negative values, add the (modulus of the smallest negative value) +1)
        let abs_minimum_value = food_source_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs();

        let normalized_food_source_values: Vec<f64> = food_source_values
            .iter()
            .map(|x| *x + abs_minimum_value + 1f64)
            .collect();

        //calculate the denominator for alpha/beta using normalized food source values
        let sum_of_normalized = normalized_food_source_values.iter().sum::<f64>();
        //Update the food source and revert trial search point values
        for (idx, new_search) in new_search_vec.iter().enumerate() {
            //Calculate Alpha/Beta here
            let alpha_beta: f64 = normalized_food_source_values[idx] / sum_of_normalized;
            //println!("sum_of_normalized={}, normalized_value={}",sum_of_normalized, normalized_food_source_values[idx] );
            if *new_search > food_source_values[idx] {
                // Update the employed bees searches to new points from trial_search_points if the new source has as higher fitness value
                employed_bees_searches[idx][chosen_dimension_vec[idx]] =
                    trial_search_points[idx][chosen_dimension_vec[idx]];
                //Update to new fitness value too
                food_source_values[idx] = *new_search;
                //Set the counter to 0 again since a better food source value was found
                attempts_per_food_source[idx] = 0;

                //Update reinforcement vector here (apply positive reinforcement)
                for (idx2, value) in common_reinforcement_vector.iter_mut().enumerate() {
                    if idx2 == chosen_dimension_vec[idx] {
                        //assert_eq!((alpha_beta * (1.0 - *value))>0.0,true);
                        *value = *value + (alpha_beta * (1.0 - *value));
                    } else {
                        //assert_eq!( (1.0 - alpha_beta)>0.0,true);
                        *value = *value * (1.0 - alpha_beta);
                    }
                    //println!("{:?}",chosen_dimension_vec[idx]);
                }
            } else {
                //Revert trial_search_points[i][chosen_dimension_vec[idx]] back to employed_bees_searches[i][chosen_dimension_vec[idx]]
                //Important because we will have to check against this same value later.
                trial_search_points[idx][chosen_dimension_vec[idx]] =
                    employed_bees_searches[idx][chosen_dimension_vec[idx]];

                //Update reinforcement vector here (apply negative reinforcement)
                for (idx2, value) in common_reinforcement_vector.iter_mut().enumerate() {
                    if idx2 == chosen_dimension_vec[idx] {
                        //assert_eq!( (1.0 - alpha_beta)>0.0,true);
                        *value = *value * (1.0 - alpha_beta);
                    } else {
                        //assert_eq!( (*value * (1.0 - alpha_beta))>0.0,true);
                        *value = (alpha_beta / ((self.number_of_dimensions as f64) - 1.0))
                            + (*value * (1.0 - alpha_beta));
                    }
                }
            };
            //println!("{:?}", common_reinforcement_vector.iter().sum::<f64>());
        }
        //println!("{:?}", common_reinforcement_vector.iter().sum::<f64>());
    }

    //update food sources alone for onlooker bees
    fn update_onlooker_food_source(
        self: &mut Self,
        new_search_vec: &Vec<f64>,                  //New food source values
        food_source_values: &mut Vec<f64>,          //Existing food source values
        employed_bees_searches: &mut Vec<Vec<f64>>, //Employed Bees Searces
        trial_search_points: &Vec<Vec<f64>>,        //Trial search points
        attempts_per_food_source: &mut Vec<usize>,  //Vector of attempts per food sources
        onlooker_chosen_dimension_vec: &Vec<usize>, //Vector of dimensions that were chosen by onlooker bees
        onlooker_mapping_to_employed: &Vec<usize>, //Vector of mappings for each onlooker bee to the employed bees' indices.
    ) {
        //Update the food source and revert trial search point values
        for (idx, new_search) in new_search_vec.iter().enumerate() {
            if *new_search > food_source_values[onlooker_mapping_to_employed[idx]] {
                // Update to new points if the new source has as higher fitness value
                employed_bees_searches[onlooker_mapping_to_employed[idx]]
                    [onlooker_chosen_dimension_vec[idx]] =
                    trial_search_points[idx][onlooker_chosen_dimension_vec[idx]];
                //Update to new fitness value too
                food_source_values[onlooker_mapping_to_employed[idx]] = *new_search;
                //If a better value of the food source was found, set the counter to 0 again
                attempts_per_food_source[onlooker_mapping_to_employed[idx]] = 0;
            }
            //no need to do anything else here, as trial_search_points[j] will be assigned afresh with each iteration
        }
    }

    //update food sources alone for onlooker bees --Extra, not in algorithm?
    fn reinforcement_update_onlooker_food_source(
        self: &mut Self,
        new_search_vec: &Vec<f64>,                  //New food source values
        food_source_values: &mut Vec<f64>,          //Existing food source values
        employed_bees_searches: &mut Vec<Vec<f64>>, //Employed Bees Searces
        trial_search_points: &Vec<Vec<f64>>,        //Trial search points
        attempts_per_food_source: &mut Vec<usize>,  //Vector of attempts per food sources
        onlooker_chosen_dimension_vec: &Vec<usize>, //Vector of dimensions that were chosen by onlooker bees
        onlooker_mapping_to_employed: &Vec<usize>, //Vector of mappings for each onlooker bee to the employed bees' indices.
        common_reinforcement_vector: &mut Vec<f64>,
    ) {
        //Normalize values (if there are negative values, add the (modulus of the smallest negative value) +1)
        let abs_minimum_value = food_source_values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs();

        let normalized_food_source_values: Vec<f64> = food_source_values
            .iter()
            .map(|x| *x + abs_minimum_value + 1f64)
            .collect();

        //calculate the denominator for alpha/beta using normalized food source values
        let sum_of_normalized = normalized_food_source_values.iter().sum::<f64>();
        //Update the food source and revert trial search point values
        for (idx, new_search) in new_search_vec.iter().enumerate() {
            let alpha_beta: f64 = normalized_food_source_values[onlooker_mapping_to_employed[idx]]
                / sum_of_normalized;

            if *new_search > food_source_values[onlooker_mapping_to_employed[idx]] {
                // Update to new points if the new source has as higher fitness value
                employed_bees_searches[onlooker_mapping_to_employed[idx]]
                    [onlooker_chosen_dimension_vec[idx]] =
                    trial_search_points[idx][onlooker_chosen_dimension_vec[idx]];
                //Update to new fitness value too
                food_source_values[onlooker_mapping_to_employed[idx]] = *new_search;
                //If a better value of the food source was found, set the counter to 0 again
                attempts_per_food_source[onlooker_mapping_to_employed[idx]] = 0;

                //Update reinforcement vector here (apply positive reinforcement)
                for (idx2, value) in common_reinforcement_vector.iter_mut().enumerate() {
                    if idx2 == onlooker_chosen_dimension_vec[onlooker_mapping_to_employed[idx]] {
                        *value = *value + (alpha_beta * (1.0 - *value));
                    } else {
                        *value = *value * (1.0 - alpha_beta);
                    }
                    //println!("{:?}",chosen_dimension_vec[idx]);
                }
            } else {
                //Update reinforcement vector here (apply negative reinforcement)
                for (idx2, value) in common_reinforcement_vector.iter_mut().enumerate() {
                    if idx2 == onlooker_chosen_dimension_vec[onlooker_mapping_to_employed[idx]] {
                        *value = *value * (1.0 - alpha_beta);
                    } else {
                        *value = (alpha_beta / ((self.number_of_dimensions as f64) - 1.0))
                            + (*value * (1.0 - alpha_beta));
                    }
                }
            }
            //no need to do anything else here, as trial_search_points[j] will be assigned afresh with each iteration
        }
    }

    //upate scout bees and other tracking variables in place without returning anything
    fn limit_exceeded_update_positions(
        self: &mut Self,
        random_generator: &mut rand::rngs::ThreadRng, //Generator object for generating random values
        scout_food_sources_values: &mut Vec<f64>,     //Values for scout food sources
        employed_bees_searches: &mut Vec<Vec<f64>>,   //Coordinates for employed bees searches.
        temporary_scout_searches: &mut Vec<Vec<f64>>, //Coordinates for temporary scout searches
        exceeded_max: &mut Vec<usize>, //Vector of food sources which exceeded the iteration limit labelled by index
        food_source_values: &mut Vec<f64>, //Value of the food source/reward
        scout_bees_searches: &mut Vec<Vec<f64>>, //Coordinates of searches made by the permanently-assigned scout bees
                                                 // problem_space_bounds: &Vec<[f64; 2]>, //Problem space bounds (upper and lower limit inclusive)
    ) {
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
                *each_dimension = random_generator.gen_range::<f64, _>(
                    self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                )
            }

            temporary_scout_searches.push(employed_bees_searches[*i].clone());
            //Perform search
            //println!("Updating position {}",*i); //Tested to be OK, does not run if scout bees' solutions have been written
        }
    }

    //Get euclidean midpoint of the current set of employed bees.
    fn euclidean_mid(employed_bees_searches: &Vec<Vec<f64>>) -> Vec<f64> {
        let n: f64 = employed_bees_searches.len() as f64;
        let d = employed_bees_searches[0].len();
        let mut results = vec![0.0f64; d];

        for value in employed_bees_searches.iter() {
            results = Optimizer::add_elementwise(&results, &value);
        }

        results.iter().map(|x| x / n).collect()
    }

    //get the magnitude of an n-dimensional vector
    fn n_dim_magnitude(vector: &Vec<f64>) -> f64 {
        vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    }

    //Karaboga's classic ABC
    //Defaults to parallel. To make it non-parallel, use Optimizer.not_parallel
    pub fn abc(
        self: &mut Self,
        problem_space_bounds: &Vec<[f64; 2]>,
        max_generations: u64,
        fitness_function: fn(&Vec<f64>) -> f64,
    ) {
        //Start timer here
        let function_real_time = Instant::now();

        //Set thread pool size based on available threads IF parallel mode is on.
        match self.parallel_mode{
        //If parallel mode is on, set thread pool size
        true =>
        match self.thread_pool_size {
            1 => println!("Running with {} threads. To change, use the builder method set_thread_pool(desired_pool_size). 
            For example: NewOptimizer::new().set_thread_pool(7)",self.thread_pool_size), 
            _=>println!("Running in parallel with {} threads.",self.thread_pool_size)
        },
        //Otherwise continue without doing anything else.
        false=> println!("Parallel mode set to off. Running sequentially and with no parallel operations")      
        }

        //set up thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_pool_size)
            .build()
            .unwrap();

        //Set metadata for this function.
        self.problem_space_bounds = problem_space_bounds.to_vec();
        self.number_of_dimensions = self.problem_space_bounds.len() as usize;
        self.max_generations = max_generations;
        self.algorithm_name = String::from("abc");
        self.algorithm_description=String::from("Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems");

        //Ensure that the metadata set does not have an unacceptable value.
        if self.employed_bees < 2 {
            panic!("Number of employed bees should be greater than or equal to 2");
        }

        //default of this algorithim is maximization, set to 'false' to minimize instead.
        let minmax_factor = if self.maximize { 1.0f64 } else { -1.0f64 };

        //Set up RNG
        let mut random_generator = rand::thread_rng();

        //Generate value for RabC that is not 0
        //let random_float:f64=random_generator.gen_range(f64::MIN_POSITIVE..1.0);

        //BEGIN ALGORITHM HERE:

        //INITIALIZE e employed bee positions in n dimensions: vec![vec![0.0f64 ... n], ... e]
        let mut employed_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees];

        //Generate intial solutions
        for each_search_point in employed_bees_searches.iter_mut() {
            for (idx, dimension) in each_search_point.iter_mut().enumerate() {
                //for every single dimension
                *dimension = random_generator.gen_range::<f64, _>(
                    self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                )
                //generate random values within problem space bounds.
            }
        }

        //Create an intermediate copy of the searches already made.
        let mut trial_search_points = employed_bees_searches.clone();
        //vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees]; //Create an intermediate copy of the searches already made.

        //Perform initial search with employed bees on every single point
        let mut food_source_values: Vec<f64> = match self.parallel_mode {
            //run in parallel
            true => thread_pool.install(|| {
                employed_bees_searches
                    .par_iter()
                    .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                    .collect()
            }),
            false => employed_bees_searches
                .iter()
                .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                .collect(),
        };

        self.update_metadata(
            &food_source_values,
            &employed_bees_searches,
            minmax_factor,
            employed_bees_searches.len(),
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
            //Update the employed bees positions in trial_search_points
            self.update_employed_bees(
                &mut random_generator,
                &mut attempts_per_food_source,
                &mut trial_search_points,
                &mut chosen_dimension_vec,
                &mut employed_bees_searches,
            );

            let new_search_vec: Vec<f64> = match self.parallel_mode {
                //Run the searches in parallel on the trial search points for employed bees
                true => thread_pool.install(|| {
                    trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                //Run searches sequentially
                false => trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            self.update_employed_food_source_and_trials(
                &new_search_vec,               //New food source values
                &mut food_source_values,       //Existing food source values
                &mut employed_bees_searches,   //Employed Bees Searches
                &mut trial_search_points,      //Trial search points
                &mut attempts_per_food_source, //Vector of attempts per food sources
                &chosen_dimension_vec,         //Vector of dimensions that were chosen per bee
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.employed_bees, //Here, we are sure that the number of searches= number of employed bees
            );

            self.update_onlooker_bees(
                &mut food_source_values,
                &mut random_generator,
                &mut onlooker_chosen_dimension_vec,
                &mut onlooker_trial_search_points,
                &mut employed_bees_searches,
                &mut onlooker_mapping_to_employed,
            );

            //Run searches
            let new_search_vec = match self.parallel_mode {
                //Run searches in parallel
                true => thread_pool.install(|| {
                    onlooker_trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                false => onlooker_trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            self.update_onlooker_food_source(
                &new_search_vec,                //New food source values
                &mut food_source_values,        //Existing food source values
                &mut employed_bees_searches,    //Employed Bees Searces
                &onlooker_trial_search_points,  //Trial search points
                &mut attempts_per_food_source,  //Vector of attempts per food sources
                &onlooker_chosen_dimension_vec, //Vector of dimensions that were chosen by onlooker bees
                &onlooker_mapping_to_employed, //Vector of mappings for each onlooker bee to the employed bees' indices.
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.onlooker_bees,
            );

            //Send Scout Bee out
            if self.permanent_scout_bees > 0 {
                //So long as there is 1 or more scout bee:
                for k in 0..self.permanent_scout_bees {
                    //Generate initial solutions -> randomly reach out with the scout bee
                    for (idx, each_dimension) in trial_scout_bees_searches[k].iter_mut().enumerate()
                    {
                        //for every single dimension
                        *each_dimension = random_generator.gen_range::<f64, _>(
                            self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                        )
                        //generate random values within problem space bounds.
                    }
                }

                //Run searches
                let new_search_vec: Vec<f64> = match self.parallel_mode {
                    //Perform search in parallel
                    true => thread_pool.install(|| {
                        trial_scout_bees_searches
                            .par_iter()
                            .map(|x| minmax_factor * fitness_function(x))
                            .collect()
                    }),
                    //perform search sequentially
                    false => trial_scout_bees_searches
                        .iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect(),
                };

                self.update_metadata(
                    &food_source_values,
                    &employed_bees_searches,
                    minmax_factor,
                    self.permanent_scout_bees,
                );

                //If replace with new value if search result is better. Started with f64::NEG_INFINITY aka update food source
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
                //Update positions with new values from permanent scout bees, and if those are used up, convert inactive employed bees to scout bees temporarily
                self.limit_exceeded_update_positions(
                    &mut random_generator,
                    &mut scout_food_sources_values,
                    &mut employed_bees_searches,
                    &mut temporary_scout_searches,
                    &mut exceeded_max,
                    &mut food_source_values,
                    &mut scout_bees_searches,
                    //&self.problem_space_bounds,
                );

                //To be run only if the number of food sources that exceeded their max limit is greater than the number of permanent scout bees on duty
                let temporary_scout_food: Vec<f64> = match self.parallel_mode {
                    //run in parallel
                    true => thread_pool.install(|| {
                        temporary_scout_searches
                            .par_iter()
                            .map(|x| minmax_factor * fitness_function(x))
                            .collect()
                    }),
                    false => temporary_scout_searches
                        .iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect(),
                };

                for idx in self.permanent_scout_bees..exceeded_max.len() {
                    //exceeded_max contains the INDEX values for food_source_values that need to be replaced.
                    food_source_values[exceeded_max[idx]] =
                        temporary_scout_food[idx - self.permanent_scout_bees]; //Deduct by the offset caused by the permanent_scout_bees
                                                                               // println!(
                                                                               //     "Replacing with temporary_scout_food number {}",
                                                                               //     idx - self.permanent_scout_bees
                                                                               // );
                }

                self.update_metadata(
                    &food_source_values,
                    &employed_bees_searches,
                    minmax_factor,
                    exceeded_max.len(),
                );
            }
            exceeded_max.clear(); //Reset the counters for which dimensions have exceeded maximum here, as we have already dealt with them
            temporary_scout_searches.clear() //reset the vector that holds the temporary scout searches
        }
        self.post_process_metadata();

        self.real_time_taken = function_real_time.elapsed();
    }

    //ABC with reinforcement learning
    pub fn rabc(
        self: &mut Self,
        problem_space_bounds: &Vec<[f64; 2]>,
        max_generations: u64,
        fitness_function: fn(&Vec<f64>) -> f64,
    ) {
        //Start timer here
        let function_real_time = Instant::now();

        //Set thread pool size based on available threads IF parallel mode is on.
        match self.parallel_mode{
        //If parallel mode is on, set thread pool size
        true =>
        match self.thread_pool_size {
            1 => println!("Running with {} threads. To change, use the builder method set_thread_pool(desired_pool_size). 
            For example: NewOptimizer::new().set_thread_pool(7)",self.thread_pool_size), 
            _=>println!("Running in parallel with {} threads.",self.thread_pool_size)
        },
        //Otherwise continue without doing anything else.
        false=> println!("Parallel mode set to off. Running sequentially and with no parallel operations")      
        }

        //set up thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_pool_size)
            .build()
            .unwrap();

        //Set metadata for this function.
        self.problem_space_bounds = problem_space_bounds.to_vec();
        self.number_of_dimensions = self.problem_space_bounds.len() as usize;
        self.max_generations = max_generations;
        self.algorithm_name = String::from("abc");
        self.algorithm_description=String::from("Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems");

        //Ensure that the metadata set does not have an unacceptable value.
        if self.employed_bees < 2 {
            panic!("Number of employed bees should be greater than or equal to 2");
        }

        //default of this algorithim is maximization, set to 'false' to minimize instead.
        let minmax_factor = if self.maximize { 1.0f64 } else { -1.0f64 };

        //Set up RNG
        let mut random_generator = rand::thread_rng();

        //Generate value for RabC that is not 0
        //let random_float:f64=random_generator.gen_range(f64::MIN_POSITIVE..1.0);

        //BEGIN ALGORITHM HERE:

        //INITIALIZE e employed bee positions in n dimensions: vec![vec![0.0f64 ... n], ... e]
        let mut employed_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees];

        //Generate intial solutions
        for each_search_point in employed_bees_searches.iter_mut() {
            for (idx, dimension) in each_search_point.iter_mut().enumerate() {
                //for every single dimension
                *dimension = random_generator.gen_range::<f64, _>(
                    self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                )
                //generate random values within problem space bounds.
            }
        }

        //Create an intermediate copy of the searches already made.
        let mut trial_search_points = employed_bees_searches.clone();
        //vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees]; //Create an intermediate copy of the searches already made.

        //Perform initial search with employed bees on every single point
        let mut food_source_values: Vec<f64> = match self.parallel_mode {
            //run in parallel
            true => thread_pool.install(|| {
                employed_bees_searches
                    .par_iter()
                    .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                    .collect()
            }),
            false => employed_bees_searches
                .iter()
                .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                .collect(),
        };

        self.update_metadata(
            &food_source_values,
            &employed_bees_searches,
            minmax_factor,
            employed_bees_searches.len(),
        );

        //Create an intermediate vector to hold the searches made for onlooker bees
        let mut onlooker_trial_search_points =
            vec![vec![0.0f64; self.number_of_dimensions]; self.onlooker_bees];
        let mut onlooker_mapping_to_employed = vec![0usize; self.onlooker_bees];

        //Create an intitial reinforcement vector
        let mut common_reinforcement_vector =
            //vec![1.0 / (self.number_of_dimensions as f64); self.number_of_dimensions];
            vec![1.0; self.number_of_dimensions];

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
            //Update the employed bees positions in trial_search_points
            // self.update_employed_bees(
            //     &mut random_generator,
            //     &mut attempts_per_food_source,
            //     &mut trial_search_points,
            //     &mut chosen_dimension_vec,
            //     &mut employed_bees_searches,
            // );

            self.reinforcement_update_employed_bees(
                &mut random_generator,
                &mut attempts_per_food_source,
                &mut trial_search_points,
                &mut chosen_dimension_vec,
                &mut employed_bees_searches,
                &common_reinforcement_vector,
            );

            let new_search_vec: Vec<f64> = match self.parallel_mode {
                //Run the searches in parallel on the trial search points for employed bees
                true => thread_pool.install(|| {
                    trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                //Run searches sequentially
                false => trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            self.reinforcement_update_employed_food_source_and_trials(
                &new_search_vec,               //New food source values
                &mut food_source_values,       //Existing food source values
                &mut employed_bees_searches,   //Employed Bees Searches
                &mut trial_search_points,      //Trial search points
                &mut attempts_per_food_source, //Vector of attempts per food sources
                &chosen_dimension_vec,         //Vector of dimensions that were chosen per bee
                &mut common_reinforcement_vector,
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.employed_bees, //Here, we are sure that the number of searches= number of employed bees
            );

            self.reinforcement_update_onlooker_bees(
                &mut food_source_values,
                &mut random_generator,
                &mut onlooker_chosen_dimension_vec,
                &mut onlooker_trial_search_points,
                &mut employed_bees_searches,
                &mut onlooker_mapping_to_employed,
                &common_reinforcement_vector,
            );

            //Run searches
            let new_search_vec = match self.parallel_mode {
                //Run searches in parallel
                true => thread_pool.install(|| {
                    onlooker_trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                false => onlooker_trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            // self.update_onlooker_food_source(
            //     &new_search_vec,                //New food source values
            //     &mut food_source_values,        //Existing food source values
            //     &mut employed_bees_searches,    //Employed Bees Searces
            //     &onlooker_trial_search_points,  //Trial search points
            //     &mut attempts_per_food_source,  //Vector of attempts per food sources
            //     &onlooker_chosen_dimension_vec, //Vector of dimensions that were chosen by onlooker bees
            //     &onlooker_mapping_to_employed, //Vector of mappings for each onlooker bee to the employed bees' indices.
            // );

            self.reinforcement_update_onlooker_food_source(
                &new_search_vec,                //New food source values
                &mut food_source_values,        //Existing food source values
                &mut employed_bees_searches,    //Employed Bees Searces
                &onlooker_trial_search_points,  //Trial search points
                &mut attempts_per_food_source,  //Vector of attempts per food sources
                &onlooker_chosen_dimension_vec, //Vector of dimensions that were chosen by onlooker bees
                &onlooker_mapping_to_employed, //Vector of mappings for each onlooker bee to the employed bees' indices.
                &mut common_reinforcement_vector,
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.onlooker_bees,
            );

            //Send Scout Bee out
            if self.permanent_scout_bees > 0 {
                //So long as there is 1 or more scout bee:
                for k in 0..self.permanent_scout_bees {
                    //Generate initial solutions -> randomly reach out with the scout bee
                    for (idx, each_dimension) in trial_scout_bees_searches[k].iter_mut().enumerate()
                    {
                        //for every single dimension
                        *each_dimension = random_generator.gen_range::<f64, _>(
                            self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                        )
                        //generate random values within problem space bounds.
                    }
                }

                //Run searches
                let new_search_vec: Vec<f64> = match self.parallel_mode {
                    //Perform search in parallel
                    true => thread_pool.install(|| {
                        trial_scout_bees_searches
                            .par_iter()
                            .map(|x| minmax_factor * fitness_function(x))
                            .collect()
                    }),
                    //perform search sequentially
                    false => trial_scout_bees_searches
                        .iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect(),
                };

                self.update_metadata(
                    &food_source_values,
                    &employed_bees_searches,
                    minmax_factor,
                    self.permanent_scout_bees,
                );

                //If replace with new value if search result is better. Started with f64::NEG_INFINITY aka update food source
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
                //Update positions with new values from permanent scout bees, and if those are used up, convert inactive employed bees to scout bees temporarily
                self.limit_exceeded_update_positions(
                    &mut random_generator,
                    &mut scout_food_sources_values,
                    &mut employed_bees_searches,
                    &mut temporary_scout_searches,
                    &mut exceeded_max,
                    &mut food_source_values,
                    &mut scout_bees_searches,
                    //&self.problem_space_bounds,
                );

                //To be run only if the number of food sources that exceeded their max limit is greater than the number of permanent scout bees on duty
                let temporary_scout_food: Vec<f64> = match self.parallel_mode {
                    //run in parallel
                    true => thread_pool.install(|| {
                        temporary_scout_searches
                            .par_iter()
                            .map(|x| minmax_factor * fitness_function(x))
                            .collect()
                    }),
                    false => temporary_scout_searches
                        .iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect(),
                };

                for idx in self.permanent_scout_bees..exceeded_max.len() {
                    //exceeded_max contains the INDEX values for food_source_values that need to be replaced.
                    food_source_values[exceeded_max[idx]] =
                        temporary_scout_food[idx - self.permanent_scout_bees]; //Deduct by the offset caused by the permanent_scout_bees
                                                                               // println!(
                                                                               //     "Replacing with temporary_scout_food number {}",
                                                                               //     idx - self.permanent_scout_bees
                                                                               // );
                }

                self.update_metadata(
                    &food_source_values,
                    &employed_bees_searches,
                    minmax_factor,
                    exceeded_max.len(),
                );
            }
            exceeded_max.clear(); //Reset the counters for which dimensions have exceeded maximum here, as we have already dealt with them
            temporary_scout_searches.clear() //reset the vector that holds the temporary scout searches
        }
        self.post_process_metadata();

        self.real_time_taken = function_real_time.elapsed();
    }

    //VS_RABC from my Final Year Thesis project
    pub fn vs_rabc(
        self: &mut Self,
        problem_space_bounds: &Vec<[f64; 2]>,
        max_generations: u64,
        fitness_function: fn(&Vec<f64>) -> f64,
    ) {
        //Start timer here
        let function_real_time = Instant::now();

        //Set thread pool size based on available threads IF parallel mode is on.
        match self.parallel_mode{
        //If parallel mode is on, set thread pool size
        true =>
        match self.thread_pool_size {
            1 => println!("Running with {} threads. To change, use the builder method set_thread_pool(desired_pool_size). 
            For example: NewOptimizer::new().set_thread_pool(7)",self.thread_pool_size), 
            _=>println!("Running in parallel with {} threads.",self.thread_pool_size)
        },
        //Otherwise continue without doing anything else.
        false=> println!("Parallel mode set to off. Running sequentially and with no parallel operations")      
        }

        //set up thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_pool_size)
            .build()
            .unwrap();

        //Set metadata for this function.
        self.problem_space_bounds = problem_space_bounds.to_vec();
        self.number_of_dimensions = self.problem_space_bounds.len() as usize;
        self.max_generations = max_generations;
        self.algorithm_name = String::from("abc");
        self.algorithm_description=String::from("Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems");
        //get max value of problem_space_bounds here:
        let mut vector_of_ranges = vec![];
        for i in problem_space_bounds.iter() {
            vector_of_ranges.push((i[0] - i[1]).abs());
        }

        self.search_distance = self.search_distance_factor
            * vector_of_ranges
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

        println!("Running vs_rabc with {:?} traversal searches per random vector, initial search distance of {:?}, and a search distance delta of {:?}",self.traversal_searches, self.search_distance, self.search_distance_delta);

        //Ensure that the metadata set does not have an unacceptable value.
        if self.employed_bees < 2 {
            panic!("Number of employed bees should be greater than or equal to 2");
        }

        //default of this algorithim is maximization, set to 'false' to minimize instead.
        let minmax_factor = if self.maximize { 1.0f64 } else { -1.0f64 };

        //Set up RNG
        let mut random_generator = rand::thread_rng();

        //Generate value for RabC that is not 0
        //let random_float:f64=random_generator.gen_range(f64::MIN_POSITIVE..1.0);

        //BEGIN ALGORITHM HERE:

        //INITIALIZE e employed bee positions in n dimensions: vec![vec![0.0f64 ... n], ... e]
        let mut employed_bees_searches =
            vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees];

        //Generate intial solutions
        for each_search_point in employed_bees_searches.iter_mut() {
            for (idx, dimension) in each_search_point.iter_mut().enumerate() {
                //for every single dimension
                *dimension = random_generator.gen_range::<f64, _>(
                    self.problem_space_bounds[idx][0]..self.problem_space_bounds[idx][1],
                )
                //generate random values within problem space bounds.
            }
        }

        //Create an intermediate copy of the searches already made.
        let mut trial_search_points = employed_bees_searches.clone();
        //vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees]; //Create an intermediate copy of the searches already made.

        //Perform initial search with employed bees on every single point
        let mut food_source_values: Vec<f64> = match self.parallel_mode {
            //run in parallel
            true => thread_pool.install(|| {
                employed_bees_searches
                    .par_iter()
                    .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                    .collect()
            }),
            false => employed_bees_searches
                .iter()
                .map(|x| -> f64 { minmax_factor * fitness_function(x) })
                .collect(),
        };

        self.update_metadata(
            &food_source_values,
            &employed_bees_searches,
            minmax_factor,
            employed_bees_searches.len(),
        );

        //Create an intermediate vector to hold the searches made for onlooker bees
        let mut onlooker_trial_search_points =
            vec![vec![0.0f64; self.number_of_dimensions]; self.onlooker_bees];
        let mut onlooker_mapping_to_employed = vec![0usize; self.onlooker_bees];

        //Create an intitial reinforcement vector
        let mut common_reinforcement_vector =
            //vec![1.0 / (self.number_of_dimensions as f64); self.number_of_dimensions];
            vec![1.0; self.number_of_dimensions];

        //create a vector to keep track of number of attempts made per food source
        let mut attempts_per_food_source = vec![1usize; self.employed_bees];

        //INITIALIZE scout bee array

        //Scout bee should search randomly at all times, and every time a
        //better solution is found, update to that solution.

        // let mut scout_bees_searches =
        //     vec![vec![0.0f64; self.number_of_dimensions]; self.permanent_scout_bees];
        // let mut trial_scout_bees_searches =
        //     vec![vec![0.0f64; self.number_of_dimensions]; self.permanent_scout_bees];
        // let mut scout_food_sources_values: Vec<f64> =
        //     vec![f64::NEG_INFINITY; self.permanent_scout_bees];

        let mut exceeded_max: Vec<usize> = vec![]; //Blank index that will contain (Index of points > max tries, )
        let mut temporary_scout_searches: Vec<Vec<f64>> = vec![];

        let mut chosen_dimension_vec = vec![0usize; self.employed_bees];
        let mut onlooker_chosen_dimension_vec = vec![0usize; self.onlooker_bees];

        //Set traversal searches
        //Loop through the algorithm here
        for iteration in 0..self.max_generations {
            self.reinforcement_update_employed_bees(
                &mut random_generator,
                &mut attempts_per_food_source,
                &mut trial_search_points,
                &mut chosen_dimension_vec,
                &mut employed_bees_searches,
                &common_reinforcement_vector,
            );

            let new_search_vec: Vec<f64> = match self.parallel_mode {
                //Run the searches in parallel on the trial search points for employed bees
                true => thread_pool.install(|| {
                    trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                //Run searches sequentially
                false => trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            self.reinforcement_update_employed_food_source_and_trials(
                &new_search_vec,               //New food source values
                &mut food_source_values,       //Existing food source values
                &mut employed_bees_searches,   //Employed Bees Searches
                &mut trial_search_points,      //Trial search points
                &mut attempts_per_food_source, //Vector of attempts per food sources
                &chosen_dimension_vec,         //Vector of dimensions that were chosen per bee
                &mut common_reinforcement_vector,
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.employed_bees, //Here, we are sure that the number of searches= number of employed bees
            );

            self.reinforcement_update_onlooker_bees(
                &mut food_source_values,
                &mut random_generator,
                &mut onlooker_chosen_dimension_vec,
                &mut onlooker_trial_search_points,
                &mut employed_bees_searches,
                &mut onlooker_mapping_to_employed,
                &common_reinforcement_vector,
            );

            //Run searches
            let new_search_vec = match self.parallel_mode {
                //Run searches in parallel
                true => thread_pool.install(|| {
                    onlooker_trial_search_points
                        .par_iter()
                        .map(|x| minmax_factor * fitness_function(x))
                        .collect()
                }),
                false => onlooker_trial_search_points
                    .iter()
                    .map(|x| minmax_factor * fitness_function(x))
                    .collect(),
            };

            self.reinforcement_update_onlooker_food_source(
                &new_search_vec,                //New food source values
                &mut food_source_values,        //Existing food source values
                &mut employed_bees_searches,    //Employed Bees Searces
                &onlooker_trial_search_points,  //Trial search points
                &mut attempts_per_food_source,  //Vector of attempts per food sources
                &onlooker_chosen_dimension_vec, //Vector of dimensions that were chosen by onlooker bees
                &onlooker_mapping_to_employed, //Vector of mappings for each onlooker bee to the employed bees' indices.
                &mut common_reinforcement_vector,
            );

            self.update_metadata(
                &food_source_values,
                &employed_bees_searches,
                minmax_factor,
                self.onlooker_bees,
            );

            //Check to see if maximum iterations has been reached for any bee
            for (idx, item) in attempts_per_food_source.iter_mut().enumerate() {
                if *item >= self.local_limit {
                    exceeded_max.push(idx); //contains index of the locations where the food sources are exhausted
                };
            }

            //only executed if max_length is exceeded.
            if exceeded_max.len() > 0 {
                if self.permanent_scout_bees > 0 {                    
                    let traversals =
                        ((exceeded_max.len() as f64) / (self.traversal_searches as f64)).ceil();
                    println!("traversals={:?}",traversals);
                    println!("exceeded_max.len()={:?}",exceeded_max.len());
                    //prepare vector to hold list of searches performed beforehand.
                    let mut ts_list: Vec<Vec<f64>> = vec![];
                    for _index1 in 0..(traversals as usize) {
                        //Get euclidean midpoint for all existing points.
                        let midpoint = Optimizer::euclidean_mid(&employed_bees_searches);

                        //Randomly select a new point, prand
                        let mut prand = vec![0.0f64; self.number_of_dimensions];
                        //generate random intial value for prand
                        for (idx, dimension) in prand.iter_mut().enumerate() {
                            //for every single dimension
                            *dimension = random_generator.gen_range::<f64, _>(
                                self.problem_space_bounds[idx][0]
                                    ..self.problem_space_bounds[idx][1],
                            )
                            //generate random values within problem space bounds.
                        }

                        //Create a vector rand_vector pointing from midpoint to prand
                        let rand_vector = Optimizer::deduct_elementwise(&prand, &midpoint);

                        //Calculate size of rand_vector. pdist = ||rand_vector||.
                        let pdist = Optimizer::n_dim_magnitude(&rand_vector);

                        let mut nfood = Optimizer::add_elementwise(
                            &midpoint,
                            &rand_vector
                                .iter()
                                .map(|x| (self.search_distance / pdist) * x)
                                .collect(),
                        );

                        //Ensure that the point nfood is not outside the limit of the space
                        //Check if out of bounds, if they are out of bounds set them to those bounds (Karaboga et al)
                        for dimension in 0..self.number_of_dimensions {
                            let (lower_bound, upper_bound) = (
                                self.problem_space_bounds[dimension][0],
                                self.problem_space_bounds[dimension][1],
                            );

                            if nfood[dimension] < lower_bound {
                                nfood[dimension] = lower_bound;
                            } else if nfood[dimension] > upper_bound {
                                nfood[dimension] = upper_bound;
                            }
                            //Otherwise, no changes.
                        }

                        ////Now, traverse the space between midpoint and nfood
                        //Sample the line between midpoint and ampling it evenly at TS points (Traversal Search points).

                        for i in 1..self.traversal_searches {
                            ts_list.push(
                                nfood
                                    .iter()
                                    .map(|x| x * ((i as f64) / (self.traversal_searches as f64)))
                                    .collect(),
                            );
                        }
                        ts_list.push(nfood.clone());
                    }
                    ts_list.truncate(exceeded_max.len()); //Only take the same number of searches as needed by exceeded_max
                                                          //ts_list will hold the list of coordinates that we are supposed to perform searches at

                    //Now, perform search for each member in that list.

                    //Run searches //New food source values
                    //NOTE: new_search_vec's values have been multiplied by minmax_factor
                    //This means that you should always try to maximize them
                    let new_search_vec = match self.parallel_mode {
                        //Run searches in parallel
                        true => thread_pool.install(|| {
                            ts_list
                                .par_iter()
                                .map(|x| minmax_factor * fitness_function(x))
                                .collect::<Vec<f64>>()
                        }),
                        false => ts_list
                            .iter()
                            .map(|x| minmax_factor * fitness_function(x))
                            .collect::<Vec<f64>>(),
                    };

                    //println!("new_search_vec={:?}", new_search_vec);

                    //Compare and record how many of the new solutions were better than the median of the old ones.
                    //Rank index of old solutions in descending numerical order
                    let mut old_solutions = vec![]; //Assemble a vector of the old solutions that cannot be improved anymore
                    for i in exceeded_max.iter() {
                        old_solutions.push(food_source_values[*i]);
                    }
                    let rank_vec = Optimizer::rank_vecf64_desc(&old_solutions);
                    //Now, get the median value
                    //If the number of solution values is even:
                    let remainder = rank_vec.len() % 2;
                    //println!("The number of solution values is: {}", rank_vec.len());
                    // println!(
                    //     "The value of that divided by two is: {}",
                    //     rank_vec.len() / 2
                    // );
                    //println!("The value of the remainder is: {}", rank_vec.len() % 2);
                    let median: f64 = if remainder == 0 {
                        //if even
                        (new_search_vec[(rank_vec.len() / 2) - 1]
                            + new_search_vec[rank_vec.len() / 2])
                            / 2.0 //use rank to get median values
                    }
                    //If the number of solution values is odd:
                    else {
                        new_search_vec[rank_vec.len() / 2]
                    };

                    //Now count the number of solutions that were better than the median.
                    let mut count_of_better = 0;
                    for item in new_search_vec.iter() {
                        if *item < median {
                        }
                        //if the number of items are less than the median:
                        else {
                            count_of_better += 1;
                        }
                    }

                    //println!("Median={}", median);
                    //println!("new_search_vec={:?}", new_search_vec);
                    //println!("Count={}", count_of_better);
                    //println!("Minmax_factor: {}", minmax_factor);
                    //If count of results that are better than the median of old searches is greater than or equal to half the number of solution values, MAINTAIN search distance.
                    //If not, then CHANGE it (randomly increase or decrease with a 50% probability)
                    if count_of_better < (new_search_vec.len() / 2) {
                        self.search_distance *= 1.0 - self.search_distance_delta;
                       
                        
                        // //change search distance using delta. 50% probability increase, 50% probability decrease
                        // if random_generator.gen_bool(0.01) {
                        //     self.search_distance *= 1.0 + self.search_distance_delta;
                        // //increase
                        // } else {
                        //     self.search_distance *= 1.0 - self.search_distance_delta;
                        //     //decrease
                        // }
                    }

                    //Now, replace the old search points and values with the new ones.
                    for (idx, value) in exceeded_max.iter().enumerate()
                    //for all the indices of the points where limit had been reached
                    {
                        employed_bees_searches[*value] = ts_list[idx].clone(); //search point
                        food_source_values[*value] = new_search_vec[idx]; //corresponding values
                    }
                }

                //Finally, update metadata here
                self.update_metadata(
                    &food_source_values,
                    &employed_bees_searches,
                    minmax_factor,
                    exceeded_max.len(),
                );
            }
            exceeded_max.clear(); //Reset the counters for which dimensions have exceeded maximum here, as we have already dealt with them
            temporary_scout_searches.clear() //reset the vector that holds the temporary scout searches
        }
        self.post_process_metadata();

        self.real_time_taken = function_real_time.elapsed();
    }

    pub fn get_results() {}

    //plot of x vs y - Both must be vectors of f64
    pub fn plot(
        x: Vec<f64>,
        y: Vec<f64>,
        x2: Vec<f64>,
        y2: Vec<f64>,
        x3: Vec<f64>,
        y3: Vec<f64>,
        filename: String,
    ) {
        //println!("{:?}",current_dir().unwrap().as_path());
        let mut mypath = current_dir().unwrap();
        mypath.push("results");
        //println!("cwd={:?}",mypath);
        fs::create_dir_all(mypath.as_path()).expect("Failed to create path.");

        let mut data: Vec<(f64, f64)> = Vec::new();
        for (idx, val) in x.iter().enumerate() {
            data.push((*val, y[idx]));
        }

        let mut data2: Vec<(f64, f64)> = Vec::new();
        for (idx, val) in x2.iter().enumerate() {
            data2.push((*val, y2[idx]));
        }

        let mut data3: Vec<(f64, f64)> = Vec::new();
        for (idx, val) in x3.iter().enumerate() {
            data3.push((*val, y3[idx]));
        }

        //println!("{:?}",data);
        //let data2 = [(1.0, 3.2), (2., 2.2), (3., 1.4), (4., 1.4), (5., 5.5)];

        //Get bounds for graph
        //x-axis
        let xmax = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let xmin = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let ymax = *y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let ymin = *y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // println!("{}", xmax);

        // println!("Position={:?}", y.iter().position(|y| *y == ymin).unwrap());

        let first_minimum = y.iter().position(|y| *y == ymin).unwrap();

        // println!("First value for ymin is {}", y[first_minimum]); //Find position of FIRST element that is the minimum
        // println!(
        //     "Number of iterations that it took to get minimum value is {}",
        //     x[first_minimum]
        // );

        // println!("{}", xmin);
        // println!("{}", ymax);
        // println!("{}", ymin);
        //y-axis
        let filename = format!("results/{}", filename.as_str());
        let filename = filename.as_str();
        let drawing_area = SVGBackend::new(filename, (1200, 800)).into_drawing_area();

        drawing_area.fill(&WHITE).unwrap();
        let mut chart_builder = ChartBuilder::on(&drawing_area);
        chart_builder
            .margin(27)
            .set_left_and_bottom_label_area_size(50);
        let mut chart_context = chart_builder
            .build_cartesian_2d(
                xmin..(xmax + (0.1 * xmax).abs()) as f64,
                (ymin - (0.1 * ymin).abs())..(ymax + (0.1 * ymax).abs()),
            )
            .unwrap();
        chart_context.configure_mesh().draw().unwrap();

        chart_context
            .draw_series(LineSeries::new(data, &BLACK))
            .unwrap()
            .label("Classic ABC")
            .legend(|(x, y)| Rectangle::new([(x - 15, y + 1), (x, y)], &BLACK));

        chart_context
            .draw_series(LineSeries::new(data2, &RED))
            .unwrap()
            .label("Reinforcement ABC")
            .legend(|(x, y)| Rectangle::new([(x - 15, y + 1), (x, y)], &RED));

        chart_context
            .draw_series(LineSeries::new(data3, &GREEN))
            .unwrap()
            .label("VS-RABC")
            .legend(|(x, y)| Rectangle::new([(x - 15, y + 1), (x, y)], &GREEN));
        // chart_context.draw_series(LineSeries::new(data2, RED)).unwrap().label("Series 2")
        //     .legend(|(x,y)| Rectangle::new([(x - 15, y + 1), (x, y)], RED));

        chart_context
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .margin(15)
            .legend_area_size(3)
            .border_style(BLUE)
            .background_style(BLUE.mix(0.1))
            .label_font(("Calibri", 12))
            .draw()
            .unwrap();
    }
}

pub struct AverageMetadata {
    pub searches_made_history: Vec<f64>,
    pub min_max_value_history: Vec<f64>,
    n: f64,
}

impl AverageMetadata {
    pub fn new() -> Self {
        Self {
            searches_made_history: Vec::new(),
            min_max_value_history: Vec::new(),
            n: 0.0,
        }
    }

    pub fn push(
        self: &mut Self,
        searches_made_history: &Vec<f64>,
        min_max_value_history: &Vec<f64>,
    ) {
        if self.searches_made_history.len() == 0 {
            self.searches_made_history = searches_made_history.clone();
        } else {
            self.searches_made_history =
                Optimizer::add_elementwise(&searches_made_history, &self.searches_made_history);
            // println!(
            //     "Added prooperly, we now have {:?}",
            //     self.searches_made_history
            // );
        }

        if self.min_max_value_history.len() == 0 {
            self.min_max_value_history = min_max_value_history.clone();
        } else {
            self.min_max_value_history =
                Optimizer::add_elementwise(&min_max_value_history, &self.min_max_value_history);
        }
        self.n += 1.0;
    }

    pub fn calculate_average(self: &mut Self) {
        self.searches_made_history = self
            .searches_made_history
            .iter()
            .map(|x| x / self.n)
            .collect();
        self.min_max_value_history = self
            .min_max_value_history
            .iter()
            .map(|x| x / self.n)
            .collect();
    }
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
    fn test_abc() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_rana = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them

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
        optimize_rana.abc(
            &problem_space_bounds,
            5u64,                  //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_rana.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        //println!("\n\nObject={:#?}\n\n", optimize_rana);
        println!(
            "Number of records for searches made = {:#?}",
            optimize_rana.searches_made_history.len()
        );

        println!("\nChecking the validity of metadata...");
        assert_eq!(
            *optimize_rana.searches_made_history.last().unwrap(),
            optimize_rana.total_searches_made
        );

        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_value_history.len()
        );
        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_point_history.len()
        );
        assert_eq!(optimize_rana.searches_made_history.len() > 0, true);

        println!("All checks run.");
    }

    #[test]
    fn test_abc_nonparallel() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_rana = Optimizer::new()
            .minimize()
            .set_thread_pool(2)
            .not_parallel(); //should strongly recommend user allow the system to decide for them

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
        optimize_rana.abc(
            &problem_space_bounds,
            5u64,                  //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_rana.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        //println!("\n\nObject={:#?}\n\n", optimize_rana);
        println!(
            "Number of records for searches made = {:#?}",
            optimize_rana.searches_made_history.len()
        );

        println!("\nChecking the validity of metadata...");
        assert_eq!(
            *optimize_rana.searches_made_history.last().unwrap(),
            optimize_rana.total_searches_made
        );

        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_value_history.len()
        );
        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_point_history.len()
        );

        assert_eq!(optimize_rana.searches_made_history.len() > 0, true);

        println!("All checks run.");
    }

    #[test]
    fn test_rabc() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_rana = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them

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
        optimize_rana.rabc(
            &problem_space_bounds,
            5u64,                  //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_rana.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        //println!("\n\nObject={:#?}\n\n", optimize_rana);
        println!(
            "Number of records for searches made = {:#?}",
            optimize_rana.searches_made_history.len()
        );

        println!("\nChecking the validity of metadata...");
        assert_eq!(
            *optimize_rana.searches_made_history.last().unwrap(),
            optimize_rana.total_searches_made
        );

        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_value_history.len()
        );
        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_point_history.len()
        );
        assert_eq!(optimize_rana.searches_made_history.len() > 0, true);

        println!("All checks run.");
    }

    #[test]
    fn test_vs_rabc() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_rana = Optimizer::new()
            .minimize()
            .set_limit(20)
            .set_thread_pool(2)
            .set_traversal_searches(4)
            .set_search_distance_factor(0.1)
            .set_search_distance_delta(0.1); //should strongly recommend user allow the system to decide for them for set_thread_pool

        //set custom metadata. Should be fine even if it's commented out.
        optimize_rana.fitness_function_name = String::from("rana");
        optimize_rana.fitness_function_description=String::from("N-dimensional, multimodal. Range = [-512,512] for all dimensions. minimum point for 2D: `[-488.632577, 512]`  Minimum value for 2D=-511.7328819,  Minimum point for 7D: `[-512, -512, -512, -512, -512, -512, -511.995602]` minimum value for 7D=-3070.2475210");
        optimize_rana.known_minimum_value = Some(-3070.2475210);
        optimize_rana.permanent_scout_bees = 1usize; //set this to 1 for vs_rabc for now
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
        optimize_rana.vs_rabc(
            &problem_space_bounds,
            35u64,                 //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_rana.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        //println!("\n\nObject={:#?}\n\n", optimize_rana);
        println!(
            "Number of records for searches made = {:#?}",
            optimize_rana.searches_made_history.len()
        );

        println!("\nChecking the validity of metadata...");
        assert_eq!(
            *optimize_rana.searches_made_history.last().unwrap(),
            optimize_rana.total_searches_made
        );

        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_value_history.len()
        );
        assert_eq!(
            optimize_rana.searches_made_history.len(),
            optimize_rana.min_max_point_history.len()
        );
        assert_eq!(optimize_rana.searches_made_history.len() > 0, true);

        println!("All checks run.");
    }

    #[test]
    fn test_abc_plotting() {
        //#region        
        //Set up problem space bounds
        let dimensionality = 900usize;
        let problem_space_bounds = vec![[-512.0, 512.0]; dimensionality]; //rana
        let problem_space_bounds_2 = vec![[-5.12, 5.12]; dimensionality]; //rastrigin
        let problem_space_bounds_3 = vec![[-5.0, 10.0]; dimensionality]; //rosenbrock
        let problem_space_bounds_4 = vec![[-500.0, 500.0]; dimensionality]; //schwefel

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");
        let traversal_searches = 100;
        let search_distance_factor = 0.2;
        let search_distance_delta = 0.1;
        //create new instance of struct
        let mut optimize_rana = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rana_2 = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rana_3 = Optimizer::new()
            .minimize()
            .set_thread_pool(2)
            .set_traversal_searches(traversal_searches)
            .set_search_distance_factor(search_distance_factor)
            .set_search_distance_delta(search_distance_delta); //should strongly recommend user allow the system to decide for them

        let mut optimize_rastrigin = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rastrigin_2 = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rastrigin_3 = Optimizer::new()
            .minimize()
            .set_thread_pool(2)
            .set_traversal_searches(traversal_searches)
            .set_search_distance_factor(search_distance_factor)
            .set_search_distance_delta(search_distance_delta); //should strongly recommend user allow the system to decide for them

        let mut optimize_rosenbrock = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rosenbrock_2 = Optimizer::new().minimize().set_thread_pool(2); //should strongly recommend user allow the system to decide for them
        let mut optimize_rosenbrock_3 = Optimizer::new()
            .minimize()
            .set_thread_pool(2)
            .set_traversal_searches(traversal_searches)
            .set_search_distance_factor(search_distance_factor)
            .set_search_distance_delta(search_distance_delta); //should strongly recommend user allow the system to decide for them

        let mut optimize_schwefel = Optimizer::new().minimize().set_thread_pool(2);
        let mut optimize_schwefel_2 = Optimizer::new().minimize().set_thread_pool(2);
        let mut optimize_schwefel_3 = Optimizer::new()
            .minimize()
            .set_thread_pool(2)
            .set_traversal_searches(traversal_searches)
            .set_search_distance_factor(search_distance_factor)
            .set_search_distance_delta(search_distance_delta);

        let total_iterations: u64 = 300;
        // println!(
        //     "Running test for {:?} dimensions over {:?} generations.",
        //     problem_space_bounds.len(),
        //     total_iterations
        // );

        //Run optimization function here.
        //take average of 20 runs for the key metadata
        //Create the structs to hold the average values

        let mut average_optimize_rana = AverageMetadata::new();
        let mut average_optimize_rana2 = AverageMetadata::new();
        let mut average_optimize_rana3 = AverageMetadata::new();

        let mut average_optimize_rastrigin = AverageMetadata::new();
        let mut average_optimize_rastrigin2 = AverageMetadata::new();
        let mut average_optimize_rastrigin3 = AverageMetadata::new();

        let mut average_optimize_rosenbrock = AverageMetadata::new();
        let mut average_optimize_rosenbrock2 = AverageMetadata::new();
        let mut average_optimize_rosenbrock3 = AverageMetadata::new();

        let mut average_optimize_schwefel = AverageMetadata::new();
        let mut average_optimize_schwefel2 = AverageMetadata::new();
        let mut average_optimize_schwefel3 = AverageMetadata::new();

        for i in 0..30 {
            println!("\nCurrently on iteration {i}\n");

            optimize_rana.abc(
                &problem_space_bounds,
                total_iterations,      //Max number of generations
                benchmark_algos::rana, //name of fitness function
            );

            average_optimize_rana.push(
                &optimize_rana
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rana.min_max_value_history,
            );

            optimize_rana.clear();

            optimize_rana_2.rabc(
                &problem_space_bounds,
                total_iterations,      //Max number of generations
                benchmark_algos::rana, //name of fitness function
            );

            average_optimize_rana2.push(
                &optimize_rana_2
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rana_2.min_max_value_history,
            );
            optimize_rana_2.clear();

            optimize_rana_3.vs_rabc(
                &problem_space_bounds,
                total_iterations,      //Max number of generations
                benchmark_algos::rana, //name of fitness function
            );

            average_optimize_rana3.push(
                &optimize_rana_3
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rana_3.min_max_value_history,
            );
            optimize_rana_3.clear();

            optimize_rastrigin.abc(
                &problem_space_bounds_2,
                total_iterations,           //Max number of generations
                benchmark_algos::rastrigin, //name of fitness function
            );

            average_optimize_rastrigin.push(
                &optimize_rastrigin
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rastrigin.min_max_value_history,
            );
            optimize_rastrigin.clear();

            optimize_rastrigin_2.rabc(
                &problem_space_bounds_2,
                total_iterations,           //Max number of generations
                benchmark_algos::rastrigin, //name of fitness function
            );

            average_optimize_rastrigin2.push(
                &optimize_rastrigin_2
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rastrigin_2.min_max_value_history,
            );
            optimize_rastrigin_2.clear();

            optimize_rastrigin_3.vs_rabc(
                &problem_space_bounds_2,
                total_iterations,           //Max number of generations
                benchmark_algos::rastrigin, //name of fitness function
            );

            average_optimize_rastrigin3.push(
                &optimize_rastrigin_3
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rastrigin_3.min_max_value_history,
            );
            optimize_rastrigin_3.clear();

            optimize_rosenbrock.abc(
                &problem_space_bounds_3,
                total_iterations,            //Max number of generations
                benchmark_algos::rosenbrock, //name of fitness function
            );

            average_optimize_rosenbrock.push(
                &optimize_rosenbrock
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rosenbrock.min_max_value_history,
            );
            optimize_rosenbrock.clear();

            optimize_rosenbrock_2.rabc(
                &problem_space_bounds_3,
                total_iterations,            //Max number of generations
                benchmark_algos::rosenbrock, //name of fitness function
            );

            average_optimize_rosenbrock2.push(
                &optimize_rosenbrock_2
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rosenbrock_2.min_max_value_history,
            );
            optimize_rosenbrock_2.clear();

            optimize_rosenbrock_3.vs_rabc(
                &problem_space_bounds_3,
                total_iterations,            //Max number of generations
                benchmark_algos::rosenbrock, //name of fitness function
            );

            average_optimize_rosenbrock3.push(
                &optimize_rosenbrock_3
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_rosenbrock_3.min_max_value_history,
            );
            optimize_rosenbrock_3.clear();

            optimize_schwefel.abc(
                &problem_space_bounds_4,
                total_iterations,          //Max number of generations
                benchmark_algos::schwefel, //name of fitness function
            );

            average_optimize_schwefel.push(
                &optimize_schwefel
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_schwefel.min_max_value_history,
            );
            optimize_schwefel.clear();

            optimize_schwefel_2.rabc(
                &problem_space_bounds_4,
                total_iterations,          //Max number of generations
                benchmark_algos::schwefel, //name of fitness function
            );

            average_optimize_schwefel2.push(
                &optimize_schwefel_2
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_schwefel_2.min_max_value_history,
            );
            optimize_schwefel_2.clear();

            optimize_schwefel_3.vs_rabc(
                &problem_space_bounds_4,
                total_iterations,          //Max number of generations
                benchmark_algos::schwefel, //name of fitness function
            );

            average_optimize_schwefel3.push(
                &optimize_schwefel_3
                    .searches_made_history
                    .clone()
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<f64>>(),
                &optimize_schwefel_3.min_max_value_history,
            );
            optimize_schwefel_3.clear();
        }

        average_optimize_rana.calculate_average();
        average_optimize_rana2.calculate_average();
        average_optimize_rana3.calculate_average();
        average_optimize_rastrigin.calculate_average();
        average_optimize_rastrigin2.calculate_average();
        average_optimize_rastrigin3.calculate_average();
        average_optimize_rosenbrock.calculate_average();
        average_optimize_rosenbrock2.calculate_average();
        average_optimize_rosenbrock3.calculate_average();
        average_optimize_schwefel.calculate_average();
        average_optimize_schwefel2.calculate_average();
        average_optimize_schwefel3.calculate_average();
        ///////////////////////////////////////////////////////////////////////////

        println!("All checks run.");

        Optimizer::plot(
            average_optimize_rana.searches_made_history,
            average_optimize_rana.min_max_value_history,
            average_optimize_rana2.searches_made_history,
            average_optimize_rana2.min_max_value_history,
            average_optimize_rana3.searches_made_history,
            average_optimize_rana3.min_max_value_history,
            String::from("rana_results.svg"),
        );

        Optimizer::plot(
            average_optimize_rastrigin.searches_made_history,
            average_optimize_rastrigin.min_max_value_history,
            average_optimize_rastrigin2.searches_made_history,
            average_optimize_rastrigin2.min_max_value_history,
            average_optimize_rastrigin3.searches_made_history,
            average_optimize_rastrigin3.min_max_value_history,
            String::from("rastrigin_results.svg"),
        );

        Optimizer::plot(
            average_optimize_rosenbrock.searches_made_history,
            average_optimize_rosenbrock.min_max_value_history,
            average_optimize_rosenbrock2.searches_made_history,
            average_optimize_rosenbrock2.min_max_value_history,
            average_optimize_rosenbrock3.searches_made_history,
            average_optimize_rosenbrock3.min_max_value_history,
            String::from("rosenbrock_results.svg"),
        );

        Optimizer::plot(
            average_optimize_schwefel.searches_made_history,
            average_optimize_schwefel.min_max_value_history,
            average_optimize_schwefel2.searches_made_history,
            average_optimize_schwefel2.min_max_value_history,
            average_optimize_schwefel3.searches_made_history,
            average_optimize_schwefel3.min_max_value_history,
            String::from("schwefel_results.svg"),
        );

        //////////////////////////////
        //Alert user that the program has completed
        println!("\x07");
        println!("\x07");
        println!("\x07");
    }

    #[test]
    fn test_abc_all() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        //create new instance of struct
        let mut optimize_multiple = Optimizer::new().minimize().set_thread_pool(62);

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
        optimize_multiple.abc(
            &problem_space_bounds,
            100u64,                //Max number of generations
            benchmark_algos::rana, //name of fitness function
        );

        println!("\nTime taken ={:?}\n", optimize_multiple.real_time_taken);
        //println!("Time taken in seconds ={}",optimize_rana.real_time_taken.as_secs());

        println!("\n\nObject={:#?}\n\n", optimize_multiple);
    }

    #[test]
    fn run_code() {
        // assert_eq!(
        //     Optimizer::set_bounds(&vec![[-1.0, 1.0]], "[]"),
        //     vec![[-1.0, 1.0000000000000002]]
        // );
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
    fn test_abc_metadata() {
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
        optimize_rana.abc(
            &problem_space_bounds,
            10u64,                 //Max iterations
            benchmark_algos::rana, //name of fitness function
        );
        println!("\n\nObject={:#?}\n\n", optimize_rana);
    }

    #[test]
    fn test_abc_minimize_flag() {
        //create new instance of struct
        let optimize_rana = Optimizer::new().minimize();
        assert_eq!(optimize_rana.maximize, false); //make sure minimization flag was really applied!
    }

    #[test]
    #[should_panic]
    fn test_abc_invalid_options() {
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
        optimize_rana.abc(
            &problem_space_bounds,
            100u64,                //Max iterations
            benchmark_algos::rana, //name of fitness function
        )
    }

    #[test]
    fn test_euclidean_mid() {
        //#region
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

        //Lower and upper bound inclusive
        //problem_space_bounds = Optimizer::set_bounds(&problem_space_bounds, "[]");

        let test_vector_1 = vec![
            vec![12.1, 12.3, 12.3, 12.5, 12.6, 12.7, 12.8],
            vec![12.1, 12.3, 12.3, 12.5, 12.6, 12.7, 12.8],
            vec![12.1, 12.3, 12.3, 12.5, 12.6, 12.7, 12.8],
        ];
        assert_eq!(
            Optimizer::euclidean_mid(&test_vector_1),
            vec![
                12.1,
                12.300000000000002,
                12.300000000000002,
                12.5,
                12.6,
                12.699999999999998,
                12.800000000000002
            ]
        );
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
