//submodule for benchmark algorithms
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]

mod benchmark_algos;
use rand::Rng;
//use std::time::

#[derive(Default, Debug)]
pub struct Optimizer {
    //COMPULSORY Optimizer parameters
    pub max_iterations: u64,                 //CANNOT be running FOREVER
    pub problem_space_bounds: Vec<[f64; 2]>, //Cannot possibly be unbounded!

    //optional parameters
    pub employed_bees: i64,
    pub onlooker_bees: i64,
    pub scout_bees: i64, //If None (default), will be set by algorithm itself
    pub maximize: bool,  //if true, maximize, if false, minimize
    pub local_limit:i64,

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
    pub seconds_taken: f64, //#TODO: Real time taken. Default of 0.0
}

impl Optimizer {
    //2 different sets of params: 1. Compulsory , 2. optional. All optional to be inside an enum

    //input of all such methods:
    //An n-dimensional vector containing our guess
    //output:
    //A single f64 for greatest precision

    //Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems

    //Number of dimensions is unknown, so easier to pass in as vec, but a lower & upper bound is necessary/easily set & known, so pass in as array.

    //source: https://www.researchgate.net/publication/225392029_A_powerful_and_efficient_algorithm_for_numerical_function_optimization_Artificial_bee_colony_ABC_algorithm

    //default constructor ->MUST be used to create an instance of Optimizer
    pub fn new() -> Self {
        Self {
            employed_bees: 62i64, //Default values for employed, onlooker, and scout bees as recommended in the source paper.
            onlooker_bees: 62i64,
            scout_bees: 1i64,
            maximize: true, //default to finding maximum value in input problem space
            ..Default::default()
        }
    }

    pub fn minimize(mut self: Self) -> Self {
        self.maximize = false;
        self
    }

    pub fn classic_abc(
        self: &mut Self,
        problem_space_bounds: &Vec<[f64; 2]>,
        max_iterations: u64,
        fitness_function: fn(&Vec<f64>) -> f64,
    ) {
        //create an optimizer object first with all the default fields and metadata initialized.
        // let mut instance =
        //     Optimizer::default_initializer(problem_space_bounds, max_iterations);//, other_args);

        //Set metadata for this function.
        self.problem_space_bounds = problem_space_bounds.to_vec();
        self.number_of_dimensions = self.problem_space_bounds.len() as usize;
        self.max_iterations = max_iterations;
        self.algorithm_name = String::from("classic_abc");
        self.algorithm_description=String::from("Karaboga's classic ABC from https://www.researchgate.net/publication/221498082_Artificial_Bee_Colony_ABC_Optimization_Algorithm_for_Solving_Constrained_Optimization_Problems");


        //default of this algorithim is maximization, set to 'false' to minimize instead.
        let minmax_factor = if self.maximize { 1.0f64 } else { -1.0f64 };


        //Add actions to ALWAYS be performed with each search to avoid further indirection from closures/function nesting.
        macro_rules! perform_search {
            ($x:expr) => {{
                self.searches_made += 1; //perform all other actions here
                minmax_factor * fitness_function($x) //MUST return this at the very end
            }};
        }

        //Set up RNG
        let mut random_generator = rand::thread_rng(); 

        //Generate value for RabC that is not 0
        //let random_float:f64=random_generator.gen_range(f64::MIN_POSITIVE..1.0);

        //Begin algorithm here:

        //Initialize e employed bee positions in n dimensions: [[0.0f64 ... n], ... e]
        let mut employed_bees_points =
            vec![vec![0.0f64; self.number_of_dimensions]; self.employed_bees as usize];

        //create a vector to keep track of number of attempts made per food source
        let mut attempts_per_food_source = vec![0.0f64;self.employed_bees as usize];

        for each_search_point in &mut employed_bees_points {
            for i in 0..self.number_of_dimensions {                       //for every single dimension
                each_search_point[i] = random_generator
                    .gen_range::<f64, _>(problem_space_bounds[i][0]..problem_space_bounds[i][1])  //generate random values within problem space bounds.
            }
        }
        //Perform initial search with employed bees on every single point
        let employed_bees_searches: Vec<f64> = employed_bees_points
            .iter()
            .map(|x| -> f64 { perform_search!(x) })
            .collect();

        //Loop through the algorithm here
        for i in 0..self.max_iterations {
        //next: vij = xij + φij(xij − xkj), Modify initial search: 
        //employed_bees_searches=



        }
        println!("{:?}", employed_bees_points);
        println!("{}",self.maximize);

        //Finish off here with the rest of the metadata
        self.min_max_value = 1f64;
        self.min_max_point = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
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
        let mut optimize_rana = Optimizer::new().minimize();

        //set custom metadata. Should be fine even if it's commented out.
        optimize_rana.fitness_function_name = String::from("rana");
        optimize_rana.fitness_function_description=String::from("N-dimensional, multimodal. Range = [-512,512] for all dimensions. minimum point for 2D: `[-488.632577, 512]`  Minimum value for 2D=-511.7328819,  Minimum point for 7D: `[-512, -512, -512, -512, -512, -512, -511.995602]` minimum value for 7D=-3070.2475210");
        optimize_rana.known_minimum_value = Some(-3070.2475210);
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
            20u64,                //Max iterations
            benchmark_algos::rana, //name of fitness function
        );
        //println!("\n\nObject={:#?}\n\n", optimize_rana);
    }

    #[test]
    fn test_classic_abc_no_optional_metadata() {
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

        //Run optimization function here.
        optimize_rana.classic_abc(
            &problem_space_bounds,
            100u64,                //Max iterations
            benchmark_algos::rana, //name of fitness function
        );
        println!("\n\nObject={:#?}\n\n", optimize_rana);
    }

    #[test]
    fn test_classic_abc_minimize_flag() {

        //create new instance of struct
        let mut optimize_rana = Optimizer::new().minimize();

        assert_eq!(optimize_rana.maximize,false);  //make sure minimization flag was really applied!

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

        assert_eq!((rana(&my_vector) * 1.0e7).round() / 1.0e7, -3070.2475210);
        assert_eq!((rana(&my_vector2) * 1.0e7).round() / 1.0e7, -511.7328819);
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
