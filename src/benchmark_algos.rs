//Constants
//Should treat this as a library crate
//Source for egg_holder minimum values: https://arxiv.org/pdf/2003.09867.pdf, for Generalized Penalized Function 2: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app1
#![allow(dead_code)]
use std::f64::consts::E;
use std::f64::consts::PI;

/// Unimodal. ONE global minimum of f(x)=0 at x=(0,...0)
///
/// Boundary values: `[-32.768, 32.768]` over all dimensions.
pub fn ackley(v: &Vec<f64>) -> f64 {
    20.0 + E
        - (v.iter().map(|x| ((2.0 * PI * x).cos())).sum::<f64>() / (v.len() as f64)).exp()
        - (20.0
            * (-0.2 * (v.iter().map(|x| (x.powi(2))).sum::<f64>() / (v.len() as f64)).sqrt()).exp())
}

///n-dimensional, Range = `[-10,10]`
///
///Multimodal. Global Minimun of f(x)=0 at x=(0,0,...0)
pub fn alpine_1(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| ((x * x.sin() + (0.1 * x)).abs())).sum()
}

///n-dimensional, has Range = `[0,10]`
///
///Global MAXIMUM of f(x)=2.808*number of dimensions at x=(7.917,7.917,...7.917)
///
///Note: The original alpine 2 (alpine_2_max) problem was actually a maximization problem. It has been negated to turn it to a minimization problem (alpine_2).
pub fn alpine_2_max(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| (x.sqrt() * x.sin())).product::<f64>()
}

/// The original alpine 2 (alpine_2_max) problem was actually a maximization problem. It has been negated to turn it to a minimization problem (alpine_2).
///
///n-dimensional. Range = `[0,10]`
///
///Global MINIMUM of f(x)=2.808*number of dimensions at x=(7.917,7.917,...7.917)
pub fn alpine_2(v: &Vec<f64>) -> f64 {
    -v.iter().map(|x| (x.sqrt() * x.sin())).product::<f64>() //Note: This needs turbofish because the compiler could not infer the type.
}

///2 dimensions ONLY. f(x)=0 at x=(3,0.5)
///
///SearchSpaceDimensions should =vector.size
///
///Range = `[-4.5,4.5]` for each dimension
pub fn beale(v: &Vec<f64>) -> f64 {
    (1.5 - v[0] + (v[0] * v[1])).powi(2)
        + (2.25 - v[0] + (v[0] * v[1].powi(2))).powi(2)
        + (2.625 - v[0] + (v[0] * v[1].powi(3))).powi(2)
}

///2 dimensions ONLY. f(x)=0 at x=(1.0,3.0)
///
///SearchSpaceDimensions should =vector.size
///
///Range = `[-10.0,10.0]` for each dimension
pub fn booth(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    (x1 + (2.0 * x2) - 7.0).powi(2) + ((2.0 * x1) + x2 - 5.0).powi(2)
}

///input for values must be a numpy array. 2D only, multimodal
///
///usually evaluated on: x1 `[-15,-5]`, x2 `[-3,3]`
///
///Min point f(x)=0 at (-10,1). Mathematically provable only one global min.
pub fn bukin_6(v: &Vec<f64>) -> f64 {
    (100.0 * (v[1] - (0.01 * v[0].powi(2))).abs().sqrt()) + (0.01 * (v[0] + 10.0).abs())
}

///2 dimensions only, multimodal.
///
///Evaluated on the `[-10, 10]`, for x1 and x2.
///
///Global minima of f(x)=-2.0621 at (-1.34941, -1.34941), (-1.34941, 1.34941), (1.34941, -1.34941), (1.34941, 1.34941)
pub fn cross_in_tray(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    -0.0001
        * (((x1.sin() * x2.sin())
            * (((100.0 - (((v[0].powi(2) + v[1].powi(2)).sqrt()) / PI)).abs()).exp()))
        .abs()
            + 1.0)
            .powf(0.1)
}

/// Global minimum of -1 if either variable =0. Range = `[-10,10]`
///
///2 Dimensional ONLY.
pub fn cross_leg_table(v: &Vec<f64>) -> f64 {
    -((v[0].sin()
        * v[1].sin()
        * ((100.0 - ((v[0].powi(2) + v[1].powi(2)).sqrt() / PI)).abs()).exp())
    .abs()
        + 1.0)
        .powf(-0.1)
}

///Multimodal, 2D only.
///
///Minimum of f(x)=-1 at x=(0,0). Range = -5.12,5.12
pub fn drop_wave(v: &Vec<f64>) -> f64 {
    -(1.0 + (12.0 * (v[0].powi(2) + v[1].powi(2)).sqrt()).cos())
        / (0.5 * (v[0].powi(2) + v[1].powi(2)) + 2.0)
}

///Range = `[-100,100]`
///
///Global Minimun of f(x)=-1 at x=(pi,pi). 2-dimensional only.
pub fn easom(v: &Vec<f64>) -> f64 {
    -(v[0]).cos() * (v[1]).cos() * (-(v[0] - PI).powi(2) - (v[1] - PI).powi(2)).exp()
}

///N-D Eggholder function. Range = `[-512,512]`
///
///Location of minimum for n=2 = (512,404.2319), Minimum point = -959.6406627106155
///
///Location of minimum for n=10 = (480.852413, 431.374221, 444.908694, 457.547223, 471.962527, 427.497291, 442.091345, 455.119420, 469.429312, 424.940608), Minimum point = -8291.2400675
pub fn egg_holder(v: &Vec<f64>) -> f64 {
    let mut result = 0.0;

    for i in 0..(v.len() - 1) {
        let sum1 = v[i + 1] + 47.0;
        //Performed multiple times throughout the algorithm
        let x = v[i];
        result += -((sum1 * (((sum1 + (x / 2.0)).abs()).sqrt()).sin())
            + (x * (((x - sum1).abs()).sqrt()).sin()));
    }

    result
}

///Range = `[-2,2]`. 2 dimensional.
///
///Optimal point location=`[0,-1]`, optimal value = 3.0
pub fn goldstein_price(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    (1.0 + ((x1 + x2 + 1.0).powi(2)
        * (19.0 - (14.0 * x1) + (3.0 * x1.powi(2)) - (14.0 * x2)
            + (6.0 * x1 * x2)
            + (3.0 * x2.powi(2)))))
        * (30.0
            + (((2.0 * x1) - (3.0 * x2)).powi(2)
                * (18.0 - (32.0 * x1) + (12.0 * x1.powi(2)) + (48.0 * x2) - (36.0 * x1 * x2)
                    + (27.0 * x2.powi(2)))))
}

///Range = `[-600,600]`
///
///Optimal point location=`[0,0,...0]`, optimal value = 0.0
pub fn griewank(v: &Vec<f64>) -> f64 {
    let sum = v.iter().map(|x| (x.powi(2))).sum::<f64>() / 4000.0;

    let mut prod = 1.0;
    for i in 0..v.len() {
        prod *= (v[i] / (((i + 1) as f64).sqrt())).cos();
    }

    sum - prod + 1.0
}

///2 Dimensions only. Range = `[-5,5]`, local minima at:
///f(x) =0 when x=(3,2)
///
///f(x)=0 when x=(-2.805118,3.131312)
///
///f(x)=0 when x=(-3.779310,-3.283186)
///
///f(x)=0 when x=(3.584428,-1.848126)
pub fn himmelblau(v: &Vec<f64>) -> f64 {
    let (x, y) = (v[0], v[1]);
    (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2)
}

///N-dimensional, multimodal, Range = `[0,10]`
///
///Location of minimum for n=4 = (3.065318, 1.531047, 0.405617, 0.393987), Minimum point = -0.6222807
///
///Location of minimum for n=3 = (3.042963, 1.482875, 0.166211), Minimum point = -0.5157855
pub fn keane(v: &Vec<f64>) -> f64 {
    let mut denominator = 0.0;
    for i in 0..v.len() {
        denominator += v[i].powi(2) * ((i + 1) as f64);
    }

    -(v.iter().map(|x| (x.cos().powi(4))).sum::<f64>()
        - (2.0 * (v.iter().map(|x| (x.cos().powi(2))).product::<f64>())))
    .abs()
        / denominator.sqrt()
}

///2 dimensions only. Any input domain, but normally `[-10,10]`. Multimodal.
///
///f(x)=0 when x=(1,1)
pub fn levy13(v: &Vec<f64>) -> f64 {
    let (x, y) = (v[0], v[1]);
    (3.0 * PI * x).sin().powi(2)
        + ((x - 1.0).powi(2) * (1.0 + (3.0 * PI * y).sin().powi(2)))
        + ((y - 1.0).powi(2) * (1.0 + (2.0 * PI * y).sin().powi(2)))
}

///2 dimensions only. Evaluated on `[-10,10]` on all dimensions.
///
///Minimum of 0.0 at (0.0,0.0)
pub fn matyas(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    (0.26 * (x1.powi(2) + x2.powi(2))) - (0.48 * (x1 * x2))
}

///2 dimensions only. Evaluated on `[-1.5,4.0]` for x1, and `[-3.0,4.0]` on all dimensions.
///
///Minimum of -1.9132 at (-0.54719,-1.54719)
pub fn mccormick(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    (x1 + x2).sin() + (x1 - x2).powi(2) - (1.5 * x1) + (2.5 * x2) + 1.0
}

/// N-dimensional, multimodal. Range = `[0,pi]`
///
///Location of minimum for n=10 = (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796), Minimum point = -9.6601517
pub fn michalewicz(v: &Vec<f64>) -> f64 {
    let mut result = 0.0;
    for i in 0..v.len() {
        let x = v[i];
        result += x.sin() * (((((i + 1) as f64) * x.powi(2)) / PI).sin()).powi(20);
    }
    -result
}

///k=100,a=10,m=4 , Range = `[-50,50]` , Generalized to n-dimensions
///
///Global Minimun of f(x)=0 at x=(-1,-1...-1)
///
///Dubious, might want to research further.
pub fn penalized1(v: &Vec<f64>) -> f64 {
    let const1 = PI / (v.len() as f64);
    //PI/n
    let const2 = 10.0 * (PI * (1.25 + (v[0] / 4.0))).sin().powi(2);
    //10 (sin(pi*y1)) ^2
    let const3 = ((v[v.len() - 1] + 1.0) / 4.0).powi(2);
    //(yn-1)^2
    let mut sum1 = 0.0;

    for i in 0..(v.len() - 1) {
        sum1 += ((v[i] + 1.0) / 4.0).powi(2)
            * (1.0 + (10.0 * (PI * (1.25 + (v[i + 1] / 4.0))).sin().powi(2)))
            + const3;
    }

    let sum2 = v
        .iter()
        .map(|x| {
            if *x > 10.0 {
                100.0 * ((x - 10.0).powi(4))
            } else if *x < -10.0 {
                100.0 * ((-x - 10.0).powi(4))
            } else {
                0.0
            }
        })
        .sum::<f64>();

    const1 * (const2 + sum1) + sum2
}

///k=100,a=5,m=4
///
///Generalized to n-dimensions. Range = `[-50,50]` for each dimension. Global Minimun of f(x)=0 at x=(1,1...1)  //had to get this corrected in some other paper.
///
///Dubious, might want to leave this one out.
pub fn penalized2(v: &Vec<f64>) -> f64 {
    let const2 = (3.0 * PI * v[0]).sin().powi(2) + (v[v.len() - 1] - 1.0).powi(2);
    let mut sum1 = 0.0;

    for i in 0..(v.len() - 1) {
        sum1 += (v[i] - 1.0).powi(2) * (1.0 + (3.0 * PI * v[i + 1]).sin().powi(2));
    }

    let sum2 = v
        .iter()
        .map(|x| {
            if *x > 5.0 {
                100.0 * ((x - 5.0).powi(4))
            } else if *x < -5.0 {
                100.0 * ((-x - 5.0).powi(4))
            } else {
                0.0
            }
        })
        .sum::<f64>();

    sum2 + 0.1 * (const2 + sum1)
}

/// N-dimensional, multimodal. Range = `[-512,512]`
///
/// minimum points:
///
/// 2D: `[-488.632577, 512]`  Minimum value =-511.7328819
///
/// 7D: `[-512, -512, -512, -512, -512, -512, -511.995602]` minimum value=-3070.2475210
pub fn rana(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..(v.len() - 1) {
        let (x, xp1) = (v[i], v[i + 1]);
        results += (v[i]
            * (((xp1 + x + 1.0).abs()).sqrt()).cos()
            * (((xp1 - x + 1.0).abs()).sqrt()).sin())
            + ((1.0 + xp1)
                * (((xp1 + x + 1.0).abs()).sqrt()).sin()
                * (((xp1 - x + 1.0).abs()).sqrt()).cos());
    }
    results
}

/// Range = `[-5.12,5.12]` for each dimension. Generalized to n dimensions.
///
/// Minimum when x=0 at all dimensions
pub fn rastrigin(v: &Vec<f64>) -> f64 {
    (10.0 * v.len() as f64)
        + v.iter()
            .map(|x| (x.powi(2)) - 10.0 * ((2.0 * PI * x).cos()))
            .sum::<f64>()
}

///Range = `[-5,10]`
///
///Generalized to n-dimensions. Global Minimun of f(x)=0 at x=(1,1...1)
pub fn rosenbrock(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..(v.len() - 1) {
        let x = v[i];
        results += 100.0 * (v[i + 1] - x.powi(2)).powi(2) + (x - 1.0).powi(2);
    }
    results
}

///One global minimum of f(x)=0 at x=(0,0,...0). Range: `[-100,100]` in n dimensions.
///
///Source: <https://www.researchgate.net/publication/316804716_Hybrid_genetic_deflated_Newton_method_for_global_optimisation#pf9>
pub fn schaffer6(v: &Vec<f64>) -> f64 {
    let sum_all: f64 = v.iter().map(|x| x.powi(2)).sum();
    -0.5 - (((sum_all).sin().powi(2) - 0.5) / (1.0 + (0.001 * sum_all.powi(2))).powi(2))
}

/// Range = `[-500,5500]`
///
/// Multimodal. Global minimum of 0 when xi=(420.9687,420.9687,420.9687,...420.9687) for all xi
///
/// Generalized to n-dimensions
pub fn schwefel(v: &Vec<f64>) -> f64 {
    418.9829 * (v.len() as f64) - v.iter().map(|x| x * x.abs().sqrt().sin()).sum::<f64>()
}

/// Range = `[-100,100]`
///
/// Unimodal. Single Minimum: 0 when xi=(0,0,0,...0) for all xi
///
/// Generalized to n-dimensions
pub fn schwefel12(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..v.len() {
        let mut sub_sum = 0.0;
        for j in 0..=i
        //make range inclusive
        {
            sub_sum += v[j];
        }
        results += sub_sum.powi(2);
    }
    results
}

///Generalized to n dimensions. Range= `[-500,500]` for each dimension.
///
///minimum point (roughly)= `[420.968746,420.968746,...420.968746]`. Value of minimum point = approx. -418.982887272433799807913601398*Number of Dimensions
pub fn schwefel226(v: &Vec<f64>) -> f64 {
    -v.iter().map(|x| x * x.abs().sqrt().sin()).sum::<f64>()
}

///Const for Shekel function
const SHEKEL4_A: [[f64; 4]; 10] = [
    [4.0, 4.0, 4.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0, 8.0],
    [6.0, 6.0, 6.0, 6.0],
    [3.0, 7.0, 3.0, 7.0],
    [2.0, 9.0, 2.0, 9.0],
    [5.0, 5.0, 3.0, 3.0],
    [8.0, 1.0, 8.0, 1.0],
    [6.0, 2.0, 6.0, 2.0],
    [7.0, 3.6, 7.0, 3.6],
];

///Const for Shekel function
const SHEKEL4_C: [f64; 10] = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5];

///4-dimensional, multimodal. MUST be 4 dimensional. Range = `[0,10]` for each dimension.
///
///Location of minimum  = (4.00004, 4.00013, 4.00004, 4.00013), Minimum point = -10.1532
///
///Source: MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
///
///min values provided by Certified Global Minima for a Benchmark of Difficult Optimization Problems, Vanaret et al.
pub fn shekel4_5(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..5 {
        let mut inner_sum = 0.0;
        for j in 0..4 {
            inner_sum += (v[j] - SHEKEL4_A[i][j]).powi(2);
        }
        results += 1.0 / (SHEKEL4_C[i] + inner_sum);
    }

    -results
}

///4-dimensional, multimodal. MUST be 4 dimensional. Range = `[0,10]` for each dimension.
///
///Location of minimum  = (4.00057,4.00069,3.99949,3.99961), Minimum point = -10.403
///
///Source: MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
///
///min values provided by Certified Global Minima for a Benchmark of Difficult Optimization Problems, Vanaret et al.
pub fn shekel4_7(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..7 {
        let mut inner_sum = 0.0;
        for j in 0..4 {
            inner_sum += (v[j] - SHEKEL4_A[i][j]).powi(2);
        }
        results += 1.0 / (SHEKEL4_C[i] + inner_sum);
    }

    -results
}

///4-dimensional, multimodal. MUST be 4 dimensional. Range = `[0,10]` for each dimension.
///
///Location of minimum  = (4.00075,4.00059,3.99966,3.99951), Minimum point = -10.5364
///
///Source: MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
///
///min values provided by Certified Global Minima for a Benchmark of Difficult Optimization Problems, Vanaret et al.
pub fn shekel4_10(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..10 {
        let mut inner_sum = 0.0;
        for j in 0..4 {
            inner_sum += (v[j] - SHEKEL4_A[i][j]).powi(2);
        }
        results += 1.0 / (SHEKEL4_C[i] + inner_sum);
    }

    -results
}

///Const for Shekel function
const SHEKEL10_A: [[f64; 10]; 30] = [
    [
        9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020,
    ],
    [
        9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374,
    ],
    [
        8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982,
    ],
    [
        2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426,
    ],
    [
        8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567,
    ],
    [
        7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208,
    ],
    [
        1.256, 3.605, 8.623, 6.905, 0.584, 8.133, 6.071, 6.888, 4.187, 5.448,
    ],
    [
        8.314, 2.261, 4.224, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762,
    ],
    [
        0.226, 8.858, 1.420, 0.945, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637,
    ],
    [
        7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.060, 5.204, 8.713, 8.247,
    ],
    [
        0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016,
    ],
    [
        2.699, 3.516, 5.874, 4.119, 4.461, 7.496, 8.817, 0.690, 6.593, 9.789,
    ],
    [
        8.327, 3.897, 2.017, 9.570, 9.825, 1.150, 1.395, 3.885, 6.354, 0.109,
    ],
    [
        2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.880, 0.564,
    ],
    [
        4.707, 5.579, 4.080, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.670,
    ],
    [
        8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826,
    ],
    [
        8.632, 4.409, 4.832, 5.768, 7.050, 6.715, 1.711, 4.323, 4.405, 4.591,
    ],
    [
        4.887, 9.112, 0.170, 8.967, 9.693, 9.867, 7.508, 7.770, 8.382, 6.740,
    ],
    [
        2.440, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.480, 9.950, 1.675,
    ],
    [
        6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258,
    ],
    [
        0.652, 2.343, 1.370, 0.821, 1.310, 1.063, 0.689, 8.819, 8.833, 9.070,
    ],
    [
        5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234,
    ],
    [
        3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.570, 6.540, 0.228, 0.027,
    ],
    [
        8.798, 0.880, 2.370, 0.168, 1.701, 3.680, 1.231, 2.390, 2.499, 0.064,
    ],
    [
        1.460, 8.057, 1.336, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224,
    ],
    [
        0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644,
    ],
    [
        0.679, 2.800, 5.523, 3.049, 2.968, 7.225, 6.730, 4.199, 9.614, 9.229,
    ],
    [
        4.263, 1.074, 7.286, 5.599, 8.291, 5.200, 9.214, 8.272, 4.398, 4.506,
    ],
    [
        9.496, 4.830, 3.150, 8.270, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732,
    ],
    [
        4.138, 2.562, 2.532, 9.661, 5.611, 5.500, 6.886, 2.341, 9.699, 6.500,
    ],
];

///Const for Shekel function
const SHEKEL10_C: [f64; 30] = [
    0.806, 0.517, 0.10, 0.908, 0.965, 0.669, 0.524, 0.902, 0.531, 0.876, 0.462, 0.491, 0.463,
    0.714, 0.352, 0.869, 0.813, 0.811, 0.828, 0.964, 0.789, 0.360, 0.369, 0.992, 0.332, 0.817,
    0.632, 0.883, 0.608, 0.326,
];

///10-dimensional, multimodal. MUST be 10 dimensional. Range = `[0,10]`
///
///Location of minimum for n=10 = (8.024968, 9.151929, 5.113991, 7.620959, 4.564020f, 4.711005, 2.996030, 6.125993, 0.734057, 4.981999), Minimum point = -10.2078768
///
///Source: MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization Ernesto P. Adorio
///
///min values provided by Certified Global Minima for a Benchmark of Difficult Optimization Problems, Vanaret et al.
pub fn shekel10(v: &Vec<f64>) -> f64 {
    let mut results = 0.0;
    for i in 0..30 {
        let mut inner_sum = 0.0;
        for j in 0..10 {
            inner_sum += (v[j] - SHEKEL10_A[i][j]).powi(2);
        }
        results += 1.0 / (SHEKEL10_C[i] + inner_sum);
    }

    -results
}

/// N-dimensional, multimodal
///
///Location of minimum for n=10 = (-1.517016, -1.403507, 1.517016, -1.403507, -1.517015, 1.403507), Minimum point = -7.4574764
///
///Range = `[-100,100]`
pub fn sine_envelope(v: &Vec<f64>) -> f64 {
    let mut result = 0.0;
    for i in 0..(v.len() - 1) {
        result += ((v[i + 1].powi(2) + v[i].powi(2)).sqrt() - 0.5)
            .sin()
            .powi(2)
            / ((0.001 * (v[i + 1].powi(2) + v[i].powi(2))) + 1.0).powi(2);
    }

    -(result + (0.5 * ((v.len() - 1) as f64)))
}

///Range=inf, but restrict to `[-100,100]`
///
///Generalized to n dimensions
///
///Minimum of f(x)=0 at x=(0,0,...0)
pub fn sphere(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| x.powi(2)).sum()
}

///Range=inf, but restrict to `[-100,100]`
///
///Generalized to n dimensions
///
///Minimum of f(x)=0 at x=(0.0,0.0,0.0,...0.0) or
pub fn step2(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| ((x + 0.5).floor()).powi(2)).sum()
}

///Multimodal, N-Dimensional, Range=`[-5,5]`
///
///Minimum of f(x)=-39.16599n at x=(-2.903534,...,-2.903534)
///
///source: <https://www.sfu.ca/~ssurjano/stybtang.html>
pub fn styblinski_tang(v: &Vec<f64>) -> f64 {
    0.5 * v
        .iter()
        .map(|x| x.powi(4) - (16.0 * x.powi(2)) + (5.0 * x))
        .sum::<f64>()
}

///Generalized to n dimensions, Range=inf, but restrict to `[-10,10]`
///
///Minimum of f(x)=0 at x=(0,0,...0)
///
///Source: <https://www.sfu.ca/~ssurjano/sumsqu.html>
pub fn sum_squares(v: &Vec<f64>) -> f64 {
    let mut result = 0.0;
    for i in 0..v.len() {
        result += ((i + 1) as f64) * v[i].powi(2);
    }
    result
}

///2 dimensions, Range= `[-5,5]`
///
///Minimum of f(x)=0 at x=(0,0)
///
///Source: <https://www.sfu.ca/~ssurjano/sumsqu.html>
pub fn three_hump_camel(v: &Vec<f64>) -> f64 {
    let (x1, x2) = (v[0], v[1]);
    (2.0 * x1.powi(2)) - (1.05 * x1.powi(4)) + (x1.powi(6) / 6.0) + (x1 * x2) + x2.powi(2)
}

///n-dimensions, Unimodal
///
///Range=`[-5,10]`
///
///Global minimum of f(x)=0 at x=(0,...0)
pub fn zakharov(v: &Vec<f64>) -> f64 {
    let sum1: f64 = v.iter().map(|x| x.powi(2)).sum();

    let mut sum2 = 0.0;
    for i in 0..v.len() {
        sum2 += 0.5 * ((i + 1) as f64) * v[i];
    }

    sum1 + sum2.powi(2) + sum2.powi(4)
}

// pub fn recommend_me_a_function() -> {

// }
