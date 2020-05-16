use std::error::Error;
use std::fs::File;
use std::convert::{TryFrom};
use csv::ReaderBuilder;
use ndarray::{Array2, s};
use ndarray_csv::Array2Reader;
extern crate tch;
use tch::{Tensor, nn, Kind, nn::OptimizerConfig, nn::ModuleT, Device,
          nn::VarStore};

pub fn load_data() -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let pylib_dir = "/Users/taku-y/venvs/test/lib/python3.7/site-packages";
    let csv_file = "/sklearn/datasets/data/boston_house_prices.csv";
    let csv_file = [pylib_dir, csv_file].concat();
    let file = File::open(csv_file)?;

    let rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut iter = rdr.into_records();

    // Skip 2 lines
    iter.next();
    iter.next();
    let mut rdr = iter.into_reader();
    let mut data: Array2<f32> = rdr.deserialize_array2((506, 14))?;
    let xs = Tensor::try_from(data.slice_mut(s![.., 10..11]).to_owned())
        .unwrap();
    let ys = Tensor::try_from(data.slice_mut(s![.., 13..14]).to_owned())
        .unwrap();

    // Standardize output values
    let ys = (&ys - (&ys.mean(Kind::Float))) / &ys.std(true);

    Ok((xs, ys))
}

pub fn create_model(vs: &nn::Path, n_inputs: i64, n_quants: i64,
    n_unitss: Vec::<i64>) -> impl nn::ModuleT {
    let mut model = nn::seq_t();
    let mut n_inputs = n_inputs;

    // Hidden layers
    for (i, n_units) in n_unitss.iter().enumerate() {
    model = model.add(nn::linear(vs / i.to_string() , n_inputs, *n_units,
                Default::default()));
    model = model.add_fn(|xs| xs.sigmoid());
    n_inputs = *n_units;
    }

    // Output layer
    model = model.add(
    nn::linear(vs / "out", n_inputs, n_quants, Default::default())
    );

    return model;
}

fn huber(x: &Tensor) -> Tensor {
    // return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    let k = 0.1f64;
    let t1 = 0.5 * x.pow(2);
    let t2 = &x.abs() - (0.5 * k);

    t1.where1(&t1.lt(k), &t2)
}

pub fn quantile_huber_loss(z: &Tensor, theta: &Tensor, tau: &Tensor) -> Tensor {
    let u = z - theta;
    let s = tau - u.lt(0f64).totype(Kind::Float);

    (s.abs() * huber(&u)).mean(Kind::Float)
}

pub fn train_qr_model(model: impl ModuleT, n_quants: i64, device: Device,
    vs: &VarStore) -> impl ModuleT {

    let n_train = 20000;
    let n_disp_steps = 2000;

    // Load data
    let (xs, ys) = load_data().unwrap();

    // Quantiles
    let tau = Tensor::arange2(
        1. / (n_quants as f64 + 1.), 0.9999, 1. / (n_quants as f64 + 1.),
        (Kind::Float, device)
    );
    println!("Quantiles");
    tau.print();

    // Initialize optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for i in 0..n_train {
        // Compute loss value
        let theta = xs.apply_t(&model, true);
        let loss = quantile_huber_loss(&ys, &theta, &tau);
        if i % n_disp_steps == 0 {
            println!("iter = {:>5}, loss = {:>4}", &i, f64::from(&loss));
        }
        opt.backward_step(&loss);
    }

    model
}
