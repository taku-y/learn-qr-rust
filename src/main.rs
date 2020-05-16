extern crate tch;
use tch::nn;

use qr::{create_model, train_qr_model};

fn main() {
    // Device
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    
    // Initialize model
    let n_inputs = 1i64;
    let n_quants = 3i64;
    let model = create_model(&vs.root(), n_inputs, n_quants, vec![8]);

    // Training loop
    let model = train_qr_model(model, n_quants, device, &vs);
}
