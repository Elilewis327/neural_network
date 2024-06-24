/*
Eli Lewis
6/23/24
design idea

first layer = size of all inputs
    for example, 28x28 image is 784 nodes

layer one of weights

weighted sum of input - bias

sigmoid function to normalize into 0-1

middle layer 1 = 16 nodes

weights 2 

weighted sum of mid 1 - bias

sigmoid function to normalize into 0-1

middle layer 2 = 16 nodes

layer three of weights 

weighted sum of mid 2 - bias

sigmoid function to normalize into 0-1

last layer = all outputs 
    for example, for number reckognition it would be 10

*/


/* TODO: 
all of learning
finish the execution step
ensure x-dimmensional vectors can be serialized
file loading / deserializing
threads / gpu ??
*/

use std::fs::File;
use std::io::Write;
use std::io;
use rand::{rngs::ThreadRng, Rng};
use serde::{Serialize, Deserialize};
use serde_json;

fn main() {
    println!("Hello, world!");

    let mut n1: Network = Network::create_network(784,2,16,10);

    let input: Vec<f32> = (0..=784).map(|x| x as f32).collect();

    println!("{:?}", n1.exec(&input));
}

#[derive(Serialize)]
pub struct Network {
    input_size: usize, // 784, or a 28x28 image
    input_weights: Vec<Vec<f32>>, // 16 * 784, each input maps to each hidden layer
    hidden_layers: Vec<Vec<f32>>, // 2 * 16, 2 hidden layers, size of 16 each
    hidden_layer_count: usize, // 2
    hidden_layer_size: usize, // 16
    hidden_layer_weights: Vec<Vec<Vec<f32>>>,  // 1 , 16, 16
    output_layer_weights: Vec<Vec<f32>>, // 1, 16, 10
    output: Vec<f32>,
    output_size: usize,
}

impl Network {
    pub fn create_network(input_size: usize, hidden_layer_count: usize, hidden_layer_size: usize, output_size: usize) -> Self {
        let mut rng: ThreadRng = rand::thread_rng();

        // maps inputs -> hidden layer 1 with 784*16 weights
        //let input_weights: Vec<Vec<f32>> = (0..input_size).map(|_| (0..hidden_layer_size).map(|_| rng.gen_range(0.0..1.0)).collect()).collect();
        let input_weights: Vec<Vec<f32>> = vec![vec![0.5; hidden_layer_size]; input_size];


        let hidden_layers: Vec<Vec<f32>> = vec![vec![0.0; hidden_layer_size]; hidden_layer_count];

        let hidden_layer_weights: Vec<Vec<Vec<f32>>> = (0..hidden_layer_count-1).map(|_| (0..hidden_layer_size).map(|_| (0..hidden_layer_size).map(|_| rng.gen_range(0.0..1.0)).collect()).collect()).collect();

        let output_layer_weights: Vec<Vec<f32>> = (0..hidden_layer_size).map(|_| (0..output_size).map(|_| rng.gen_range(0.0..1.0)).collect()).collect();

        let output: Vec<f32> = vec![0.0; output_size];

        Self {
            input_size,
            input_weights,
            hidden_layers,
            hidden_layer_count,
            hidden_layer_size,
            hidden_layer_weights,
            output,
            output_size,
            output_layer_weights,
        }
    }

    fn reset_hidden_layers(&mut self){
        // Reset hidden layers to zero
        for layer in &mut self.hidden_layers {
            layer.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    pub fn exec(&mut self, input: &Vec<f32>) -> &Vec<f32> {

        // input weights
        for i in 0..self.input_size {
            for j in 0..self.hidden_layer_size {
                self.hidden_layers[0][j] += input[i] * self.input_weights[i][j];
            }
        }

        &self.hidden_layers[0]
    }

    fn serialize(&mut self) -> serde_json::Result<String>{
        serde_json::to_string(&self)
    }

    pub fn save(&mut self, filename: &str) -> Result<(), io::Error> {
        // open file
        let mut file: File = File::open(filename)?;
        
        // serialize data
        let str: String = self.serialize()
            .expect("Error serializing data.");

        // write data
        file.write_all(str.as_bytes())?;

        Ok(())
    }
}
