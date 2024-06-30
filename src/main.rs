#![allow(unused)]
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

only supports middle layers of a constant size

*/


/* TODO: 
all of learning
flexibility of the mid layers varies in support
threads / gpu ??
*/

use std::fs::File;
use std::io::{Write, Read};
use std::io;
use std::ops::BitAnd;
use std::process::Output;
use rand::{rngs::ThreadRng, Error, Rng};
use std::f32::consts::E;
use bytes::{Bytes, BytesMut, Buf, BufMut};

fn main() {
    println!("neural net starting");

    //let mut n1: Network = Network::new(784,2,16, 10);
    //n1.save("./1.bin");
    let mut n1: Network = Network::import("./1.bin");

    let input: Vec<f32> = (0..=784).map(|x| x as f32).collect();

    println!("{:?}", n1.exec(&input));


}

pub struct Network {
    input_size: usize, // 784, or a 28x28 image
    input_weights: Vec<Vec<f32>>, // 16 * 784, each input maps to each hidden layer
    hidden_layers: Vec<Vec<f32>>, // 2 * 16, 2 hidden layers, size of 16 each
    hidden_layer_count: usize, // 2
    hidden_layer_size: usize, // 16
    hidden_layer_weights: Vec<Vec<Vec<f32>>>,  // 1 , 16, 16
    output_layer_weights: Vec<Vec<f32>>, // 16, 10
    output: Vec<f32>,
    output_size: usize,
    biases: Vec<Vec<f32>>,
}

impl Network {
    pub fn new(input_size: usize, hidden_layer_count: usize, hidden_layer_size: usize, output_size: usize) -> Self {
        let mut rng: ThreadRng = rand::thread_rng();

        // maps inputs -> hidden layer 1 with 784*16 weights
        let input_weights: Vec<Vec<f32>> = (0..input_size).map(|_| (0..hidden_layer_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        //let input_weights: Vec<Vec<f32>> = vec![vec![0.5; hidden_layer_size]; input_size];

        let hidden_layers: Vec<Vec<f32>> = vec![vec![0.0; hidden_layer_size]; hidden_layer_count];

        let hidden_layer_weights: Vec<Vec<Vec<f32>>> = (0..hidden_layer_count-1).map(|_| (0..hidden_layer_size).map(|_| (0..hidden_layer_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect()).collect();

        let output_layer_weights: Vec<Vec<f32>> = (0..hidden_layer_size).map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();

        let output: Vec<f32> = vec![0.0; output_size];

        
        let mut biases: Vec<Vec<f32>> = vec![Vec::new(); 2+(hidden_layer_count-1)];
        for i in 0..hidden_layer_count {
            biases[i] = (0..hidden_layer_size).map(|_| rng.gen_range(-10.0..10.0)).collect();
        }

        //this is really annoying
        let length = biases.len()-1;
        biases[length] = (0..hidden_layer_size).map(|_| rng.gen_range(-10.0..10.0)).collect();;


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
            biases,
        }
    }

    // imports a saved network from a file
    pub fn import(filename: &str) -> Self {

        let mut file: File = File::open(filename).expect(std::format!("file {} failed to open", filename).as_str());

        let mut sizes = [0; 1 + 7*4];
        file.read_exact(&mut sizes);

        let mut buff: BytesMut = BytesMut::new();
        buff.put(&sizes[..]);

        let version: u8 = buff.get_u8();
        let input_size: usize = buff.get_u32_le() as usize;
        let hidden_layer_count: usize = buff.get_u32_le() as usize;
        let hidden_layer_size: usize = buff.get_u32_le() as usize;
        let output_size: usize = buff.get_u32_le() as usize; 
        let input_weight_size: usize = buff.get_u32_le() as usize;
        let hidden_layer_weight_size: usize = buff.get_u32_le() as usize;
        let output_weight_size: usize = buff.get_u32_le() as usize;

        println!("version: {} input_size: {} hidden_layer_count: {} hidden_layer_size: {} output_size: {} input_weight_size: {} hidden_layer_weight: {} output_weight_size: {}", version, input_size, hidden_layer_count, hidden_layer_size, output_size, input_weight_size, hidden_layer_count, output_weight_size);  


        let mut input_weights: Vec<Vec<f32>> = vec![vec![0.0; hidden_layer_size]; input_size];
        let hidden_layers: Vec<Vec<f32>> = vec![vec![0.0; hidden_layer_size]; hidden_layer_count];
        let mut hidden_layer_weights: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; hidden_layer_size]; hidden_layer_size]; hidden_layer_count];
        let mut output_layer_weights: Vec<Vec<f32>> = vec![vec![0.0; output_size]; hidden_layer_size];
        let output: Vec<f32> = vec![0.0; output_size];
        let mut biases: Vec<Vec<f32>> = vec![Vec::new(); 2+(hidden_layer_count-1)];

        let input_weight_size: usize = input_size*hidden_layer_size;
        let hidden_layer_weight_size: usize = (hidden_layer_count-1)*hidden_layer_size*hidden_layer_size;
        let output_weight_size: usize = hidden_layer_size*output_size;
        let biases_size: usize = (hidden_layer_count)*hidden_layer_size + output_size;
        let capacity: usize = ( input_weight_size +  hidden_layer_weight_size + output_weight_size + biases_size) * 4;

        let mut data = vec![0; capacity];
        file.read_exact(&mut data);

        let mut buff: BytesMut = BytesMut::new();
        buff.put(&data[..]);

        for i in 0..input_size {
            for j in 0..hidden_layer_size {
                input_weights[i][j] = buff.get_f32_le();
            }
        }
        for i in 0..(hidden_layer_count-1) {
            for j in 0..hidden_layer_size {
                for k in 0..hidden_layer_size {
                hidden_layer_weights[i][j][k] = buff.get_f32_le();
                }
            }
        }   


        for i in 0..hidden_layer_size {
            for j in 0..output_size {
               output_layer_weights[i][j] = buff.get_f32_le();
            }
        }

        
        for i in 0..hidden_layer_count {
            for j in 0..hidden_layer_size{
                biases[i].push(buff.get_f32_le());
            }
        }

        for i in 0..output_size {
            biases[hidden_layer_count].push(buff.get_f32_le());
        }
        

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
            biases
        }
    }
    

    fn reset_hidden_layers(&mut self){
        // Reset hidden layers to zero
        for layer in &mut self.hidden_layers {
            layer.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn sigmoid(&mut self, number: f32) -> f32 {
        1.0 / ( 1.0 + E.powf( -1.0 * number))
    }

    pub fn exec(&mut self, input: &Vec<f32>) -> &Vec<f32> {

        // input -> layer 1
        for i in 0..self.input_size {
            for j in 0..self.hidden_layer_size {
                self.hidden_layers[0][j] += input[i] * self.input_weights[i][j];
            }
        }

        for i in 0..self.hidden_layer_size {
            self.hidden_layers[0][i] =  self.sigmoid(self.hidden_layers[0][i]);
        }

         // layer 1 -> layer 2
         for i in 0..self.hidden_layer_size {
            for j in 0..self.hidden_layer_size {
                self.hidden_layers[1][j] += self.hidden_layers[0][j] * self.hidden_layer_weights[0][i][j];
            }
        }

        for i in 0..self.hidden_layer_size {
            self.hidden_layers[1][i] =  self.sigmoid(self.hidden_layers[1][i]);
        }

        // layer 2 -> output
        for i in 0..self.hidden_layer_size {
            for j in 0..self.output_size {
                self.output[j] += self.hidden_layers[1][j] * self.output_layer_weights[i][j];
            }
        }

        for i in 0..self.output_size {
            self.output[i] =  self.sigmoid(self.output[i]);
        }


        &self.output
    }

    /*
    Byte structure
    1 bytes: Version
    4 bytes: input_size
    4 bytes: hidden_layer_count
    4 bytes: hidden_layer_size
    4 bytes: output size
    4 bytes: input_weight_size 
    4 bytes: hidden_layer_weight_size 
    4 bytes: output_weight_size
    x bytes: input_weights
    x bytes: hidden_weights
    x bytes: output_weights
    x bytes: biases

    multiply * 4 for f32
     */
    fn as_bytes(&mut self) -> Result<BytesMut, io::Error>{
        let input_weight_size: usize = self.input_size*self.hidden_layer_size;
        let hidden_layer_weight_size: usize = (self.hidden_layer_count-1)*self.hidden_layer_size*self.hidden_layer_size;
        let output_weight_size: usize = self.hidden_layer_size*self.output_size;
        let biases_size: usize = (self.hidden_layer_count)*self.hidden_layer_size + self.output_size;
        let capacity: usize = 1 + 7*4 + ( input_weight_size +  hidden_layer_weight_size + output_weight_size + biases_size) * 4;

        let mut output: BytesMut = BytesMut::with_capacity(capacity);

        //trying to test this leads to the dark side aka insanity .. it makes no sense! why is 12544 \00\01\00\00????? and not \00\31\00\00?????
        output.put_u8(1);
        output.put_u32_le(self.input_size as u32);
        output.put_u32_le(self.hidden_layer_count as u32);
        output.put_u32_le(self.hidden_layer_size as u32);
        output.put_u32_le(self.output_size as u32);
        output.put_u32_le(input_weight_size as u32);
        output.put_u32_le(hidden_layer_weight_size as u32);
        output.put_u32_le(output_weight_size as u32);

        for i in 0..self.input_size {
            for j in 0..self.hidden_layer_size {
                output.put_f32_le(self.input_weights[i][j]);
            }
        }

        for i in 0..(self.hidden_layer_count-1) {
            for j in 0..self.hidden_layer_size {
                for k in 0..self.hidden_layer_size {
                    output.put_f32_le(self.hidden_layer_weights[i][j][k]);
                }
            }
        } 

        for i in 0..self.hidden_layer_size {
            for j in 0..self.output_size {
                output.put_f32_le(self.output_layer_weights[i][j]);
            }
        }

        for i in 0..(2+(self.hidden_layer_count-1)) {
            for j in 0..self.biases[i].len(){
                output.put_f32_le(self.biases[i][j]); 
            }
        }

        Ok(output)
    }

    pub fn save(&mut self, filename: &str) -> Result<(), io::Error> {
        // open file
        let mut file: File = File::create(filename)?;
        
        // data
        let data: BytesMut  = self.as_bytes()
            .expect("Error formatting data.");

        // write data
        file.write_all(&data)?;

        Ok(())
    }
}
