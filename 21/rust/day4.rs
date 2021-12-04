use std::io;
use std::collections::{HashMap, HashSet};

struct Board {
    rows: Vec<u8>,
    cols: Vec<u8>,
    sum: u32,
}

impl Board {
    fn new(sum: u32) -> Board {
        Board {
            rows: vec![0; 5],
            cols: vec![0; 5],
            sum: sum,
        }
    }
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    sin.read_line(&mut s).unwrap();

    let to_x:Vec<u32> = s.trim_end().split(',').map(|x| x.parse().unwrap()).collect();
    let mut map = HashMap::new();
    for x in &to_x {
        map.insert(x, Vec::new());
    }
    let mut boards = Vec::new();
    let mut remaining = HashSet::new();
    let mut k = 0;

    loop {
        if let Ok(0) = sin.read_line(&mut s) {break;}
    
        let mut sum = 0;
        for i in 0..5 {
            s.clear();
            sin.read_line(&mut s).unwrap();
            for (j, x) in s.split_ascii_whitespace().enumerate() {
                let x = x.parse::<u32>().unwrap();
                sum += x;
                if let Some(entry) = map.get_mut(&x) {
                    entry.push((k, i, j));
                }
            }
        }
        boards.push(Board::new(sum));
        remaining.insert(k);
        k += 1;
    }   

    let n = k;
    
    'outer: 
    for x in &to_x {
        for (k, i, j) in map.get(&x).unwrap() {
            let board = &mut boards[*k];
            board.sum -= x;
            board.rows[*i] += 1;
            board.cols[*j] += 1;
            if (board.rows[*i] == 5) || (board.cols[*j] == 5) {
                remaining.remove(k);
                if remaining.len() == n - 1 {
                    println!("Part1: {}", board.sum * x);
                } else if remaining.len() == 0 {
                    println!("Part2: {}", board.sum * x);
                    break 'outer;
                }
                
            }
        }
    }
}