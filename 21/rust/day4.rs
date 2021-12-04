use std::io;
use std::collections::HashMap;

#[derive(Debug)]
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
        map.insert(x, HashMap::new());
    }
    let mut boards = Vec::new();
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
                    entry.insert(k, (i, j));
                }
            }
        }
        boards.push(Board::new(sum));
        k += 1;
    }   
    
    let mut to_x = to_x.iter();
    let mut removed = 100_000;

    loop {
        let x = to_x.next().unwrap();
        for (k, (i, j)) in map.get(&x).unwrap() {
            let board = &mut boards[*k];
            board.sum -= x;
            board.rows[*i] += 1;
            board.cols[*j] += 1;
            if (board.rows[*i] == 5) || (board.cols[*j] == 5) {
                removed = *k;
                println!("Part1: {}", board.sum * x);
            }
        }
        if removed != 100_000 {
            break;
        }
    }

    let mut remaining = Vec::new();
    for i in 0..boards.len() {
        if i != removed {
            remaining.push(i);
        }
    }

    for x in to_x {
        let mut new = Vec::new();
        let tmp_map = map.get(&x).unwrap();
        for k in &remaining {            
            if let Some((i, j)) = tmp_map.get(&k) {
                let board = &mut boards[*k];
                board.sum -= x;
                board.rows[*i] += 1;
                board.cols[*j] += 1;
                if (board.rows[*i] == 5) || (board.cols[*j] == 5) {
                    removed = *k;
                } else {
                    new.push(*k);
                }
            } else {
                new.push(*k);
            }
        }
        if new.len() == 0 {
            let board = &boards[removed];
            println!("Part2: {} with score {}", removed, x*board.sum);
            break;
        } else {
            remaining = new;
        }
    }
}