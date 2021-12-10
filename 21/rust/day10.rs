use std::io::{self, Read};

fn part2(b: u8) -> u32 {
    match &b {
        41 => 1,
        93 => 2,
        125 => 3,
        _ => 4
    }
}

fn solve(line: &str) -> (u32, u64) {
    let mut stack = Vec::new();
    for b in line.bytes() {
        match b {
            40 => stack.push(41),
            60 | 91 | 123 => stack.push(b+2),
            41 | 62 | 93 | 125 => {
                let popped = stack.pop().unwrap();
                if popped != b {
                    return (match b {
                        41 => 3,
                        93 => 57,
                        125 => 1197,
                        _ => 25137
                    }, 0);
                }
            },
            _ => (),
        }
    }
    let a2 = stack.into_iter().rev().map(part2).fold(0, |a, b| a*5 + b as u64);
    return (0, a2);
}

fn main() {
    let mut input = String::new();
    let mut sin = io::stdin();

    sin.read_to_string(&mut input).unwrap();
    let sol: Vec<(u32, u64)> = input.split_ascii_whitespace().map(solve).collect();
    let ans1: u32 = sol.iter().map(|(a, _)| a).sum();
    let mut ans2: Vec<u64> = sol.iter().map(|&(_, b)| b).filter(|&x| x > 0).collect();
    ans2.sort();

    println!("Part1: {}", ans1);
    println!("Part2: {}", ans2[ans2.len()/2]);
}