use std::io;
use std::collections::VecDeque;

type Grid = Vec<Vec<u32>>;

fn step(grid: &mut Grid) -> u32 {
    let mut q = VecDeque::new();
    let mut flashed = 0;
    
    for i in 1..=10 {
        for j in 1..=10 {
            grid[i][j] += 1;
            if grid[i][j] == 10 {
                q.push_back((i, j));
            }
        }
    }

    while let Some((i, j)) = q.pop_front() {
        for u in (i-1)..=(i+1) {
            for v in (j-1)..=(j+1) {
                grid[u][v] += 1;
                if grid[u][v] == 10 {
                    q.push_back((u, v));
                }
            }
        }
    }

    for i in 1..=10 {
        for j in 1..=10 {
            if grid[i][j] >= 10 {
                grid[i][j] = 0;
                flashed += 1;
            }
        }
    }

    return flashed;
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    
    let mut grid = Grid::with_capacity(12);
    grid.push(vec![100; 12]);
    loop {
        if let Ok(0) = sin.read_line(&mut s) {break;}
        let mut v = vec![100];
        v.extend(s.trim_end().bytes().map(|x| (x - '0' as u8) as u32));
        v.push(100);
        grid.push(v);
        s.clear();
    }
    grid.push(vec![100; 12]);

    let mut part1 = 0;
    for i in 0.. {
        let tmp = step(&mut grid);
        if i < 100 {
            part1 += tmp;
        } else if i == 100 {
            println!("Part1: {}", part1);
        }

        if tmp == 100 {
            println!("Part2: {}", i+1);
            break;
        }
    }
}