use std::io;

type Grid<T> = Vec<Vec<T>>;

fn part1(matrix: &Grid<u8>, i: usize, j: usize) -> bool {
    let elem = matrix[i][j];
    if i > 0 && matrix[i-1][j] <= elem { return false;}
    if i < matrix.len()-1 && matrix[i+1][j] <= elem { return false;}
    if j > 0 && matrix[i][j-1] <= elem { return false;}
    if j < matrix[0].len()-1 && matrix[i][j+1] <= elem { return false;}
    true
}

fn dfs(matrix: &Grid<u8>, i: usize, j: usize, visited: &mut Grid<bool>) -> u32 {
    if matrix[i][j] == 9 || visited[i][j]{
        return 0;
    }
    visited[i][j] = true;
    let mut size = 1;
    if i > 0 && !visited[i-1][j] { 
        size += dfs(matrix, i-1, j, visited);
    }
    if i < matrix.len()-1 && !visited[i+1][j] {
        size += dfs(matrix, i+1, j, visited);
    }
    if j > 0 && !visited[i][j-1] {
        size += dfs(matrix, i, j-1, visited);
    }
    if j < matrix[0].len()-1 && !visited[i][j+1] {
        size += dfs(matrix, i, j+1, visited);
    }

    return size;
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    
    let mut matrix: Grid<u8> = Vec::new();
    let mut visited: Grid<bool> = Vec::new();
    loop {
        if let Ok(0) = sin.read_line(&mut s) { break; }
        matrix.push(s.trim_end().bytes().map(|x| x - '0' as u8).collect());
        visited.push(vec![false; matrix[0].len()]);
        s.clear();
    }
    
    let mut sizes = Vec::new();
    let mut part1_sol = 0;
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            if !visited[i][j] {
                let size = dfs(&matrix, i, j, &mut visited);
                if size > 0 {sizes.push(size)}
            }
            if part1(&matrix, i, j) {
                part1_sol += 1 + matrix[i][j] as u32
            }
        }
    }
    sizes.sort_by(|a, b| b.cmp(a));
    println!("Part1: {}", part1_sol);
    println!("Part2: {}", sizes[0]*sizes[1]*sizes[2]);
}