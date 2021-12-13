use std::io;

fn print_grid(matrix: &Vec<Vec<bool>>,
    min_i: usize, min_j: usize, 
    max_i: usize, max_j: usize)
{
    for i in min_i..=max_i {
        for j in  min_j..=max_j {
            let x = if matrix[i][j] {"#"} else {" "};
            print!("{}", x);
        }
        println!("");
    }
}

fn n_dots(matrix: &Vec<Vec<bool>>,
    min_i: usize, min_j: usize, 
    max_i: usize, max_j: usize) -> u32
{
    let mut ans = 0;
    for i in min_i..=max_i {
        for j in  min_j..=max_j {
            ans += if matrix[i][j] {1} else {0};
        }
    }
    return ans;
}

fn fold_along_x(matrix: &mut Vec<Vec<bool>>, 
    min_i: usize, min_j: usize, 
    max_i: usize, max_j: usize, x: usize) -> (usize, usize)
{
    let mid = (min_j + max_j)/2;
    if x < mid {
        for i in min_i..=max_i {
            for j in 1..=(x-min_j) {
                matrix[i][x+j] |= matrix[i][x-j];
            }
        }
        return (x+1, max_j);
    } else {
        for i in min_i..=max_i {
            for j in 1..=(max_j-x) {
                matrix[i][x-j] |= matrix[i][x+j];
            }
        }
        return (min_i, x-1);
    }
}

fn fold_along_y(matrix: &mut Vec<Vec<bool>>, 
    min_i: usize, min_j: usize, 
    max_i: usize, max_j: usize, y: usize) -> (usize, usize)
{
    let mid = (min_i + max_i)/2;
    if y < mid {
        for j in min_j..=max_j {
            for i in 1..=(y-min_i) {
                matrix[y+i][j] |= matrix[y-i][j];
            }
        }
        return (y+1, max_i);
    } else {
        for j in min_j..=max_j {
            for i in 1..=(max_i-y) {
                matrix[y-i][j] |= matrix[y+i][j];
            }
        }
        return (min_j, y-1);
    }
}

fn main() 
{
    let mut s = String::new();
    let sin = io::stdin();
    let mut max_x = 0;
    let mut max_y = 0;
    let mut dots = Vec::new();
    loop {
        if let Ok(1) = sin.read_line(&mut s) {break;}
        let mut it = s.trim_end().split(",");
        let x: usize = it.next().unwrap().parse().unwrap();
        let y: usize = it.next().unwrap().parse().unwrap();
        if x > max_x {max_x = x;}
        if y > max_y {max_y = y;}
        dots.push((y, x));
        s.clear();
    }

    let mut matrix = vec![vec![false; max_x+1]; max_y+1];
    for (i, j) in dots { matrix[i][j] = true; }

    let mut min_i = 0;
    let mut min_j = 0;
    let mut max_i = max_y;
    let mut max_j = max_x;
    
    for i in 1.. {
        if let Ok(0) = sin.read_line(&mut s) {break;}
        let mut it = s.trim_end().split("=");
        let command = it.next().unwrap().bytes().last().unwrap();
        let arg: usize = it.next().unwrap().parse().unwrap();
        if command == 'y' as u8 {
            let (a, b) = fold_along_y(&mut matrix, min_i, min_j, max_i, max_j, arg);
            min_i = a;
            max_i = b;
        } else {
            let (a, b) = fold_along_x(&mut matrix, min_i, min_j, max_i, max_j, arg);
            min_j = a;
            max_j = b;
        }
        if i == 1 {
            println!("Part1: {}", n_dots(&matrix, min_i, min_j, max_i, max_j));
        }
        s.clear();
    }
    print_grid(&matrix, min_i, min_j, max_i, max_j);
}