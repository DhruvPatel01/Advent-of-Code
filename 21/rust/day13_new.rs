use std::io;
use std::collections::HashSet;
use std::iter::FromIterator;

fn print_grid(points: &HashSet<(i32, i32)>) {
    let max_x = points.iter().fold(0, |a, (x, _)| a.max(*x));
    let max_y = points.iter().fold(0, |a, (_, y)| a.max(*y));
    let max_x = max_x as usize;
    let max_y = max_y as usize;
    let mut matrix = vec![vec![false; max_x+1]; max_y+1];
    for (i, j) in points { matrix[*j as usize][*i as usize] = true; }

    for j in 0..=max_y {
        for i in 0..=max_x {
            print!("{}", if matrix[j][i] {"#"} else {" "});
        }
        println!("");
    }
}

fn fold(points: HashSet<(i32, i32)>, axis: u8, line: i32) -> HashSet<(i32, i32)>
{
    if axis == 'x' as u8 {
        return HashSet::from_iter(
            points.into_iter().map(
                |(x, y)| if x <= line {(x, y)} else {(2*line-x, y)}));
    } else {
        return HashSet::from_iter(
            points.into_iter().map(
                |(x, y)| if y <= line {(x, y)} else {(x, 2*line-y)}));
    }
}

fn main() 
{
    let mut s = String::new();
    let sin = io::stdin();
    let mut dots = HashSet::new();
    loop {
        if let Ok(1) = sin.read_line(&mut s) {break;}
        let mut it = s.trim_end().split(",");
        let x: i32 = it.next().unwrap().parse().unwrap();
        let y: i32 = it.next().unwrap().parse().unwrap();
        dots.insert((x, y));
        s.clear();
    }

    for i in 0.. {
        if let Ok(0) = sin.read_line(&mut s) {break;}
        let mut it = s.trim_end().split("=");
        let command = it.next().unwrap().bytes().last().unwrap();
        let arg: i32 = it.next().unwrap().parse().unwrap();
        dots = fold(dots, command, arg);
        if i == 0 { println!("Part1: {}", dots.len()); }
        s.clear();
    }
    print_grid(&dots);
}