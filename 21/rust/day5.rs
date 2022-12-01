//unrefined code.

use std::io;

fn swap(x: &mut usize, y: &mut usize) {
    let tmp = *x;
    *x = *y;
    *y = tmp;
}

fn read_pt(end_points: &str) -> (usize, usize) {
    let mut point = end_points.split(',');
    let x: usize = point.next().unwrap().trim().parse().unwrap();
    let y: usize = point.next().unwrap().trim().parse().unwrap();
    (x, y)
}

fn max(x: usize, x1: usize, x2: usize) -> usize {
    if x1 >= x2 { if x1 > x {x1} else {x} }
    else {if x2 > x {x2} else {x}}
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();

    let mut lines = Vec::new();
    let mut max_x = 0;
    let mut max_y = 0;

    loop {
        if let Ok(0) = sin.read_line(&mut s) {break;}
        let mut end_points = s.split("->");
        let (mut x1, mut y1) = read_pt(end_points.next().unwrap());
        let (mut x2, mut y2) = read_pt(end_points.next().unwrap());

        max_x = max(max_x, x1, x2);
        max_y = max(max_y, y1, y2);

        if x1 > x2 || y1 > y2 {
            swap(&mut x1, &mut x2);
            swap(&mut y1, &mut y2);
        }

        lines.push(((x1, y1), (x2, y2)));
        s.clear();
    }
    
    let mut matrix_part1 = Vec::new();
    let mut matrix_part2 = Vec::new();

    for _ in 0..=max_y {
        matrix_part1.push(vec![0; max_x+1]);
        matrix_part2.push(vec![0; max_x+1]);
    }

    for ((x1, y1), (x2, y2)) in lines {
        if x1 == x2 {
            for i in y1..=y2 {
                matrix_part1[i][x1] += 1;
                matrix_part2[i][x1] += 1;
            }
        } else if y1 == y2 {
            for i in x1..=x2 {
                matrix_part1[y1][i] += 1;
                matrix_part2[y1][i] += 1;
            }
        } else {
            let (x1, y1) = (x1 as i32, y1 as i32);
            let (x2, y2) = (x2 as i32, y2 as i32);
            let dir_x = if x1 > x2 {-1} else {1};
            let dir_y = if y1 > y2 {-1} else {1};
            for i in 0..=(x2-x1).abs() {
                matrix_part2[(y1+dir_y*i) as usize][(x1+dir_x*i) as usize] += 1;
            }
        }
    }

    let mut ans1 = 0;
    let mut ans2 = 0;
    for i in 0..=max_y {
        for j in 0..=max_x {
            if matrix_part1[i][j] > 1 {ans1 += 1;}
            if matrix_part2[i][j] > 1 {ans2 += 1;}
        }
    }

    println!("Part1: {}", ans1);
    println!("Part2: {}", ans2);
}