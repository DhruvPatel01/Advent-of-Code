use std::io::{self, Read};

fn main() {
    let mut inp = String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let mut inp = inp.split_whitespace();

    let mut v = Vec::new();

    while let Some(command) = inp.next() {
        let value: i32 = inp.next().unwrap().parse().unwrap();
        match command {
            "up" => v.push((0, -value)),
            "down" => v.push((0, value)),
            "forward" => v.push((value, 0)),
            _ => ()
        }
    }

    let (a, b) = v.iter().fold((0, 0), |(x, y), (a, b)| (x+a, y+b));
    println!("Part1: {}", a*b);

    let (_, x, y) = v.iter().fold((0, 0, 0), 
        |(aim, x1, y1), (x2, y2)| (aim+y2, x1+x2, y1+aim*x2)
    );
    println!("Part2: {}", x*y);
}