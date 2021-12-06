use std::io; 
use std::collections::HashMap;

type Cache = HashMap<(u32, u32), u64>;

fn g(n: u32, m: u32, cache_g: &mut Cache, cache_h: &mut Cache ) -> u64 {
    if let Some(&x) = cache_g.get(&(n, m)) {
        x
    } else if n < m+9 {
        1
    } else {
        let x = g(n, m+9, cache_g, cache_h) + h(n, m+16, cache_g,  cache_h);
        cache_g.insert((n, m), x);
        x
    }
}

fn h(n: u32, m: u32, cache_g: &mut Cache, cache_h: &mut Cache) -> u64 {
    if let Some(&x) = cache_h.get(&(n, m)) {
        x
    } else if n < m {
        1
    } else {
        let x = g(n, m, cache_g, cache_h) + h(n, m+7, cache_g,  cache_h);
        cache_h.insert((n, m), x);
        x
    }
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    sin.read_line(&mut s).unwrap();

    let x: Vec<u32> = s.trim_end().split(',').map(|x| x.parse().unwrap()).collect();
    let mut cache_g: Cache = HashMap::new();
    let mut cache_h: Cache = HashMap::new();

    let mut ans = 0;
    for elem in &x {
        ans += h(80, elem+1, &mut cache_g, &mut cache_h);
    }
    println!("Part1: {}", ans);

    ans = 0;
    for elem in &x {
        ans += h(256, elem+1, &mut cache_g, &mut cache_h);
    }
    println!("Part2: {}", ans);
}