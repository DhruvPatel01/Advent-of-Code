use std::io;
use std::collections::{HashMap, HashSet};

fn is_small_cave(cave: &str) -> bool {
    return cave.chars().all(|c| c.is_ascii_lowercase());
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    let mut graph = HashMap::new();
    let mut map = HashMap::new();
    map.insert("end", 0);

    let mut big_idx: i32 = 1;
    let mut sml_idx: i32 = -1;
    let tmp = Vec::with_capacity(2);
    loop {
        if let Ok(0) = sin.read_line(&mut s) {break;};
       
        for (i, u) in s.trim_end().split("-").enumerate() {
            let u = String::from(u);
            if !map.contains_key(&u as &str) {
                if is_small_cave(&u) {
                    map.insert(&u, sml_idx);
                    sml_idx -= 1;
                } else {
                    map.insert(&u, big_idx);
                    big_idx += 1;
                }
            }
            tmp[i] = map.get(&u as &str).unwrap();
        }
        graph.entry(tmp[0]).or_insert(Vec::new()).push(tmp[1]);
        graph.entry(tmp[1]).or_insert(Vec::new()).push(tmp[0]);

        s.clear();
    }
    println!("{:#?}", map);
    println!("{:#?}", graph);
}