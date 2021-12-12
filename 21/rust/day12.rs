use std::io;
use std::collections::{HashMap, HashSet};

fn is_small_cave(cave: &str) -> bool {
    return cave.chars().all(|c| c.is_ascii_lowercase());
}

fn dfs1<'a>(u: &'a str, graph: &'a HashMap<String, Vec<String>>, 
        stack: &mut HashSet<&'a str>) -> u32{
    if u == "end" {
        return 1;
    } else if stack.contains(u) {
        return 0;
    } else if is_small_cave(u) {
        stack.insert(u);
    }
    let mut paths = 0;
    for nbr in graph.get(u).unwrap() {
        paths += dfs1(nbr, graph, stack);
    }
    stack.remove(u);
    return paths;
}

fn dfs2<'a>(u: &'a str, graph: &'a HashMap<String, Vec<String>>, 
        stack: &mut HashSet<&'a str>, double: Option<&'a str>) -> u32{
    let mut double = double;
    if u == "end" {
        return 1;
    } else if is_small_cave(u) {
        if stack.contains(u) {
            if let Some(_) = double {
                return 0;
            }
            double = Some(u);
        }
        stack.insert(u);
    }

    let mut paths = 0;
    for nbr in graph.get(u).unwrap() {
        if nbr != "start" {
            paths += dfs2(nbr, graph, stack, double);
        }
    }
    stack.remove(u);
    if let Some(v) = double {  stack.insert(v); }
    return paths;
}

fn main() {
    let mut s = String::new();
    let sin = io::stdin();
    let mut graph = HashMap::new();
    loop {
        if let Ok(0) = sin.read_line(&mut s) {break;};
        let mut line = s.trim_end().split("-");
        let u = String::from(line.next().unwrap());
        let v = String::from(line.next().unwrap());

        let nu = graph.entry(u.clone()).or_insert(Vec::new());
        nu.push(v.clone());

        let nv = graph.entry(v).or_insert(Vec::new());
        nv.push(u);

        s.clear();
    }

    let mut s = HashSet::new();
    println!("Part1: {}", dfs1("start", &graph, &mut s));
    println!("Part2: {}", dfs2("start", &graph, &mut s, None));
    
}