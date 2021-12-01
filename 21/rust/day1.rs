use std::io;

fn main() {
    let mut s = String::new();
    let sin = io::stdin();

    let mut prev = 0;
    let mut ans = -1;
    loop {
        s.clear();
        match sin.read_line(&mut s) {
            Ok(0) => {break;}
            Ok(_) => {
                let next: u32 = s.trim_end().parse().unwrap();
                if next > prev {
                    ans += 1;
                }
                prev = next;
            }, 
            _ => {break;}
        }
    }
    println!("{}", ans);
}