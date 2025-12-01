import requests
import datetime
import sys
import os

USER_AGENT = "github.com/DhruvPatel01 by dhruv.nanosoft@gmail.com"

def main():
    if not os.path.exists('./session_cookie.txt'):
        print("session_cookie.txt doesn't exist. Please save session cookie into it!")
        sys.exit(1)

    with open("./session_cookie.txt") as f:
        cookie = f.read().strip()
        
    now = datetime.datetime.now()
    year = str(now.year)
    day = str(now.day)

    if len(sys.argv) == 3:
        year = sys.argv[1]
        day = sys.argv[2]
    elif len(sys.argv) == 2:
        day = sys.argv[1]
    day = int(day)

    target_file = f"./{year[-2:]}/inputs/day{day:02}.txt"
    print(f"Downloading file: {target_file}")
    headers = {"user-agent": "github.com/DhruvPatel01/advent-of-code by dhruv.nanosoft@gmail.com"}
    cookies = {"session": cookie}
    content = requests.get(f"https://adventofcode.com/{year}/day/{day}/input", headers=headers, cookies=cookies)

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w") as f:
        f.write(content.text)
    
if __name__ == '__main__':
    main()
