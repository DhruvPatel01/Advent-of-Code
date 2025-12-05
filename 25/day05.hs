module Day04 where

import qualified Data.List as L

type Range = (Int, Int)

input =  parse . lines <$> readFile "./inputs/day05.txt"

parse lines = (merged_ranges, map read (drop 1 ingredients) :: [Int])
    where (raw_ranges, ingredients) = break null lines
          parse_range word = (read a::Int, read (drop 1 b)::Int)
            where (a, b) = break (=='-') word
          sorted_ranges = L.sort $ map parse_range raw_ranges
          merged_ranges = L.sort $  L.foldl' maybeMerge [] sorted_ranges


maybeMerge [] r = [r]
maybeMerge old@((a,b):rem) r@(c, d)
    | b < c = r:old
    | d <= b = old
    | otherwise = (a, d): rem

inRange n (a,b) = a <= n && n <= b

part1 ranges ingredients = length . filter id $ [any (inRange x) ranges | x <- ingredients]

part2 ranges = sum $ map (\(x, y) -> y-x+1) ranges

main = do
    (ranges, ingredients) <- input
    print $ part1 ranges ingredients
    print $ part2 ranges