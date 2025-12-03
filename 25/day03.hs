module Day03 where

import qualified Data.List as L
import Data.Char (digitToInt)

input = map (map digitToInt) . lines <$> readFile  "./inputs/day03.txt"

stub :: (Ord a, Num a) => a -> a -> [a] -> [a] -> [a]
stub target_l n max_so_far singles | n == target_l = max_so_far
stub target_l n max_so_far singles = stub target_l (n+1) new_maxes (tail singles)
    where 
        maxes = L.scanl1 max max_so_far -- for each position find the maximum prefix before that position
        new_maxes = zipWith (\x y -> x*10 + y) maxes singles -- form a new maximum number that ends at current position

findMaxNum target_l inp = maximum $ stub target_l 0 (replicate (length inp) 0) inp

part1 :: [[Int]] -> Int
part1 = sum. map (findMaxNum 2)

part2 :: [[Int]] -> Int
part2 = sum. map (findMaxNum 12)

main = do
    inp <- input
    print $ part1 inp
    print $ part2 inp