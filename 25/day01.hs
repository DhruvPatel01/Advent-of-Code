module Day01 where

import qualified Data.List as L

input = map parse . lines <$> readFile "./inputs/day01.txt"

parse :: String -> Int
parse ('L':n) = - (read n)
parse ('R':n) = read n



part1 :: [Int] -> Int
part1 input = length . filter (==0) $ applied
    where  applied = map (`mod` 100) $ L.scanl' (+) 50 input

-- part2 :: [Int] -> Int
part2 input = arounds + final
    where
        arounds = sum . map ((`div` 100) . abs) $ input -- count how many times do i go around end-up at the same position
        leftovers = map (`rem` 100) input
        positions = map (`mod` 100) $ L.scanl' (+) 50 input  -- positions of the dial after each step
        non_zero_moves = filter ((/=0) . fst ) $ zip positions leftovers -- removing zeros to avoid double counting
        final = length . filter (\x -> x <= 0 || x >= 100) . map (uncurry (+)) $ non_zero_moves


main = do
    inp <- input
    print $ part1 inp
    print $ part2 inp