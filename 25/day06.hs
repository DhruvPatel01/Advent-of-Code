module Day06 where

import qualified Data.List as L

input =  lines <$> readFile "./inputs/day06.txt"

applyOp ("+":xs) = sum (map read xs)
applyOp ("*":xs) = product (map read xs)

part1 = sum . map applyOp

groups :: [String] -> [String] -> [[String]]
groups grp [] = [grp]
groups grp (s:rem) | all (==' ') s = grp : groups [ ] rem
groups grp (s: rem) = groups (s : grp)  rem

part2 input = sum $ map applyOp combined_input
    where 
        ops = words (last input)
        args = groups [] $ L.transpose (init input)
        combined_input = [o:a | (o,a) <- zip ops args]


main = do
    inp <- input
    print $ part1 $ (L.transpose . map words . reverse) inp
    print $ part2 inp
