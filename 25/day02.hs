module Day2 where

import qualified Data.List as L
import Data.List.Split (splitOneOf)


asTuples (a:b:cs) = (a, b) : asTuples cs
asTuples _ = []

input :: IO [(Int, Int)]
input = asTuples . map read. splitOneOf ",-"  <$> readFile "./inputs/day02.txt"

part1_single :: (Int, Int) -> [Int]
part1_single (start, stop) = filter (>= start) . takeWhile (<= stop) $ map ( read . (\x -> x ++ x) .  show)  [prefix..]
    where
        start_str = show start
        n = length start_str  `div` 2
        prefix::Int= if n > 0 then read (take n start_str) else 1

part1 :: [(Int, Int)] -> Int
part1 = sum . concatMap part1_single

curious ((x,y):rs) = (y-x) + curious rs
curious [] = 0

isRep' :: String ->Int -> Bool -- returns if n is made of repeated prefixes
isRep' n p | length n `mod` p /= 0 = False
isRep' n p | p >= length n = False
isRep' n p = isMadeOf (take p n) (drop p n)


isMadeOf prefix [] = True
isMadeOf prefix num = (n >= p) && (prefix == hd) && isMadeOf prefix tl
    where n = length num
          p = length prefix
          (hd, tl) = L.splitAt p num

isRep n = any (isRep' n) [1..length n `div` 2]

part2 :: [(Int, Int)] -> Int
part2 = sum . concatMap (map read . filter isRep . map show . \(x,y) -> [x..y])
        

main = do
    inp <- input
    print $ part1 inp
    print $ part2 inp