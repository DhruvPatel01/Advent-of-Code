module Day04 where

import qualified Data.Map.Strict as M
import qualified Data.Set as S
import Data.Maybe (fromMaybe)

input = M.fromList . withCoords . lines <$> readFile "./inputs/day04.txt"


withCoords :: [String] -> [((Int,Int), Char)]
withCoords ls = [((r,c), cell) | (r, line) <- zip [0..] ls, (c, cell) <- zip [0..] line]
 
gridValueAt (r, c) = fromMaybe '.' . M.lookup (r, c)

neighboursIdxs (r, c) = [(r+dr, c+dc) | dr <- [-1,0,1], dc <- [-1,0,1], abs dr +abs dc > 0]
neighbours grid rc = map (`gridValueAt` grid)  (neighboursIdxs rc)

part1 inp =  length . filter (<4) $ num_neighbours
    where 
        num_neighbours =  [length . filter (=='@') $ neighbours inp k | (k, a) <- M.assocs inp, a == '@']

removablePositions inp = 
    [
    k | (k, a) <- M.assocs inp, 
    a == '@', 
    length (filter (=='@') $ neighbours inp k) < 4
    ]

countRolls  = length . filter (=='@') . M.elems 

part2 inp = countRolls . snd . last . takeWhile (uncurry (/=)) $ zip afterRemoval (drop 1 afterRemoval) 
    where afterRemoval = iterate (\inp -> M.withoutKeys inp (S.fromList . removablePositions  $ inp)) inp

main = do
    inp <- input
    print $ part1 inp
    print $ countRolls inp - part2 inp