module Day08 where

import qualified Data.Map.Strict as M
import qualified Data.List as L
import Data.Maybe
import Data.List.Split (splitOn)

-- input

input = map parse . lines <$> readFile "./inputs/day08.txt"

parse :: String -> [Int]
parse = map read . splitOn "," 


-- union find related code

type Forest = M.Map Int Int
type ForestWithSize = (Forest, Int)

getSetId ::  Forest -> Int -> Int
getSetId forest k = if parent == k then k else getSetId forest parent 
    where parent =  fromJust $ M.lookup k forest

union::ForestWithSize -> (Int, Int) -> ForestWithSize
union (forest, s) (a, b) = if sa /= sb then (M.insert sa sb forest, s-1) else (forest, s)
    where
        sa = getSetId forest a
        sb = getSetId forest b


setSizes :: Forest -> [Int]
setSizes f =   reverse . L.sort . map length.  L.group . L.sort . map (getSetId f) $ M.keys f

---

type IndexPoint = (Int, [Int])
distanceBetween :: IndexPoint -> IndexPoint -> Int
distanceBetween (_, a) (_, b) = sum . map (^2) $ zipWith (-) a b

getPairs inp = map extract sorted
    where 
        indexed = zip [0..] inp :: [IndexPoint]
        sorted = L.sortOn (uncurry distanceBetween) [(x, y) | (x:ys) <- L.tails indexed, y <- ys]
        extract (x, y) = (fst x, fst y)


part1 pairs numPoints = product . take 3 . setSizes $ final_forest
    where
        start_forest = (M.fromList (zip [0..numPoints-1] [0..numPoints-1]), numPoints)
        (final_forest, final_size) = L.foldl' union start_forest pairs


part2_stub fs@(f, s) pairs = if ns == 1 then (a, b) else part2_stub nfs rem
    where
        ((a,b):rem) = pairs
        nfs@(nf, ns) =  fs `union` (a, b)

part2 pairs numPoints = part2_stub start_forest pairs
    where 
        start_forest =  (M.fromList (zip [0..numPoints-1] [0..numPoints-1]), numPoints)


main = do
    inp <- input
    let all_pairs = getPairs inp
    print $ part1 (take 1000 all_pairs) (length inp)
    let (a, b) = part2 all_pairs (length inp)
    print $ head (inp !! a) * head (inp !! b)